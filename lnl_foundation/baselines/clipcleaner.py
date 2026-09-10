"""CLIPCleaner initial sample selection on FMLNL's cached CLIP features.

This ports ``combined_selection`` from the authors' implementation: class-
descriptor zero-shot predictions and balanced visual logistic regression are
each evaluated by per-class loss/GMM and label-consistency rules, then the
four selected sets are intersected.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture


ARCHITECTURES = {
    "clip_vit_b16": "ViT-B-16",
    "clip_vit_l14": "ViT-L-14",
}


def _metadata(dataset):
    path = Path(__file__).with_name("clipcleaner_prompts.json")
    return json.loads(path.read_text(encoding="ascii"))[dataset]


def _score(prediction, labels, mode):
    prediction = torch.as_tensor(prediction)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if mode in {"celoss", "perclass_celoss"}:
        score = prediction.log().gather(1, labels[:, None]).squeeze(1)
        if mode == "perclass_celoss":
            for class_id in range(int(labels.max()) + 1):
                index = torch.where(labels == class_id)[0]
                values = score[index]
                score[index] = (values - values.min()) / (values.max() - values.min())
        return score
    label_probability = prediction.gather(1, labels[:, None]).squeeze(1)
    return label_probability / prediction.max(dim=1).values


def _load_text_model(backbone, device):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("CLIPCleaner requires open_clip_torch.") from exc
    architecture = ARCHITECTURES[backbone]
    pretrained = open_clip.get_pretrained_cfg(architecture, "openai")
    previous_logging_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            architecture,
            pretrained=None,
            load_weights=False,
            force_quick_gelu=pretrained.get("quick_gelu", False),
            image_mean=pretrained["mean"],
            image_std=pretrained["std"],
            image_interpolation=pretrained["interpolation"],
            image_resize_mode=pretrained["resize_mode"],
        )
    finally:
        logging.disable(previous_logging_level)
    checkpoint = open_clip.download_pretrained(pretrained, prefer_hf_hub=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*torch.load.*TorchScript archive.*")
        open_clip.load_checkpoint(model, checkpoint, weights_only=True)
    return model.eval().to(device), open_clip.get_tokenizer(architecture), checkpoint


@torch.no_grad()
def zero_shot_prediction(features, dataset, backbone, device):
    """Reproduce CLIPCleaner's descriptor-expanded zero-shot probabilities."""
    metadata = _metadata(dataset)
    model, tokenizer, checkpoint = _load_text_model(backbone, device)
    features = F.normalize(features.float(), dim=1).to(device)
    columns = []
    for class_name, descriptors in zip(metadata["class_names"], metadata["detailed_features"]):
        prompts = [f"a photo of a {class_name.lower()} {descriptor}" + metadata["suffix"]
                   for descriptor in descriptors]
        text = F.normalize(model.encode_text(tokenizer(prompts).to(device)).float(), dim=1)
        columns.append(torch.exp((features @ text.T) / 0.07).sum(1))
    similarity = torch.stack(columns, dim=1)
    prediction = (similarity / similarity.sum(1, keepdim=True)).cpu()
    if not torch.isfinite(prediction).all():
        raise ValueError("CLIPCleaner zero-shot prediction contains NaN or Inf.")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction, str(checkpoint)


class CLIPCleaner:
    """The four-way intersection used by CLIPCleaner's initial cleaner."""

    def __init__(self, theta_gmm=0.5, theta_cons=0.8):
        self.theta_gmm = float(theta_gmm)
        self.theta_cons = float(theta_cons)

    def detect(self, features, noisy_labels, prediction_zero):
        labels = np.asarray(noisy_labels, dtype=np.int64)
        classes = int(labels.max()) + 1
        classifier = LogisticRegression(
            random_state=0, max_iter=10000, class_weight="balanced"
        ).fit(features.cpu().numpy(), labels)
        prediction_lr = torch.from_numpy(classifier.predict_proba(features.cpu().numpy()))
        scores = [
            _score(prediction_zero, labels, "perclass_celoss"),
            _score(prediction_zero, labels, "consistency"),
            _score(prediction_lr, labels, "perclass_celoss"),
            _score(prediction_lr, labels, "consistency"),
        ]
        kinds = ("loss", "consistency", "loss", "consistency")
        by_label = [np.where(labels == class_id)[0] for class_id in range(classes)]
        selections, confidences = [], []
        for score, kind in zip(scores, kinds):
            score = score.numpy()
            selected_by_class = []
            confidence = np.zeros(len(labels), dtype=np.float64)
            for index in by_label:
                if kind == "loss":
                    gmm = GaussianMixture(2)
                    values = score[index].reshape(-1, 1)
                    gmm.fit(values)
                    clean_component = int(gmm.means_.argmax())
                    confidence[index] = gmm.predict_proba(values)[:, clean_component]
                    selected_by_class.append(index[confidence[index] >= self.theta_gmm])
                else:
                    confidence[index] = score[index]
                    selected_by_class.append(index[score[index] >= self.theta_cons])
            selections.append(np.concatenate(selected_by_class))
            confidences.append(confidence)

        selected = selections[0]
        for candidate in selections[1:]:
            selected = np.intersect1d(selected, candidate)

        # Preserve the upstream empty-class safeguard, including its rank order.
        counts = np.asarray([(labels[selected] == class_id).sum() for class_id in range(classes)])
        empty_classes = np.where(counts == 0)[0]
        if len(empty_classes):
            non_empty = counts[counts != 0]
            smallest = int(len(labels) / classes / classes)
            if len(non_empty):
                smallest = max(int(non_empty.min()), smallest)
            for class_id in empty_classes:
                index = by_label[class_id]
                rank = scores[0].numpy()[index].argsort()
                selected = np.concatenate([selected, np.asarray(index[rank[:smallest]])])
        selected = np.unique(selected)

        predicted_clean = np.zeros(len(labels), dtype=bool)
        predicted_clean[selected] = True
        margins = []
        for confidence, kind in zip(confidences, kinds):
            threshold = self.theta_gmm if kind == "loss" else self.theta_cons
            margins.append(confidence - threshold)
        # The minimum margin is a continuous relaxation of the exact intersection.
        noise_score = -np.stack(margins).min(axis=0)
        if not np.isfinite(noise_score).all():
            raise ValueError("CLIPCleaner score contains NaN or Inf.")
        return {
            "predicted_clean": predicted_clean,
            "noise_score": noise_score,
            "component_clean_confidence": np.stack(confidences, axis=1),
        }
