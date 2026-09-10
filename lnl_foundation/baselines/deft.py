"""DeFT phase-one noisy-label detector adapted to FMLNL's CIFAR labels.

The training equations and PEFT configuration follow the authors'
``main_phase1.py`` and ``main_real_phase1.py`` implementations. Synthetic
noise uses the former; Human noise uses the latter's real-noise rule.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from lnl_foundation.baselines.vendor.deft import Model
from lnl_foundation.baselines.vendor.deft.clip.model import build_model
from lnl_foundation.utils import set_seed


ARCHITECTURES = {
    "clip_vit_b16": "ViT-B-16",
    "clip_vit_l14": "ViT-L-14",
}
DEFT_ARCHITECTURES = {
    "clip_vit_b16": "ViT-B/16",
    "clip_vit_l14": "ViT-L/14",
}


@dataclass(frozen=True)
class DeFTConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.03
    weight_decay: float = 5e-4
    momentum: float = 0.9
    vpt_len: int = 20
    n_ctx: int = 16
    synthetic_warmup: int = 1
    human_warmup: int = 5


class _NoisyCifar(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        self.num_classes = int(self.labels.max()) + 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.transform(Image.fromarray(self.images[index]))
        target = int(self.labels[index])
        complement = np.random.choice(self.num_classes)
        while complement == target:
            complement = np.random.choice(self.num_classes)
        return image, target, complement, index


def _train_transform(dataset):
    statistics = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "cifar100": ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    }
    mean, std = statistics[dataset]
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _class_names(dataset):
    path = Path(__file__).with_name("deft_class_names.json")
    return json.loads(path.read_text(encoding="ascii"))[dataset]


def _model_config(dataset, backbone, classes, config):
    return SimpleNamespace(
        backbone=DEFT_ARCHITECTURES[backbone],
        num_class=classes,
        class_names=_class_names(dataset),
        prec="fp16",
        finetune=False,
        bias_tuning=False,
        vpt_shallow=False,
        vpt_deep=True,
        vpt_len=config.vpt_len,
        adapter=False,
        adapter_dim=0,
        lora=False,
        lora_dim=0,
        ssf=False,
        partial=None,
        N_CTX=config.n_ctx,
        CLASS_TOKEN_POSITION="end",
    )


def _load_model(dataset, backbone, classes, device, config):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("DeFT requires open_clip_torch to obtain OpenAI CLIP weights.") from exc
    architecture = ARCHITECTURES[backbone]
    pretrained = open_clip.get_pretrained_cfg(architecture, "openai")
    checkpoint = open_clip.download_pretrained(pretrained, prefer_hf_hub=False)
    try:
        archive = torch.jit.load(checkpoint, map_location="cpu").eval()
        state_dict = archive.state_dict()
    except RuntimeError:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    clip_model = build_model(state_dict).to(device)
    model = Model(_model_config(dataset, backbone, classes, config), clip_model).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.tuner.parameters():
        parameter.requires_grad_(True)
    for parameter in model.prompt_learner.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.SGD(
        list(model.tuner.parameters()) + list(model.prompt_learner.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )
    return model, optimizer, str(checkpoint)


def _sce(logits, labels, classes):
    ce = F.cross_entropy(logits, labels, reduction="none")
    probabilities = F.softmax(logits, dim=1).clamp(1e-7, 1.0)
    targets = F.one_hot(labels, classes).float().clamp(1e-4, 1.0)
    rce = -(probabilities * targets.log()).sum(dim=1)
    return ce + rce


def run_deft(images, noisy_labels, dataset, backbone, seed, device,
             num_workers=4, config=None, human=False):
    """Train DeFT phase one and return its final-epoch clean selection."""
    if device.type != "cuda":
        raise ValueError("The original fp16 DeFT phase-one implementation requires CUDA.")
    config = config or DeFTConfig()
    labels = np.asarray(noisy_labels, dtype=np.int64)
    classes = int(labels.max()) + 1
    set_seed(seed)
    model, optimizer, checkpoint = _load_model(
        dataset, backbone, classes, device, config
    )
    data = _NoisyCifar(images, labels, _train_transform(dataset))
    loader = DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    clean_prediction = torch.zeros(len(data), dtype=torch.bool)
    clean_score = torch.zeros(len(data), dtype=torch.float32)
    warmup = config.human_warmup if human else config.synthetic_warmup

    for epoch in range(1, config.epochs + 1):
        model.train()
        progress = tqdm(loader, desc=f"DeFT seed={seed} epoch {epoch}/{config.epochs}")
        for inputs, targets, complements, indices in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            complements = complements.to(device, non_blocking=True)
            positive, negative = model(inputs, return_neg=True)
            predicted = positive.argmax(1)
            paired = torch.stack([positive.detach(), negative], dim=-1).softmax(dim=-1)
            p_yes, p_no = paired[:, :, 0], paired[:, :, 1]
            p_clean = p_yes[torch.arange(len(targets), device=device), targets]
            selected = p_clean > 0.5
            if human:
                selected &= p_yes.argmax(1) == targets
            if epoch == config.epochs:
                clean_prediction[indices] = selected.detach().cpu()
                clean_score[indices] = p_clean.detach().float().cpu()

            classification = _sce(positive, targets, classes) if human and epoch <= warmup \
                else F.cross_entropy(positive, targets, reduction="none")
            if epoch <= warmup:
                loss_classification = classification.mean()
            else:
                loss_classification = classification[selected].mean()
            log_no = p_no.clamp(1e-5, 1.0).log()
            log_yes = p_yes.clamp(1e-5, 1.0).log()
            pseudo = targets.clone()
            if (human and epoch > 1) or (not human and epoch > warmup):
                pseudo[~selected] = predicted[~selected]
            loss_negative = F.nll_loss(log_no, complements) + F.nll_loss(log_yes, pseudo)
            loss = loss_classification + loss_negative
            if not torch.isfinite(loss):
                raise ValueError("DeFT produced a non-finite training loss.")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress.set_postfix(loss=f"{loss.item():.4f}", clean=int(selected.sum()))
        scheduler.step()

    result = {
        "predicted_clean": clean_prediction.numpy(),
        "noise_score": 1.0 - clean_score.numpy(),
        "clean_score": clean_score.numpy(),
        "checkpoint": checkpoint,
        "warmup": warmup,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result
