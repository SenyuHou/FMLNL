from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


BACKBONE_ALIASES = {
    "vit_b16": "vit_b16_imagenet",
    "vit_l16": "vit_l16_imagenet",
}


@dataclass(frozen=True)
class BackboneSpec:
    provider: str
    architecture: str
    checkpoint: str
    pretraining: str


BACKBONE_SPECS = {
    "vit_b16_imagenet": BackboneSpec(
        provider="timm",
        architecture="ViT-B/16",
        checkpoint="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretraining="ImageNet-21k supervised, ImageNet-1k fine-tuned",
    ),
    "vit_l16_imagenet": BackboneSpec(
        provider="timm",
        architecture="ViT-L/16",
        checkpoint="vit_large_patch16_224.augreg_in21k_ft_in1k",
        pretraining="ImageNet-21k supervised, ImageNet-1k fine-tuned",
    ),
    "clip_vit_b16": BackboneSpec(
        provider="open_clip",
        architecture="ViT-B/16",
        checkpoint="ViT-B-16 / openai",
        pretraining="OpenAI vision-language contrastive",
    ),
    "clip_vit_l14": BackboneSpec(
        provider="open_clip",
        architecture="ViT-L/14",
        checkpoint="ViT-L-14 / openai",
        pretraining="OpenAI vision-language contrastive",
    ),
    "dinov2_vit_b14": BackboneSpec(
        provider="timm",
        architecture="ViT-B/14",
        checkpoint="vit_base_patch14_dinov2.lvd142m",
        pretraining="DINOv2 self-supervised (LVD-142M)",
    ),
}


def canonical_backbone_name(name: str) -> str:
    canonical = BACKBONE_ALIASES.get(name, name)
    if canonical not in BACKBONE_SPECS:
        choices = ", ".join(sorted((*BACKBONE_SPECS, *BACKBONE_ALIASES)))
        raise ValueError(f"Unsupported backbone {name!r}. Choose one of: {choices}")
    return canonical


def _spatial_size(value) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    values = tuple(value)
    return values[-2], values[-1]


class FrozenBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.requested_name = model_name
        self.model_name = canonical_backbone_name(model_name)
        self.spec = BACKBONE_SPECS[self.model_name]

        if self.spec.provider == "timm":
            self._load_timm(pretrained)
        else:
            self._load_open_clip(pretrained)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.eval()
        self._print_summary()

    def _load_timm(self, pretrained: bool) -> None:
        try:
            import timm
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as exc:
            raise ImportError("timm is required. Install dependencies from requirements.txt.") from exc

        self.model = timm.create_model(self.spec.checkpoint, pretrained=pretrained, num_classes=0)
        self.preprocess_config = resolve_model_data_config(self.model)
        self.preprocess = create_transform(**self.preprocess_config, is_training=False)
        self.input_size = _spatial_size(self.preprocess_config["input_size"])
        self.feature_dim = int(self.model.num_features)
        self.preprocessing_source = "timm pretrained config"

    def _load_open_clip(self, pretrained: bool) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for CLIP backbones. Install dependencies from requirements.txt."
            ) from exc

        architecture = self.spec.checkpoint.split(" / ", 1)[0]
        pretrained_config = open_clip.get_pretrained_cfg(architecture, "openai")
        model, _, preprocess = open_clip.create_model_and_transforms(
            architecture,
            pretrained=None,
            load_weights=False,
            force_quick_gelu=pretrained_config.get("quick_gelu", False),
            image_mean=pretrained_config["mean"],
            image_std=pretrained_config["std"],
            image_interpolation=pretrained_config["interpolation"],
            image_resize_mode=pretrained_config["resize_mode"],
        )
        if pretrained:
            checkpoint_path = open_clip.download_pretrained(pretrained_config, prefer_hf_hub=False)
            open_clip.load_checkpoint(model, checkpoint_path, weights_only=True)
        self.model = model.visual
        self.preprocess = preprocess
        model_config = open_clip.get_model_config(architecture)
        self.input_size = _spatial_size(model_config["vision_cfg"]["image_size"])
        self.feature_dim = int(model_config["embed_dim"])
        self.preprocess_config = {
            "input_size": (3, *self.input_size),
            "interpolation": pretrained_config["interpolation"],
            "resize_mode": pretrained_config["resize_mode"],
            "mean": pretrained_config["mean"],
            "std": pretrained_config["std"],
        }
        self.preprocessing_source = "open_clip OpenAI pretrained transform"

    def _print_summary(self) -> None:
        print(f"Backbone: {self.model_name}", flush=True)
        print(f"Architecture: {self.spec.architecture}", flush=True)
        print(f"Pretraining: {self.spec.pretraining}", flush=True)
        print(f"Checkpoint: {self.spec.checkpoint}", flush=True)
        print(f"Feature dimension: {self.feature_dim}", flush=True)
        print(f"Input size: {self.input_size[0]}x{self.input_size[1]}", flush=True)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        features = self.model(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if isinstance(features, dict):
            features = features.get("x_norm_clstoken", features.get("features"))
        if features is None or features.ndim != 2:
            shape = None if features is None else tuple(features.shape)
            raise ValueError(f"Backbone must return [B, D] global features, got {shape}.")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Configured feature dimension {self.feature_dim} does not match model output {features.shape[-1]}."
            )
        return features

    forward = forward_features

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self
