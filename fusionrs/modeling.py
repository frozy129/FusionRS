from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, __version__ as transformers_version

ImageInput = Union[str, Path, Image.Image]


def _feature_tensor(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    for name in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value[:, 0] if name == "last_hidden_state" else value

    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value) and value.ndim == 2:
                return value

    raise TypeError(f"Cannot extract a feature tensor from {type(output)!r}")


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _open_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    path = Path(image).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")

    try:
        with Image.open(path) as opened:
            opened.load()
            return opened.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {path}") from exc


class FusionRSEncoder:
    def __init__(
        self,
        model: CLIPModel,
        processor: object,
        device: Union[str, torch.device],
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: Union[str, torch.device, None] = None,
        local_files_only: bool = True,
    ) -> "FusionRSEncoder":
        model_path = Path(model_dir).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {model_path}")

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        major_version = int(transformers_version.split(".", 1)[0])
        processor_backend = {"backend": "pil"} if major_version >= 5 else {"use_fast": False}
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            **processor_backend,
        )
        model = CLIPModel.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        return cls(model=model, processor=processor, device=resolved_device)

    @torch.inference_mode()
    def encode_images(self, images: Sequence[ImageInput]) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required")
        loaded = [_open_image(image) for image in images]
        inputs = self.processor(images=loaded, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        output = self.model.get_image_features(**inputs)
        return _normalize(_feature_tensor(output)).cpu()

    def encode_rgb(self, images: Sequence[ImageInput]) -> torch.Tensor:
        return self.encode_images(images)

    def encode_ir(self, images: Sequence[ImageInput]) -> torch.Tensor:
        return self.encode_images(images)

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("At least one text string is required")
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        output = self.model.get_text_features(**inputs)
        return _normalize(_feature_tensor(output)).cpu()

    def score_pairs(
        self,
        rgb_images: Sequence[ImageInput],
        ir_images: Sequence[ImageInput],
        texts: Sequence[str],
    ) -> dict[str, list[float]]:
        if not (len(rgb_images) == len(ir_images) == len(texts)):
            raise ValueError("RGB images, IR images, and texts must have equal lengths")

        rgb = self.encode_rgb(rgb_images)
        ir = self.encode_ir(ir_images)
        text = self.encode_text(texts)
        return {
            "rgb_text": (rgb * text).sum(dim=-1).tolist(),
            "ir_text": (ir * text).sum(dim=-1).tolist(),
            "rgb_ir": (rgb * ir).sum(dim=-1).tolist(),
        }
