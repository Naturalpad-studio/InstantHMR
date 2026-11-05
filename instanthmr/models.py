"""The released InstantHMR models and their download from the Hugging Face Hub."""
from __future__ import annotations

from pathlib import Path

REPO_ID = "momolesang/InstantHMR"

MODELS = {
    "nano": "instanthmr-nano.onnx",
    "tiny": "instanthmr-tiny.onnx",
    "small": "instanthmr-small.onnx",
    "medium": "instanthmr-medium.onnx",
    "medium-commercial": "instanthmr-medium-commercial.onnx",
}
DEFAULT_MODEL = "medium"
BODY = "mhr-body.onnx"


def _download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, filename))


def resolve_model(model: str | Path) -> Path:
    """A local `.onnx` path, or a model name downloaded once into the Hugging Face cache."""
    path = Path(model)
    if path.suffix == ".onnx":
        if not path.exists():
            raise FileNotFoundError(f"model not found: {path}")
        return path
    if str(model) not in MODELS:
        raise ValueError(f"unknown model {model!r}: choose one of {', '.join(MODELS)} or pass an .onnx path")
    return _download(MODELS[str(model)])


def resolve_body(model_path: Path) -> Path:
    """The MHR body graph: next to a local model if there is one, else downloaded."""
    local = Path(model_path).with_name(BODY)
    return local if local.exists() else _download(BODY)
