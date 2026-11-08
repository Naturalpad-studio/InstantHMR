"""InstantHMR: real-time 3D human mesh recovery from a single image."""
from .inference import HMRPrediction, InstantHMR
from .models import DEFAULT_FIT, DEFAULT_MODEL, MODELS, resolve_model
from .pipeline import FrameResult, PosePipeline
from .skeleton import JOINT_NAMES, SKELETON_EDGES

__all__ = ["InstantHMR", "HMRPrediction", "PosePipeline", "FrameResult", "MODELS", "DEFAULT_MODEL", "DEFAULT_FIT",
           "resolve_model", "JOINT_NAMES", "SKELETON_EDGES"]
__version__ = "1.1.0"
