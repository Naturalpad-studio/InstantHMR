"""The in-graph MHR body against the rig it replaces.

    python -m pytest -q training/test_mhr_body.py
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RIG = HERE / "checkpoints" / "mhr_model.pt"
pytestmark = pytest.mark.skipif(not RIG.exists(), reason=f"needs the MHR rig at {RIG}")


@pytest.fixture(scope="module")
def poses():
    """Real poses from a corpus ($INSTANTHMR_DATA, default data/), else random parameters."""
    root = Path(os.environ.get("INSTANTHMR_DATA", HERE.parent / "data"))
    files = sorted(glob.glob(str(root / "*" / "annotations" / "*.npz")))[:16]
    if files:
        p = np.stack([np.load(f)["mhr_model_params"] for f in files]).astype(np.float32)
        s = np.stack([np.load(f)["shape_params"] for f in files]).astype(np.float32)
    else:
        g = np.random.default_rng(0)
        p = (g.standard_normal((16, 204)) * 0.3).astype(np.float32)
        s = g.standard_normal((16, 45)).astype(np.float32)
    p[:, :3] = 0
    return torch.from_numpy(p), torch.from_numpy(s)


@pytest.fixture(scope="module")
def body():
    from mhr_body import MHRBody
    return MHRBody.from_rig(RIG).eval()


def test_keypoints_match_the_training_readout(body, poses):
    import train as T
    ref = T.MHRForwardPass(str(RIG), "cpu", landmark_assets=str(HERE / "assets" / "mhr_landmarks70.npz"))
    with torch.no_grad():
        want = ref.get_native_keypoints(*poses, exact=True)
        got, _ = body(*poses)
    assert float((got - want).abs().max()) < 1e-5            # metres: 0.01 mm


def test_vertices_match_the_rig_without_pose_correctives(body, poses):
    rig = torch.jit.load(str(RIG), map_location="cpu").eval()
    p, s = poses
    with torch.no_grad():
        v, _ = rig(s, p, torch.zeros(len(p), 72), False)
        want = torch.stack([v[..., 0], -v[..., 1], -v[..., 2]], -1) / 100.0
        _, got = body(p, s)
    assert got.shape == want.shape == (len(p), 18439, 3)
    assert float((got - want).abs().max()) < 1e-5
