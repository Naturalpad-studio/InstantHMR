#!/usr/bin/env python3
"""Checks for --prev-token (foot lock) on real data.

    python tools/verify_prev_token.py --data_root data

1. OFF IS BIT-IDENTICAL. With the flag off, dataset samples, the loss and the
   model output equal `git show HEAD`'s, on augmented samples drawn under the
   same seeds. The window corpus is not exercised (HEAD predates window0_p);
   the change does not touch that branch.
2. PAIR GEOMETRY. With the noise at zero, on Harmony4D pairs under flip and
   rotation: previous-pose feet + true displacement == the current GT feet,
   in the augmented frame, to float precision. This is what the zero-velocity
   loss relies on.
3. Contact labels derived on Harmony4D: how many pairs are found, and the
   contact / non-contact / masked split.

Exit status 0 when 1 and 2 pass.
"""
from __future__ import annotations

import argparse
import importlib
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "instanthmr_distill_train"
sys.path.insert(0, str(TRAIN))


def seeded(fn, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return fn()


def same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=str(ROOT / "data"))
    p.add_argument("--n", type=int, default=48)
    args = p.parse_args()

    head_path = TRAIN / "_head_train_distill_mhr_only.py"
    head_path.write_text(subprocess.run(
        ["git", "show", "HEAD:instanthmr_distill_train/train_distill_mhr_only.py"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    try:
        H = importlib.import_module("_head_train_distill_mhr_only")
    finally:
        head_path.unlink()
    import train_distill_mhr_only as T
    ok = True

    # ---- 1. off == HEAD ----------------------------------------------------
    kw = dict(augment=True, per_dataset_caps={"*": 64}, cliff_focal=True,
              crop_centre_fix=True, cliff_follows_aug=True, image_size=224)
    ds_h = H.SAM3DStudentDataset(args.data_root, **kw)
    ds_n = T.SAM3DStudentDataset(args.data_root, temporal=None, **kw)
    idx = random.Random(0).sample(range(len(ds_n)), args.n)
    bad = 0
    for k, i in enumerate(idx):
        a = seeded(lambda: ds_h[i], k)
        b = seeded(lambda: ds_n[i], k)
        if set(a) != set(b) or not all(same(a[x], b[x]) for x in a):
            bad += 1
    print(f"[1] {args.n} augmented samples, flag off vs HEAD: {args.n - bad} identical")
    ok &= bad == 0

    cfg_h, cfg_n = H.DistillConfig(), T.DistillConfig()
    for c, M in ((cfg_h, H), (cfg_n, T)):
        M.apply_rebalanced_losses(c)
        c.cont_head = c.root_rot_loss = True
        c.bound_scales = c.cliff_focal = True
        c.backbone = "hgnetv2_b4"
    dev = T.device
    W = np.load(cfg_n.kp_regressor_path)
    mhr = T.MHRForwardPass(cfg_n.mhr_model_path, dev, kp_regressor=W)
    batch = torch.utils.data.default_collate([seeded(lambda: ds_n[i], 100 + j) for j, i in enumerate(idx[:8])])
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.manual_seed(0); m_h = H.InstantHMRStudent(cfg_h, pretrained=False).to(dev).eval()
    m_n = T.InstantHMRStudent(cfg_n, pretrained=False).to(dev).eval()
    m_n.load_state_dict(m_h.state_dict())
    print(f"    state_dict keys: HEAD {len(m_h.state_dict())}, now {len(m_n.state_dict())}")
    ok &= len(m_h.state_dict()) == len(m_n.state_dict())
    with torch.no_grad():
        o_h = m_h(batch["image"], batch["cliff_cond"])
        o_n = m_n(batch["image"], batch["cliff_cond"])
        o_n2 = m_n(batch["image"], batch["cliff_cond"])
    out_same = all(torch.equal(o_n2[k], o_h[k]) for k in o_h) and set(o_h) == set(o_n)
    print(f"    model outputs identical: {out_same}")
    ok &= out_same
    l_h, l_n = H.DistillationLoss(cfg_h, mhr), T.DistillationLoss(cfg_n, mhr)
    with torch.no_grad():
        for _ in range(2):          # alternate: the first MHR call in a process differs ~5e-10
            a = l_h({k: v.clone() for k, v in o_h.items()}, batch)
            b = l_n({k: v.clone() for k, v in o_h.items()}, batch)
    loss_same = set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    print(f"    loss terms identical: {loss_same} ({len(a)} terms)")
    ok &= loss_same

    # ---- 2./3. pairs on Harmony4D -----------------------------------------
    cfg = T.DistillConfig()
    cfg.prev_token = True
    cfg.prev_mask_p = 0.0
    cfg.prev_noise_joint = cfg.prev_noise_xy = cfg.prev_noise_logz = 0.0
    tk = T.temporal_kwargs(cfg)
    h4d = Path(args.data_root) / "sam3d_gt_harmony4d"
    ds = T.SAM3DStudentDataset(str(h4d.parent), temporal=tk, geom_p=1.0, geom_flip_p=0.5,
                               per_dataset_caps={"*": 0, "sam3d_gt_harmony4d": 3000}, **{
                                   k: v for k, v in kw.items() if k not in ("augment", "per_dataset_caps")})
    worst, n_pair, n_flip, cstat = 0.0, 0, 0, np.zeros(3)
    for k in range(400):
        s = seeded(lambda: ds[k], 1000 + k)
        if not s["has_pair"]:
            continue
        n_pair += 1
        n_flip += int(s["aug_flip"] > 0.5)
        cur = (s["joints_3d"] + s["cam_trans"])[T.FOOT_KP]
        worst = max(worst, float((s["prev_feet"] + s["feet_disp"] - cur).abs().max()))
        cv, c = s["contact_valid"].numpy(), s["contact"].numpy()
        cstat += [(c * cv).sum(), ((1 - c) * cv).sum(), (1 - cv).sum()]
    print(f"[2] {n_pair}/400 Harmony4D samples found their previous frame "
          f"({n_flip} flipped); max |prev + disp - current| = {worst:.2e} m")
    ok &= n_pair > 0 and worst < 1e-5
    tot = cstat.sum()
    print(f"[3] per-foot labels: contact {cstat[0]/tot:.0%}, moving {cstat[1]/tot:.0%}, "
          f"masked {cstat[2]/tot:.0%} (thresholds {cfg.contact_thr*1000:.0f} / "
          f"{cfg.noncontact_thr*1000:.0f} mm per pair)")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
