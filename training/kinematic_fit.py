"""MHR kinematic fit as a static, export-ready PyTorch module (ONNX -> TensorRT / ORT CUDA / ORT QNN, fp16).

Used by ``tools/export_onnx.py --fit``: the network's MHR pose is refined so that its 30 body keypoints, projected
into the frame, land on the network's own 2D (SimCC) keypoints, weighted by their spread.

The pose (root rotation + 62 body angles, MHR parameters 3:68) and the camera translation are moved so that the
30 projected body keypoints land on given 2D targets, weighted by their spread, with a quadratic prior pulling
towards the network's prediction. Levenberg-Marquardt, a fixed number of iterations, no data-dependent control
flow:

* forward kinematics as in the MHR rig (prefix-multiplication levels, quaternions), from constants read once out
  of ``checkpoints/mhr_model.pt``;
* an exact Jacobian: parameters 3:68 drive only joint rotations, so the derivative of a joint position along a
  rotation axis w of an ancestor j is w x (p - p_j); keypoints are a fixed linear map of joint positions;
* the 68x68 normal equations solved by Newton-Schulz iterations (matrix products only: no solve / inverse
  operator exists in ONNX, and phone NPUs are slow on long chains of tiny operations).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

BODY_KP = list(range(21)) + [41, 62] + list(range(63, 70))
POSE = slice(3, 68)
N_POSE = 65
LN2 = math.log(2.0)


def _qmul(a, b):
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack((w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2), -1)


def _cross(a, b):
    ax, ay, az = a.unbind(-1)
    bx, by, bz = b.unbind(-1)
    return torch.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), -1)


def _rotate(q, v):
    u, w = q[..., :3], q[..., 3:]
    uv = _cross(u, v)
    return v + 2 * (uv * w + _cross(u, uv))


def _normalize(q):
    return q / torch.sqrt((q * q).sum(-1, keepdim=True)).clamp_min(1e-6)


def _half_angle_quats(e):
    """Euler XYZ (roll, pitch, yaw) -> the three elementary quaternions qx, qy, qz (x, y, z, w)."""
    h = e * 0.5
    c, s = torch.cos(h), torch.sin(h)
    z = c[..., 0] * 0
    qx = torch.stack((s[..., 0], z, z, c[..., 0]), -1)
    qy = torch.stack((z, s[..., 1], z, c[..., 1]), -1)
    qz = torch.stack((z, z, s[..., 2], c[..., 2]), -1)
    return qx, qy, qz


def rig_constants(mhr_path: str | Path, kp_regressor_path: str | Path) -> dict:
    """Read the TorchScript rig once: what forward kinematics and the keypoint readout need."""
    rig = torch.jit.load(str(mhr_path), map_location="cpu").eval()
    ct = rig.character_torch
    sk = ct.skeleton
    parents = sk.joint_parents.numpy().astype(np.int64)
    chains = []
    for j in range(len(parents)):
        c = [j]
        while parents[c[-1]] >= 0:
            c.append(int(parents[c[-1]]))
        chains.append(c[::-1])
    partners, actives, level = [], [], 0
    while True:                                   # pymomentum's prefix-multiplication levels
        partner, active = np.arange(len(parents)), np.zeros(len(parents), bool)
        for c in chains:
            i = len(c) - 1
            if (i >> level) & 1:
                partner[c[i]] = c[((i >> level) << level) - 1]
                active[c[i]] = True
        if not active.any():
            break
        partners.append(partner); actives.append(active); level += 1
    desc = np.zeros((len(parents), len(parents)), np.float32)   # desc[j, k] = 1 if k is a strict descendant of j
    for k, c in enumerate(chains):
        for j in c[:-1]:
            desc[j, k] = 1.0
    return dict(parameter_transform=ct.parameter_transform.parameter_transform[:, :204].numpy(),
                joint_offsets=sk.joint_translation_offsets.numpy(), joint_prerotations=sk.joint_prerotations.numpy(),
                fk_partner=np.stack(partners), fk_active=np.stack(actives), joint_parents=parents, descendants=desc,
                kp_regressor=np.load(kp_regressor_path).astype(np.float32))


class MHRKinematics(nn.Module):
    """Forward kinematics of the MHR skeleton (raw units: cm, Y-up) and the 30 body keypoints (vision frame)."""

    def __init__(self, constants: dict):
        super().__init__()
        P = torch.as_tensor(constants["parameter_transform"], dtype=torch.float32)       # (889, 204)
        n_j = P.shape[0] // 7
        rot = P.reshape(n_j, 7, 204)[:, 3:6, POSE]                                         # (127, 3, 65)
        self.register_buffer("P", P)
        self.register_buffer("P_rot", rot.reshape(n_j * 3, N_POSE))
        self.register_buffer("offsets", torch.as_tensor(constants["joint_offsets"], dtype=torch.float32))
        self.register_buffer("prerot", torch.as_tensor(constants["joint_prerotations"], dtype=torch.float32))
        self.register_buffer("partner", torch.as_tensor(constants["fk_partner"], dtype=torch.long))
        self.register_buffer("active", torch.as_tensor(constants["fk_active"], dtype=torch.bool))
        A = torch.as_tensor(constants["kp_regressor"], dtype=torch.float32)[BODY_KP]        # (30, 127)
        D = torch.as_tensor(constants["descendants"], dtype=torch.float32)                  # (127, 127)
        self.register_buffer("A", A)
        self.register_buffer("AD", (A[:, None, :] * D[None]).reshape(-1, n_j))               # (30*127, 127)
        self.register_buffer("AD_sum", (A[:, None, :] * D[None]).sum(-1))                   # (30, 127)
        self.register_buffer("vision", torch.tensor([0.01, -0.01, -0.01]))
        self.register_buffer("axes", torch.eye(3))                                         # e_x, e_y, e_z
        self.register_buffer("conj", torch.tensor([-1.0, -1.0, -1.0, 1.0]))
        self.n_j = n_j

    @classmethod
    def from_rig(cls, mhr_path: str | Path, kp_regressor_path: str | Path) -> "MHRKinematics":
        return cls(rig_constants(mhr_path, kp_regressor_path))

    def forward(self, mp):
        """(B, 204) -> joint positions (B, 127, 3) raw, global quaternions (B, 127, 4), local Euler quats."""
        jp = (mp @ self.P.T).unflatten(-1, (self.n_j, 7))
        qx, qy, qz = _half_angle_quats(jp[..., 3:6])
        q = _qmul(self.prerot, _qmul(qz, _qmul(qy, qx)))
        t, s = jp[..., :3] + self.offsets, torch.exp(LN2 * jp[..., 6:])
        for lvl in range(self.partner.shape[0]):                # global = parent_global o local, by levels
            idx = self.partner[lvl]
            tp, qp, sp = t.index_select(-2, idx), _normalize(q.index_select(-2, idx)), s.index_select(-2, idx)
            tn, qn, sn = tp + sp * _rotate(qp, t), _qmul(qp, _normalize(q)), sp * s
            a = self.active[lvl][:, None]
            t, q, s = torch.where(a, tn, t), torch.where(a, qn, q), torch.where(a, sn, s)
        return t, _normalize(q), qx, qy

    def keypoints(self, t):
        """(B, 127, 3) raw joint positions -> (B, 30, 3) body keypoints, vision frame (metres, Y-down)."""
        return (self.A @ t) * self.vision

    def keypoint_jacobian(self, t, q, qx, qy):
        """d(30 body keypoints, vision frame) / d(MHR params 3:68): (B, 30, 3, 65)."""
        B = t.shape[0]
        ex, ey, ez = (self.axes[i].expand_as(t) for i in range(3))
        cx, cy = qx * self.conj, qy * self.conj                    # conjugates
        wx = _rotate(q, ex)                                         # world axes of the three Euler rotations
        wy = _rotate(q, _rotate(cx, ey))
        wz = _rotate(q, _rotate(cx, _rotate(cy, ez)))
        w = torch.stack((wx, wy, wz), 2)                            # (B, 127, 3 axes, 3)
        S = (self.AD @ t).reshape(B, -1, self.n_j, 3)               # (B, 30, 127, 3): sum_k A D p_k
        lever = S - self.AD_sum[None, :, :, None] * t[:, None]      # (B, 30, 127, 3)
        M = _cross(w[:, None], lever[:, :, :, None])                # (B, 30, 127, 3 axes, 3 coords)
        M = M.permute(0, 1, 4, 2, 3).reshape(B, -1, self.n_j * 3)   # (B, 30*3, 127*3)
        J = (M @ self.P_rot).reshape(B, -1, 3, N_POSE)              # (B, 30, 3, 65)
        return J * self.vision[None, None, :, None]


def newton_schulz_solve(H, g, iters: int, eye):
    """x = H^-1 g for SPD H (B, n, n), by Jacobi scaling and Newton-Schulz on the inverse."""
    d = torch.rsqrt((H * eye).sum(-1).clamp_min(1e-12))                         # (B, n): 1 / sqrt(diag H)
    Hs = H * d[:, :, None] * d[:, None, :]
    X = eye / Hs.abs().sum(-1).amax(-1)[:, None, None]                           # ||I - X0 Hs|| < 1
    for _ in range(iters):
        X = X @ (2 * eye - Hs @ X)
    return d * (X @ (d * g)[..., None])[..., 0]


class KinematicFit(nn.Module):
    """Levenberg-Marquardt on (MHR params 3:68, camera translation), fixed iterations, export-ready."""

    def __init__(self, kin: MHRKinematics, lam_pose: float = 30.0, lam_t: float = 1e-2, iters: int = 1,
                 ns_iters: int = 10, damping: float = 1e-3):
        super().__init__()
        self.kin = kin
        self.register_buffer("prior", torch.cat([torch.full((N_POSE,), lam_pose), torch.full((3,), lam_t)]))
        self.register_buffer("eye", torch.eye(N_POSE + 3)[None])
        self.register_buffer("zero", torch.zeros(1))
        self.iters, self.ns_iters, self.damping = iters, ns_iters, damping

    def forward(self, mp, t, target, sigma, cam):
        """mp (B, 204), t (B, 3), target / sigma (B, 30, 2), cam (B, 3) = (focal, cx, cy), all in one unit.

        Residuals and Jacobian are scaled by 1/16 and the prior by 1/256 (the same problem): with a 1 % of the crop
        sigma floor the weighted Jacobian reaches ~100 per entry, and J^T J would overflow fp16 (65504) unscaled."""
        inv_sigma = (1.0 / 16.0) / sigma
        x0 = torch.cat([mp[:, POSE], t], 1)
        x, mu = x0, x0[:, :1] * 0 + self.damping
        prior, eye = self.prior * (1.0 / 256.0), self.eye
        for i in range(self.iters):
            m = torch.cat([mp[:, :3], x[:, :N_POSE], mp[:, 68:]], 1)
            pos, q, qx, qy = self.kin(m)
            X = self.kin.keypoints(pos) + x[:, None, N_POSE:]
            z = X[..., 2:].clamp_min(1e-3)
            f = cam[:, None, :1]
            uv = f * X[..., :2] / z + cam[:, None, 1:]
            r = ((uv - target) * inv_sigma).flatten(1)
            Jk = self.kin.keypoint_jacobian(pos, q, qx, qy)                       # (B, 30, 3, 65)
            zero = z * 0                                                          # d(u, v) / d(X, Y, Z)
            du = torch.stack((f / z, zero, -f * X[..., :1] / z ** 2), -1)[..., 0, :]
            dv = torch.stack((zero, f / z, -f * X[..., 1:2] / z ** 2), -1)[..., 0, :]
            Jp = torch.stack((du, dv), 2)                                         # (B, 30, 2, 3)
            Jx = (torch.cat([Jp @ Jk, Jp], -1) * inv_sigma[..., None]).flatten(1, 2)   # (B, 60, 68)
            JtJ = Jx.transpose(1, 2) @ Jx
            H = JtJ + eye * (prior + mu * (JtJ * eye).sum(-1))[:, None, :]
            g = (Jx.transpose(1, 2) @ r[..., None])[..., 0] + prior * (x - x0)
            step = newton_schulz_solve(H, g, self.ns_iters, eye)
            x = x - torch.where(step == step, step, self.zero)                    # a NaN step is dropped
        return torch.cat([mp[:, :3], x[:, :N_POSE], mp[:, 68:]], 1), x[:, N_POSE:]


class FitHead(nn.Module):
    """One person's network outputs on ``n_crops`` crops -> fused 2D keypoints -> KinematicFit.

    Crop 0 is the main crop (its MHR parameters and camera translation start the fit). With two crops, crop 1 is
    the zoomed-in one: each body keypoint's 2D is the inverse-variance mean of the two SimCC estimates, keypoints
    outside crop 1 using crop 0 alone. ``crop_box`` (n_crops, 4) is each crop's pixel square (x1, y1, w, h) in the
    full image, i.e. what the 288 px input was resized from; ``cam`` (1, 3) is (focal, cx, cy).
    """

    def __init__(self, kin: MHRKinematics, bin_centers: torch.Tensor, n_crops: int, sigma_floor: float = 0.01,
                 **fit_kwargs):
        super().__init__()
        self.fit = KinematicFit(kin, **fit_kwargs)
        self.register_buffer("centers", bin_centers.detach().float().clone())
        self.register_buffer("body", torch.as_tensor(BODY_KP))
        self.n_crops, self.sigma_floor = n_crops, sigma_floor

    def forward(self, mhr_params, cam_trans, logits, crop_box, cam):
        """Returns the fitted (1, 204) parameters and (1, 3) translation, the fused (1, 70, 2) 2D in units of crop 0
        (origin at its centre), and crop 0's centre (1, 2) and side (1, 1) in pixels: ``pixels = c0 + 2D * s0``.

        Every 2D quantity is expressed in crop-0 units so that it stays near 1 whatever the frame resolution: in
        pixels, a broad SimCC on a 2000 px close-up crop gives sigma^2 ~ 1e5, beyond fp16 (65504), and absolute
        coordinates near 4000 px keep only 2 px of fp16 precision."""
        prob = torch.softmax(logits, -1)                                       # (N, 70, 2, bins)
        mu = (prob * self.centers).sum(-1)                                     # crop units, [-1, 1] = the crop
        sd = torch.sqrt((prob * (self.centers - mu[..., None]) ** 2).sum(-1).clamp_min(1e-6))
        s0 = crop_box[:1, 2:3]
        c0 = crop_box[:1, :2] + 0.5 * crop_box[:1, 2:]
        rel = (crop_box[:, None, 2:] / s0)                                     # (N, 1, 2) crop size / crop 0's
        off = (crop_box[:, None, :2] + 0.5 * crop_box[:, None, 2:] - c0) / s0  # (N, 1, 2) crop centre, crop-0 units
        xy = off + mu * 0.5 * rel                                              # (N, 70, 2)
        sig = torch.clamp(sd, min=2 * self.sigma_floor) * 0.5 * rel            # floor: sigma_floor of the crop side
        if self.n_crops == 2:
            inside = (mu[1].abs() <= 1).all(-1, keepdim=True).to(xy.dtype)    # keypoint within the zoomed crop
            r2 = (sig[0] / sig[1]) ** 2 * inside                               # w1 / w0
            fused = xy[0] + (r2 / (1 + r2)) * (xy[1] - xy[0])                  # inverse-variance mean
            sigma = sig[0] * torch.rsqrt(1 + r2)
        else:
            fused, sigma = xy[0], sig[0]
        cam_n = torch.cat([cam[:, :1] / s0, (cam[:, 1:] - c0) / s0], 1)
        mp, t = self.fit(mhr_params[:1], cam_trans[:1], fused[None].index_select(1, self.body),
                         sigma[None].index_select(1, self.body), cam_n)
        return mp, t, fused[None], c0, s0
