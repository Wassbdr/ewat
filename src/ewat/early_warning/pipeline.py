"""Online early-warning loop built from the EWAT components.

For every 30 s step ``t`` of a signal ``S(t) ∈ ℝ^{N×17}``:

1. **Baseline**: the first ``window_ref`` steps of the stream define a
   per-(service, feature) baseline; every row is turned into a robust,
   log-compressed z-score against it.
2. **Step 0 — drift gate**: :class:`~ewat.drift.detector.DriftDetector`
   (MMD² with random Fourier features, σ and ε calibrated on normal training
   data) scores the current window against the reference window.
3. **Step 1–2 — typing**: only when MMD² ≥ ε, the last ``type_window`` steps
   go through :class:`~ewat.encoder.stgcn.STGCNEncoder` +
   :class:`~ewat.typing.siamese.SiameseTyper`; the embedding is classified by
   :class:`~ewat.openset.mahalanobis.MahalanobisOOD` into ``normal``,
   ``drift_traffic_ramp`` (harmless) or one of the fault types.
4. **Decision**: a fault type predicted on ``confirm_steps`` consecutive gated
   steps raises an alert. A harmless drift typed on ``absorb_steps``
   consecutive gated steps is absorbed: the MMD reference is reloaded with the
   recent window (the RECALIBRATE regime, W_ref ← W_cur), so a later fault
   shows up as a fresh MMD² spike.

The typer is trained here on seeded synthetic episodes
(:mod:`ewat.synthetic`); no real-cluster data is involved.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ewat.drift.detector import DriftDetector
from ewat.drift.mmd import RFFKernel
from ewat.encoder.stgcn import STGCNEncoder
from ewat.openset.mahalanobis import MahalanobisOOD
from ewat.synthetic.generator import BENIGN_DRIFT, FAULT_TYPES, SERVICES, Episode
from ewat.typing.siamese import ContrastiveLoss, SiameseTyper

CLASSES: list[str] = ["normal", BENIGN_DRIFT, *FAULT_TYPES]
_CLS_IDX = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class PipelineConfig:
    window_ref: int = 10
    window_cur: int = 5
    post_drift: int = 3
    type_window: int = 8
    rff_dim: int = 256
    sigma_scale: float = 0.5
    eps_quantile: float = 0.995
    confirm_steps: int = 2
    absorb_steps: int = 6
    min_type_prob: float = 0.5
    # typer
    d_hidden: int = 32
    d_embed: int = 32
    d_proj: int = 16
    margin: float = 1.0
    train_iters: int = 600
    batch_size: int = 128
    lr: float = 3e-3
    seed: int = 42


@dataclass
class StepResult:
    """Pipeline output for one step."""

    t: int
    mmd2: float
    epsilon: float
    gated: bool  # MMD² ≥ ε → typing was run
    predicted: str | None  # class predicted by the typer (gated steps only)
    probability: float | None
    regime: str  # warmup | normal | benign_drift | absorbed | suspect | alert
    alert: bool  # an alert is active at this step
    alert_type: str | None
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeRun:
    steps: list[StepResult] = field(default_factory=list)

    @property
    def first_alert(self) -> StepResult | None:
        return next((s for s in self.steps if s.alert), None)

    def array(self, name: str) -> np.ndarray:
        return np.array([getattr(s, name) for s in self.steps])


def normalise(signal: np.ndarray, n_ref: int) -> np.ndarray:
    """Robust z-score against the first ``n_ref`` steps, then sign·log1p.

    Uses only past data: the baseline is the warm-up window.
    """
    x = np.asarray(signal, dtype=np.float64)
    ref = x[:n_ref]
    mu = ref.mean(axis=0)
    sd = np.maximum(ref.std(axis=0), 0.05 * np.abs(mu) + 1e-6)
    z = (x - mu) / sd
    return np.sign(z) * np.log1p(np.abs(z))


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class EarlyWarningPipeline:
    """Calibrated drift gate + trained typer. Build with :meth:`fit` or :meth:`load`."""

    def __init__(
        self,
        config: PipelineConfig,
        sigma: float,
        epsilon: float,
        typer: SiameseTyper,
        classifier: MahalanobisOOD,
    ) -> None:
        self.config = config
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        self.typer = typer.eval()
        self.classifier = classifier

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        episodes: list[Episode],
        config: PipelineConfig | None = None,
        log: bool = False,
    ) -> EarlyWarningPipeline:
        cfg = config or PipelineConfig()
        _set_seed(cfg.seed)
        z_all = [normalise(e.signal, cfg.window_ref) for e in episodes]
        sigma, epsilon = calibrate_gate(episodes, z_all, cfg)
        typer, clf = _train_typer(episodes, z_all, cfg, log=log)
        return cls(cfg, sigma, epsilon, typer, clf)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "sigma": self.sigma,
                "epsilon": self.epsilon,
                "typer_state": self.typer.state_dict(),
                "classifier": {
                    "class_means": self.classifier.class_means_,
                    "precision": self.classifier.precision_,
                },
                "classes": CLASSES,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> EarlyWarningPipeline:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = PipelineConfig(**ckpt["config"])
        typer = _build_typer(cfg)
        typer.load_state_dict(ckpt["typer_state"])
        clf = MahalanobisOOD(n_classes=len(CLASSES))
        clf.class_means_ = ckpt["classifier"]["class_means"]
        clf.precision_ = ckpt["classifier"]["precision"]
        clf._n_features = clf.class_means_.shape[1]
        clf.fitted_ = True
        return cls(cfg, ckpt["sigma"], ckpt["epsilon"], typer, clf)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _new_detector(self) -> DriftDetector:
        cfg = self.config
        return DriftDetector(
            kernel=RFFKernel(sigma=self.sigma, rff_dim=cfg.rff_dim, seed=cfg.seed),
            epsilon_drift=self.epsilon,
            window_ref_size=cfg.window_ref,
            window_cur_size=cfg.window_cur,
            post_drift_window_s=cfg.post_drift,
        )

    @torch.no_grad()
    def classify_windows(self, z_windows: np.ndarray, adj_windows: np.ndarray) -> np.ndarray:
        """``(B, W, N, 17)`` normalised windows → ``(B, n_classes)`` probabilities."""
        emb = self.typer.embed(
            torch.from_numpy(z_windows).float(),
            torch.from_numpy(adj_windows).float(),
            lengths=_lengths(z_windows),
        ).numpy()
        proba = self.classifier.predict_proba(emb)[:, : len(CLASSES)]
        return proba / proba.sum(axis=1, keepdims=True)

    def run(self, signal: np.ndarray, adjacency: np.ndarray) -> EpisodeRun:
        """Replay a stream step by step (only past data is used at each step)."""
        cfg = self.config
        T = signal.shape[0]
        if T < cfg.window_ref + cfg.window_cur:
            raise ValueError(f"need at least {cfg.window_ref + cfg.window_cur} steps, got {T}")
        z = normalise(signal, cfg.window_ref).astype(np.float32)
        adj = np.nan_to_num(np.asarray(adjacency, dtype=np.float32))
        det = self._new_detector()
        run = EpisodeRun()
        fault_streak, drift_streak = 0, 0
        last_fault: str | None = None
        alert_type: str | None = None

        for t in range(T):
            t0 = time.perf_counter()
            res = det.update(z[t].astype(np.float64))
            warm = t < cfg.window_ref + cfg.window_cur - 1 or res.mmd2 == 0.0
            gated = (not warm) and res.mmd2 >= self.epsilon
            pred, prob = None, None
            regime = "warmup" if t < cfg.window_ref + cfg.window_cur - 1 else "normal"

            if gated:
                lo = max(0, t - cfg.type_window + 1)
                zw, aw = z[lo : t + 1], adj[lo : t + 1]
                if len(zw) < cfg.type_window:  # left-pad by repeating the first row
                    pad = cfg.type_window - len(zw)
                    zw = np.concatenate([np.repeat(zw[:1], pad, 0), zw])
                    aw = np.concatenate([np.repeat(aw[:1], pad, 0), aw])
                p = self.classify_windows(zw[None], aw[None])[0]
                k = int(p.argmax())
                pred, prob = CLASSES[k], float(p[k])

                if pred in FAULT_TYPES and prob >= cfg.min_type_prob:
                    fault_streak = fault_streak + 1 if pred == last_fault else 1
                    last_fault = pred
                    drift_streak = 0
                    regime = "suspect"
                    if fault_streak >= cfg.confirm_steps and alert_type is None:
                        alert_type = pred
                elif pred == BENIGN_DRIFT:
                    fault_streak, last_fault = 0, None
                    drift_streak += 1
                    regime = "benign_drift"
                    if drift_streak >= cfg.absorb_steps and alert_type is None:
                        # RECALIBRATE: W_ref ← recent window (benign regime accepted)
                        det.reset()
                        recent = z[max(0, t - cfg.window_ref + 1) : t + 1]
                        det.load_reference(recent.reshape(len(recent), -1))
                        drift_streak = 0
                        regime = "absorbed"
                else:
                    fault_streak, last_fault = 0, None
            else:
                fault_streak, last_fault = 0, None

            if alert_type is not None:
                regime = "alert"
            run.steps.append(
                StepResult(
                    t=t,
                    mmd2=float(res.mmd2),
                    epsilon=self.epsilon,
                    gated=bool(gated),
                    predicted=pred,
                    probability=prob,
                    regime=regime,
                    alert=alert_type is not None,
                    alert_type=alert_type,
                    latency_ms=(time.perf_counter() - t0) * 1e3,
                )
            )
        return run


def calibrate_gate(
    episodes: list[Episode], z_all: list[np.ndarray], cfg: PipelineConfig
) -> tuple[float, float]:
    """σ (scaled median heuristic) and ε (quantile of normal-phase MMD²)."""
    normal_rows = np.concatenate(
        [z[cfg.window_ref : e.drift_start].reshape(-1, z[0].size) for z, e in zip(z_all, episodes)]
    )
    median = RFFKernel(rff_dim=cfg.rff_dim, seed=cfg.seed).fit_sigma(normal_rows).sigma
    sigma = cfg.sigma_scale * median
    # Replay the detector on normal phases: ref = warm-up window, cur slides.
    kernel = RFFKernel(sigma=sigma, rff_dim=cfg.rff_dim, seed=cfg.seed)
    mmd_normal: list[float] = []
    for z, e in zip(z_all, episodes):
        flat = z.reshape(len(z), -1)
        ref = flat[: cfg.window_ref]
        for t in range(cfg.window_ref + cfg.window_cur - 1, e.drift_start):
            mmd_normal.append(kernel.mmd_squared(ref, flat[t - cfg.window_cur + 1 : t + 1]))
    return float(sigma), float(np.quantile(mmd_normal, cfg.eps_quantile))


# ----------------------------------------------------------------------
# Typer training
# ----------------------------------------------------------------------


def _lengths(windows: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Typing windows are never padded: every one has its full length."""
    return torch.full((windows.shape[0],), windows.shape[1], dtype=torch.long)


def _build_typer(cfg: PipelineConfig) -> SiameseTyper:
    enc = STGCNEncoder(
        d_feat=17,
        n_nodes=len(SERVICES),
        d_hidden=cfg.d_hidden,
        d_embed=cfg.d_embed,
        use_layer_norm=True,
        use_self_loops=True,
        dropout=0.1,
    )
    return SiameseTyper(enc, d_proj=cfg.d_proj)


# Steps after onset before a fault is visible in the signal. A leak needs a few
# steps of RAM growth; labelling its first windows as "leak" would teach the
# typer that any drifted window is a leak.
_VISIBLE_AFTER = {"resource_leak": 6}


def window_label(e: Episode, t: int) -> str | None:
    """Class of the window ending at ``t``; ``None`` right at a phase boundary."""
    if t < e.drift_start:
        return "normal"
    if e.fault_start is None or t < e.fault_start:
        return BENIGN_DRIFT if t >= e.drift_start + 1 else None
    return e.scenario if t >= e.fault_start + _VISIBLE_AFTER.get(e.scenario, 1) else None


def training_windows(
    episodes: list[Episode], z_all: list[np.ndarray], cfg: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = cfg.type_window
    xs, adjs, ys = [], [], []
    for e, z in zip(episodes, z_all):
        for t in range(cfg.window_ref + W - 1, e.T):
            lab = window_label(e, t)
            if lab is None:
                continue
            xs.append(z[t - W + 1 : t + 1])
            adjs.append(e.adjacency[t - W + 1 : t + 1])
            ys.append(_CLS_IDX[lab])
    return (
        np.stack(xs).astype(np.float32),
        np.stack(adjs).astype(np.float32),
        np.array(ys),
    )


def _train_typer(
    episodes: list[Episode], z_all: list[np.ndarray], cfg: PipelineConfig, log: bool = False
) -> tuple[SiameseTyper, MahalanobisOOD]:
    X, A, y = training_windows(episodes, z_all, cfg)
    rng = np.random.default_rng(cfg.seed)
    _set_seed(cfg.seed)
    typer = _build_typer(cfg)
    opt = torch.optim.AdamW(typer.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = ContrastiveLoss(margin=cfg.margin)
    by_class = [np.where(y == c)[0] for c in range(len(CLASSES))]
    Xt, At = torch.from_numpy(X), torch.from_numpy(A)

    typer.train()
    for it in range(cfg.train_iters):
        # class-balanced anchors, half positive / half negative partners
        cls_a = rng.integers(0, len(CLASSES), cfg.batch_size)
        same = rng.random(cfg.batch_size) < 0.5
        cls_b = np.where(
            same, cls_a, (cls_a + rng.integers(1, len(CLASSES), cfg.batch_size)) % len(CLASSES)
        )
        ia = np.array([rng.choice(by_class[c]) for c in cls_a])
        ib = np.array([rng.choice(by_class[c]) for c in cls_b])
        lens = _lengths(Xt[ia])
        _, _, dist = typer(Xt[ia], At[ia], Xt[ib], At[ib], lengths_i=lens, lengths_j=lens)
        loss = loss_fn(dist, torch.from_numpy(same))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log and (it % 100 == 0 or it == cfg.train_iters - 1):
            print(f"  typer iter {it:4d}  contrastive loss {loss.item():.4f}", flush=True)
    typer.eval()

    with torch.no_grad():
        emb = torch.cat(
            [
                typer.embed(Xt[i : i + 1024], At[i : i + 1024], lengths=_lengths(Xt[i : i + 1024]))
                for i in range(0, len(Xt), 1024)
            ]
        ).numpy()
    clf = MahalanobisOOD(n_classes=len(CLASSES)).fit(emb, y)
    return typer, clf
