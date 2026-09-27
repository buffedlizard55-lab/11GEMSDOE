"""Detector: a CPU gradient-boosted pixel model with an optional inpainting objective.

Why not a deep net: the whole point of this repository is that a hypothesis must
be *testable this week on two cores and 3 GB*.  The organizer's own reference
solution uses a U-Net (https://github.com/drivendataorg/gems-prize-reference-solution)
and that is a reasonable target for a later run, but a hypothesis that cannot be
measured is not a hypothesis, it is a hope.  Every arm here is the same model with
the same hyper-parameters, so a measured difference is attributable to the thing
that changed.

The inpainting objective (H4) is the one genuinely new lever.  The official task
scores faults that are *missing* from the supplied catalogue, and a standard
binary classifier is explicitly taught "there is no fault here" at every pixel
that is not labelled - including the unmapped continuation of a mapped fault.
``sample_training_pixels`` removes that contradiction: a random contiguous
fraction of each training trace is hidden from the model, and those pixels are
excluded from the negative sample as well, so the model is never penalised for
finding a fault where the cartographer simply stopped drawing.  This is the same
"newly mapped geometry of an existing fault system" the organiser confirmed counts
as a new fault (chrisk-dd, 2026-09-22,
https://community.drivendata.org/t/where-do-you-draw-the-line/11536).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

__all__ = ["DetectorConfig", "sample_training_pixels", "fit_detector", "predict_grid"]


@dataclass(frozen=True)
class DetectorConfig:
    """One configuration, shared by every arm of an experiment."""

    max_iter: int = 200
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 15
    random_state: int = 42
    n_negatives: int = 400_000
    #: fraction of each *training* trace hidden per epoch, H4 only. 0 disables it.
    inpaint_fraction: float = 0.0
    inpaint_repeats: int = 1

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _sample_balanced(cube: np.memmap, positive: np.ndarray, usable: np.ndarray,
                     cfg: DetectorConfig, rng: np.random.Generator):
    """Positives = all usable positive pixels; negatives = a uniform random sample."""
    pos_idx = np.flatnonzero(positive & usable)
    neg_pool = np.flatnonzero(~positive & usable)
    if neg_pool.size == 0:
        raise ValueError("no usable negative pixels")
    n_neg = min(cfg.n_negatives, neg_pool.size)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False)
    return pos_idx, neg_idx


def _hide_trace_segments(cube_shape: tuple[int, int], positive: np.ndarray,
                         fraction: float, rng: np.random.Generator):
    """Hide ``fraction`` of each trace as a contiguous terminal segment.

    A *terminal* segment is used deliberately: the organiser's confirmation that a
    "new fault" may be "newly mapped geometry of an existing fault system" makes
    continuation beyond a mapped end the highest-value target, and hiding a middle
    gap would teach interpolation rather than extrapolation.
    """
    from scipy.ndimage import find_objects, label

    from .heldout import STRUCTURE_8

    lab, n = label(positive, structure=STRUCTURE_8)
    hidden = np.zeros(cube_shape, dtype=bool)
    if n == 0:
        return hidden
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == i)
        if ys.size == 0:
            continue
        # order the trace along its long axis and hide the last `fraction` of it
        if ys.max() - ys.min() >= xs.max() - xs.min():
            key = ys * (lab.shape[1] + 1) + xs
        else:
            key = xs * (lab.shape[0] + 1) + ys
        order = np.argsort(key)
        k = max(1, int(round(fraction * order.size)))
        take = order[-k:]
        hidden[sl[0].start + ys[take], sl[1].start + xs[take]] = True
    return hidden


def sample_training_pixels(cube: np.memmap, positive: np.ndarray, usable: np.ndarray,
                           cfg: DetectorConfig, seed: int = 0, return_indices: bool = False):
    """Return the ``(X, y)`` matrix the classifier is fitted on.

    ``return_indices`` additionally returns the flat pixel indices, which is what
    lets a test assert *where* the model was and was not shown a label.
    """
    rng = np.random.default_rng(seed)
    height, width = usable.shape
    train_pos = positive & usable
    keep_neg = usable.copy()

    if cfg.inpaint_fraction > 0.0:
        pos_parts, neg_parts = [], []
        for rep in range(max(1, cfg.inpaint_repeats)):
            r = np.random.default_rng(seed + 1000 * (rep + 1))
            hidden = _hide_trace_segments((height, width), train_pos, cfg.inpaint_fraction, r)
            # the model never sees the hidden pixels, and is never told "no fault"
            # there either - that is the whole point of the objective
            keep_neg &= ~hidden
            train_pos_r = train_pos & ~hidden
            p, n = _sample_balanced(cube, train_pos_r, keep_neg, cfg, r)
            pos_parts.append(p)
            neg_parts.append(n)
        pos_idx = np.concatenate(pos_parts)
        neg_idx = np.concatenate(neg_parts)
    else:
        pos_idx, neg_idx = _sample_balanced(cube, train_pos, keep_neg, cfg, rng)

    idx = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(pos_idx.size, dtype=np.uint8),
                       np.zeros(neg_idx.size, dtype=np.uint8)])
    order = rng.permutation(idx.size)
    idx, y = idx[order], y[order]
    X = np.asarray(cube[:, idx // width, idx % width].T)   # (n, C) uint8
    return (X, y, idx) if return_indices else (X, y)


def fit_detector(X: np.ndarray, y: np.ndarray, cfg: DetectorConfig) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        early_stopping=cfg.early_stopping,
        validation_fraction=cfg.validation_fraction,
        n_iter_no_change=cfg.n_iter_no_change,
        random_state=cfg.random_state,
    )
    clf.fit(X, y)
    return clf


def predict_grid(clf, cube: np.memmap, usable: np.ndarray,
                 row_block: int = 512) -> np.ndarray:
    """Probability field over the whole grid, float32, 0 outside ``usable``."""
    _, height, width = cube.shape
    out = np.zeros((height, width), dtype=np.float32)
    for r0 in range(0, height, row_block):
        r1 = min(height, r0 + row_block)
        block = np.asarray(cube[:, r0:r1, :]).reshape(cube.shape[0], -1).T
        p = clf.predict_proba(block)[:, 1].astype(np.float32)
        out[r0:r1, :] = p.reshape(r1 - r0, width)
    out[~usable] = 0.0
    return out
