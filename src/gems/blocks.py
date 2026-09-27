"""Spatially blocked cross-validation.

WHY THIS EXISTS (and why a random split is not acceptable here)
----------------------------------------------------------------
A fault trace is kilometres long and the competition grid is 100 m, so a random
pixel split puts the same physical structure on both sides of the split.  A
model is then scored on pixels it effectively trained on, and the resulting
number is not a generalisation estimate.  The official task is explicitly about
faults that are *not* in the supplied catalogue, so the validation split has to
be spatial.

DESIGN
------
* The grid is cut into contiguous ``block_px`` squares ("super-regions").
* Blocks are assigned to folds in a deterministic, geometry-stratified way so
  every fold gets a comparable share of the grid and of the catalogue.
* A ``collar_px`` ring of blocks around each held-out block is removed from
  training.  ``collar_px`` defaults to 3 = the metric's R at 100 m, so a model
  cannot be handed the neighbourhood it is scored on.
* ``scored_mask`` restricts *evaluation* to held-out blocks while leaving the
  distance geometry of the full truth raster intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_BLOCK_PX",
    "DEFAULT_COLLAR_PX",
    "DEFAULT_N_FOLDS",
    "FoldPlan",
    "make_fold_plan",
]

DEFAULT_BLOCK_PX = 512      # 51.2 km at 100 m
DEFAULT_COLLAR_PX = 3       # the metric's R, in pixels
DEFAULT_N_FOLDS = 4


@dataclass(frozen=True)
class FoldPlan:
    """Deterministic block->fold assignment for one grid shape."""

    height: int
    width: int
    block_px: int
    collar_px: int
    n_folds: int
    fold_of_block: np.ndarray   # (n_by, n_bx) int8, -1 for blocks past the edge
    block_row0: int
    block_col0: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def block_index_map(self) -> np.ndarray:
        """Per-pixel block index, -1 outside any block.

        The right and bottom edges of the last block row/column are clipped to the
        grid; the contest grid (3730 x 3292 px) is not a multiple of 512.
        """
        nb_y = self.fold_of_block.shape[0]
        nb_x = self.fold_of_block.shape[1]
        rows = min(self.block_row0 + nb_y * self.block_px, self.height) - self.block_row0
        cols = min(self.block_col0 + nb_x * self.block_px, self.width) - self.block_col0
        block_ids = np.arange(nb_y * nb_x, dtype=np.int32).reshape(nb_y, nb_x)
        full = np.repeat(np.repeat(block_ids, self.block_px, axis=0), self.block_px, axis=1)
        idx = np.full((self.height, self.width), -1, dtype=np.int32)
        idx[self.block_row0:self.block_row0 + rows,
            self.block_col0:self.block_col0 + cols] = full[:rows, :cols]
        return idx

    def fold_of_pixel(self) -> np.ndarray:
        """Per-pixel fold id, -1 where no block covers the pixel."""
        return self.fold_of_block.ravel()[self.block_index_map()]

    def held_out_mask(self, fold: int, apply_collar: bool = True) -> np.ndarray:
        """Pixels scored in ``fold`` (optionally minus the training collar)."""
        fmap = self.fold_of_pixel()
        mask = fmap == fold
        if apply_collar:
            mask &= ~self._collar_mask(fmap, fold)
        return mask

    def train_mask(self, fold: int, apply_collar: bool = True) -> np.ndarray:
        fmap = self.fold_of_pixel()
        mask = fmap != fold
        if apply_collar:
            mask &= ~self._collar_mask(fmap, fold)
        return mask

    def _collar_mask(self, fmap: np.ndarray, fold: int) -> np.ndarray:
        if self.collar_px <= 0:
            return np.zeros(fmap.shape, dtype=bool)
        from scipy.ndimage import binary_dilation

        ring = fmap == fold
        return binary_dilation(ring, iterations=self.collar_px)

    def as_dict(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "block_px": self.block_px,
            "collar_px": self.collar_px,
            "n_folds": self.n_folds,
            "block_px_km_at_100m": self.block_px * 0.1,
            "fold_block_counts": np.bincount(
                self.fold_of_block[self.fold_of_block >= 0], minlength=self.n_folds).tolist(),
        }


def make_fold_plan(height: int, width: int, n_folds: int = DEFAULT_N_FOLDS,
                   block_px: int = DEFAULT_BLOCK_PX, collar_px: int = DEFAULT_COLLAR_PX,
                   truth: np.ndarray | None = None, seed: int = 0) -> FoldPlan:
    """Assign blocks to folds.

    With ``truth`` supplied, blocks are sorted by catalogue-pixel count and dealt
    round-robin to folds.  That keeps the *fault-bearing* share of each fold
    comparable, which matters because the folds are averaged.  Without ``truth``,
    a deterministic spatial sweep (diagonal order) is used, which needs no RNG.
    """
    nb_y = (height + block_px - 1) // block_px
    nb_x = (width + block_px - 1) // block_px
    fold_of_block = np.full((nb_y, nb_x), -1, dtype=np.int8)

    if truth is None:
        # deterministic diagonal sweep, no RNG: block (by, bx) sorts by by + bx
        rank_of_block = (np.arange(nb_y)[:, None] + np.arange(nb_x)[None, :]).ravel()
        for rank, flat in enumerate(np.argsort(rank_of_block, kind="stable")):
            fold_of_block.flat[flat] = rank % n_folds
    else:
        counts = np.zeros(nb_y * nb_x, dtype=np.int64)
        t = np.asarray(truth).astype(bool)
        for by in range(nb_y):
            for bx in range(nb_x):
                y0, x0 = by * block_px, bx * block_px
                sub = t[y0:y0 + block_px, x0:x0 + block_px]
                counts[by * nb_x + bx] = int(sub.sum())
        # Deal the highest-burden blocks out first; ties broken by block index so
        # the plan is reproducible without an RNG.
        order = np.lexsort((np.arange(nb_y * nb_x), -counts))
        for rank, flat in enumerate(order):
            fold_of_block.flat[flat] = rank % n_folds

    return FoldPlan(height=height, width=width, block_px=block_px, collar_px=collar_px,
                    n_folds=n_folds, fold_of_block=fold_of_block,
                    block_row0=0, block_col0=0)
