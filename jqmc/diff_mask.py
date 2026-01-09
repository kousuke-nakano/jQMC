"""Utilities for masking PyTree leaves by derivative type.

This module centralizes the logic for selectively stopping gradients on
parameter- or coordinate-like leaves so we can reuse the same dataclasses
for all derivative modes (params, positions, or none) without proliferating
specialized subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Iterable

import jax

# Field-name heuristics used when no explicit metadata is provided.  This is
# intentionally conservative and targets only the main variational and
# coordinate arrays we need to mask in practice.
_PARAM_FIELD_NAMES = {
    "lambda_matrix",
    "j_matrix",
    "jastrow_1b_param",
    "jastrow_2b_param",
    "jastrow_3b_param",
    "params",
}
_COORD_FIELD_NAMES = {"positions"}


@dataclass(frozen=True)
class DiffMask:
    """Simple mask controlling which leaf types remain differentiable."""

    params: bool = True
    coords: bool = True

    def update(self, *, params: bool | None = None, coords: bool | None = None) -> "DiffMask":
        """Return a new mask with any provided overrides applied."""
        return DiffMask(
            params=self.params if params is None else params,
            coords=self.coords if coords is None else coords,
        )


def _maybe_stop_gradient(value: Any, *, allow_params: bool, allow_coords: bool, tag: str | None) -> Any:
    """Stop gradients on a value based on its tag and mask flags."""
    if tag == "param" and not allow_params:
        return _stop_leaf(value)
    if tag == "coord" and not allow_coords:
        return _stop_leaf(value)
    return value


def _stop_leaf(value: Any) -> Any:
    """Apply ``stop_gradient`` to array-like leaves; pass everything else through."""
    if value is None:
        return None
    if isinstance(value, (int, float, complex, bool)):
        return value
    try:
        return jax.lax.stop_gradient(value)
    except Exception:
        return value


def _tag_for_field(field) -> str | None:
    """Infer the diff tag for a dataclass field using metadata or heuristics."""
    if field.metadata and "diff_tag" in field.metadata:
        return field.metadata["diff_tag"]
    if field.name in _PARAM_FIELD_NAMES:
        return "param"
    if field.name in _COORD_FIELD_NAMES:
        return "coord"
    return None


def _mask_sequence(seq: Iterable[Any], mask: DiffMask):
    """Apply ``apply_diff_mask`` elementwise to a list/tuple preserving type."""
    mapped = [apply_diff_mask(v, mask) for v in seq]
    return type(seq)(mapped) if isinstance(seq, tuple) else mapped


def apply_diff_mask(obj: Any, mask: DiffMask) -> Any:
    """Return a copy of ``obj`` with gradients stopped according to ``mask``.

    The function recurses through dataclass fields (including Flax ``struct``
    dataclasses), honoring optional ``diff_tag`` metadata on fields when
    present. Otherwise, a small set of field-name heuristics is used to decide
    whether a leaf should be treated as a parameter ("param") or coordinate
    ("coord").  Lists/tuples are traversed elementwise; everything else is
    returned unchanged.
    """
    if is_dataclass(obj):
        updates = {}
        for f in fields(obj):
            value = getattr(obj, f.name)

            if is_dataclass(value):
                value = apply_diff_mask(value, mask)
            elif isinstance(value, (list, tuple)):
                value = _mask_sequence(value, mask)

            tag = _tag_for_field(f)
            value = _maybe_stop_gradient(value, allow_params=mask.params, allow_coords=mask.coords, tag=tag)
            updates[f.name] = value

        replace_fn = getattr(obj, "replace", None)
        if callable(replace_fn):
            return replace_fn(**updates)
        return obj.__class__(**updates)

    if isinstance(obj, (list, tuple)):
        return _mask_sequence(obj, mask)

    return obj
