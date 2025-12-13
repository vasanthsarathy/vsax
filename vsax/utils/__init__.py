"""Utility functions."""

from vsax.utils.batch import vmap_bind, vmap_bundle, vmap_similarity
from vsax.utils.coerce import coerce_to_array
from vsax.utils.repr import format_similarity_results, pretty_repr
from vsax.utils.validation import validate_positive_int, validate_string

__all__ = [
    "coerce_to_array",
    "validate_positive_int",
    "validate_string",
    "vmap_bind",
    "vmap_bundle",
    "vmap_similarity",
    "pretty_repr",
    "format_similarity_results",
]
