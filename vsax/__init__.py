"""VSAX: JAX-native Vector Symbolic Algebra library.

VSAX provides composable symbolic representations using hypervectors, algebraic operations
for binding and bundling, and encoding strategies for symbolic and structured data.

Example:
    >>> from vsax.core import VSAModel
    >>> from vsax.representations import ComplexHypervector
    >>> from vsax.ops import FHRROperations
    >>> from vsax.sampling import sample_complex_random
    >>> model = VSAModel(
    ...     dim=512,
    ...     rep_cls=ComplexHypervector,
    ...     opset=FHRROperations(),
    ...     sampler=sample_complex_random
    ... )
"""

__version__ = "0.1.0"

from vsax.core.base import AbstractHypervector, AbstractOpSet
from vsax.core.model import VSAModel

__all__ = [
    "AbstractHypervector",
    "AbstractOpSet",
    "VSAModel",
    "__version__",
]
