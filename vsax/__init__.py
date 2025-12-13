"""VSAX: JAX-native Vector Symbolic Algebra library.

VSAX provides composable symbolic representations using hypervectors, algebraic operations
for binding and bundling, and encoding strategies for symbolic and structured data.

Example:
    >>> from vsax import create_fhrr_model, VSAMemory
    >>> from vsax.representations import ComplexHypervector
    >>> from vsax.ops import FHRROperations
    >>> from vsax.sampling import sample_complex_random
    >>>
    >>> # Create an FHRR model
    >>> model = VSAModel(
    ...     dim=512,
    ...     rep_cls=ComplexHypervector,
    ...     opset=FHRROperations(),
    ...     sampler=sample_complex_random
    ... )
"""

__version__ = "0.2.0"

from vsax.core.base import AbstractHypervector, AbstractOpSet
from vsax.core.model import VSAModel
from vsax.ops import BinaryOperations, FHRROperations, MAPOperations
from vsax.representations import BinaryHypervector, ComplexHypervector, RealHypervector
from vsax.sampling import sample_binary_random, sample_complex_random, sample_random

__all__ = [
    # Core
    "AbstractHypervector",
    "AbstractOpSet",
    "VSAModel",
    # Representations
    "ComplexHypervector",
    "RealHypervector",
    "BinaryHypervector",
    # Operations
    "FHRROperations",
    "MAPOperations",
    "BinaryOperations",
    # Sampling
    "sample_random",
    "sample_complex_random",
    "sample_binary_random",
    # Version
    "__version__",
]
