"""Clifford-inspired operator layer for VSAX.

Provides exact, compositional, invertible operators for hypervector transformations.
Compatible with FHRR (ComplexHypervector) representations only.

The operator layer implements a Clifford-inspired algebra where:
- Operators represent "what happens" (transformations, relations, actions)
- Hypervectors represent "what exists" (concepts, objects, symbols)
- Operators are exact (perfect inversion)
- Operators are compositional (can be combined algebraically)
- Operators are typed (semantic metadata via OperatorKind)

Example:
    >>> from vsax import create_fhrr_model, VSAMemory
    >>> from vsax.operators import CliffordOperator
    >>> import jax
    >>>
    >>> model = create_fhrr_model(dim=512)
    >>> memory = VSAMemory(model)
    >>> memory.add("test")
    >>>
    >>> # Create operator
    >>> op = CliffordOperator.random(512, key=jax.random.PRNGKey(0))
    >>>
    >>> # Apply and invert
    >>> transformed = op.apply(memory["test"])
    >>> recovered = op.inverse().apply(transformed)
"""

from vsax.operators.base import AbstractOperator
from vsax.operators.clifford import CliffordOperator
from vsax.operators.kinds import OperatorKind, OperatorMetadata

__all__ = [
    # Base
    "AbstractOperator",
    "CliffordOperator",
    # Kinds
    "OperatorKind",
    "OperatorMetadata",
]
