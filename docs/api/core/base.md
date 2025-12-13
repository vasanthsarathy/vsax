# Base Classes

Core abstract classes that define the VSA interface.

## AbstractHypervector

::: vsax.core.base.AbstractHypervector
    options:
      show_source: true
      members:
        - vec
        - shape
        - dtype
        - normalize
        - to_numpy

## AbstractOpSet

::: vsax.core.base.AbstractOpSet
    options:
      show_source: true
      members:
        - bind
        - bundle
        - inverse
        - permute
