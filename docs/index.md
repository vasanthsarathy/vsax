# VSAX: Vector Symbolic Algebra for JAX

VSAX is a GPU-accelerated, JAX-native Python library for Vector Symbolic Architectures (VSAs). It provides composable symbolic representations using hypervectors, algebraic operations for binding and bundling, and encoding strategies for symbolic and structured data.

## Features

- **Three VSA Models**: FHRR, MAP, and Binary implementations
- **GPU-Accelerated**: Built on JAX for high-performance computation
- **Modular Architecture**: Clean separation between representations and operations
- **Type-Safe**: Full type annotations with mypy support
- **Well-Tested**: Comprehensive test suite with high coverage

## Installation

```bash
pip install vsax
```

For development:

```bash
git clone https://github.com/yourusername/vsax.git
cd vsax
pip install -e ".[dev]"
```

## Quick Example

```python
from vsax import VSAModel, AbstractHypervector, AbstractOpSet

# More examples coming in iteration 2+
```

## Development Status

Currently in **Iteration 1**: Foundation & Infrastructure

- ✅ Core abstract classes (AbstractHypervector, AbstractOpSet)
- ✅ VSAModel dataclass
- ✅ Package structure
- ✅ Testing infrastructure
- ✅ CI/CD pipeline

**Coming Next (Iteration 2)**: All three VSA models (FHRR, MAP, Binary) with representations and operations.

## License

VSAX is released under the MIT License. See [LICENSE](https://github.com/yourusername/vsax/blob/main/LICENSE) for details.
