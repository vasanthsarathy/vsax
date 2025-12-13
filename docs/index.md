# VSAX: Vector Symbolic Algebra for JAX

VSAX is a GPU-accelerated, JAX-native Python library for Vector Symbolic Architectures (VSAs). It provides composable symbolic representations using hypervectors, algebraic operations for binding and bundling, and encoding strategies for symbolic and structured data.

## Features

- 🚀 **Three VSA Models**: FHRR, MAP, and Binary implementations ✅
- ⚡ **GPU-Accelerated**: Built on JAX for high-performance computation
- 🧩 **Modular Architecture**: Clean separation between representations and operations
- 🧬 **Complete Representations**: Complex, Real, and Binary hypervectors ✅
- ⚙️ **Full Operation Sets**: FFT-based FHRR, MAP, and XOR/majority Binary ops ✅
- 🎲 **Random Sampling**: Sampling utilities for all representation types ✅
- 💯 **Type-Safe**: Full type annotations with mypy support
- ✅ **Well-Tested**: 175 tests with 96% coverage

## Installation

### From PyPI (Coming Soon)

```bash
pip install vsax
```

### From Source

```bash
git clone https://github.com/yourusername/vsax.git
cd vsax

# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

## Quick Example

### Simple API (v0.3.0+)

```python
from vsax import create_fhrr_model, VSAMemory

# Create model with factory function
model = create_fhrr_model(dim=512)

# Create memory for symbols
memory = VSAMemory(model)
memory.add_many(["dog", "cat", "animal"])

# Access and manipulate symbols
dog = memory["dog"]
animal = memory["animal"]

# Bind two concepts (circular convolution)
dog_is_animal = model.opset.bind(dog.vec, animal.vec)

# Bundle multiple concepts (sum and normalize)
pets = model.opset.bundle(memory["dog"].vec, memory["cat"].vec)
```

### MAP Model (Real Hypervectors)

```python
from vsax import RealHypervector, MAPOperations, sample_random

model = VSAModel(
    dim=512,
    rep_cls=RealHypervector,
    opset=MAPOperations(),
    sampler=sample_random
)

# Element-wise multiplication for binding
# Element-wise mean for bundling
```

### Binary Model (Bipolar Hypervectors)

```python
from vsax import BinaryHypervector, BinaryOperations, sample_binary_random

model = VSAModel(
    dim=512,
    rep_cls=BinaryHypervector,
    opset=BinaryOperations(),
    sampler=sample_binary_random
)

# XOR binding (exact unbinding)
# Majority voting for bundling
```

## Development Status

**Current**: Iteration 2 Complete ✅

### Completed

**Iteration 1** (v0.1.0): Foundation & Infrastructure
- ✅ Core abstract classes (AbstractHypervector, AbstractOpSet)
- ✅ VSAModel dataclass
- ✅ Package structure
- ✅ Testing infrastructure (pytest, coverage)
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Documentation site (MkDocs)

**Iteration 2** (v0.2.0): Core Algebras
- ✅ All 3 representations (Complex, Real, Binary)
- ✅ All 3 operation sets (FHRR, MAP, Binary)
- ✅ Sampling utilities
- ✅ 175 comprehensive tests with 96% coverage
- ✅ Full integration tests

### Coming Next

**Iteration 3** (v0.3.0): Models & Memory
- VSAMemory for symbol storage
- Factory functions for easy model creation
- Integration utilities

**Iteration 4** (v0.4.0): First Usable Release
- ScalarEncoder and DictEncoder
- Complete working examples
- Tutorial notebooks

## Documentation

- [Getting Started](getting-started.md) - Installation and first steps
- [User Guide](guide/representations.md) - Detailed guides for all components
- [Examples](examples/fhrr.md) - Working examples for all three models
- [API Reference](api/index.md) - Complete API documentation
- [Design Specification](design-spec.md) - Technical design details

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/yourusername/vsax/blob/main/CONTRIBUTING.md) for guidelines.

## License

VSAX is released under the MIT License. See [LICENSE](https://github.com/yourusername/vsax/blob/main/LICENSE) for details.

## Citation

If you use VSAX in your research, please cite:

```bibtex
@software{vsax2025,
  title = {VSAX: Vector Symbolic Algebra for JAX},
  author = {VSAX Contributors},
  year = {2025},
  version = {0.2.0},
  url = {https://github.com/yourusername/vsax}
}
```
