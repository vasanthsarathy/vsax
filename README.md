# VSAX: Vector Symbolic Algebra for JAX

[![PyPI version](https://img.shields.io/pypi/v/vsax.svg)](https://pypi.org/project/vsax/)
[![Python Version](https://img.shields.io/pypi/pyversions/vsax.svg)](https://pypi.org/project/vsax/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://vasanthsarathy.github.io/vsax/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

VSAX is a GPU-accelerated, JAX-native Python library for Vector Symbolic Architectures (VSAs). It provides composable symbolic representations using hypervectors, algebraic operations for binding and bundling, and encoding strategies for symbolic and structured data.

## Features

- 🚀 **Three VSA Models**: FHRR, MAP, and Binary implementations ✅
- 🏭 **Factory Functions**: One-line model creation with sensible defaults ✅ **NEW in v0.3.0**
- 💾 **VSAMemory**: Dictionary-style symbol management ✅ **NEW in v0.3.0**
- ⚡ **GPU-Accelerated**: Built on JAX for high-performance computation
- 🧩 **Modular Architecture**: Clean separation between representations and operations
- 🧬 **Complete Representations**: Complex, Real, and Binary hypervectors ✅
- ⚙️ **Full Operation Sets**: FFT-based FHRR, MAP, and XOR/majority Binary ops ✅
- 🎲 **Random Sampling**: Sampling utilities for all representation types ✅
- 📊 **Encoders**: Scalar and dictionary encoders for structured data *(coming in v0.4.0)*
- 🔍 **Similarity Metrics**: Cosine, dot, and Hamming similarity *(coming in v0.5.0)*
- 📚 **Comprehensive Documentation**: Full API docs and examples ✅
- ✅ **89% Test Coverage**: 230 tests ensuring reliability

## Installation

### From PyPI (Recommended)

```bash
pip install vsax
```

Or with uv:

```bash
uv pip install vsax
```

### From Source

#### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. Install it first:

```bash
# Install uv (Unix/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install uv (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install VSAX:

```bash
git clone https://github.com/vasanthsarathy/vsax.git
cd vsax

# Create virtual environment and install package
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

#### Using pip

```bash
git clone https://github.com/vasanthsarathy/vsax.git
cd vsax
pip install -e .
```

### Development Installation

#### Using uv (Recommended)

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev,docs]"
```

#### Using pip

```bash
pip install -e ".[dev,docs]"
```

## Quick Start

**New in v0.3.0:** Easy-to-use factory functions and VSAMemory for symbol management!

### Simple Example

```python
from vsax import create_fhrr_model, VSAMemory

# Create model with factory function (one line!)
model = create_fhrr_model(dim=512)

# Create memory for symbol management
memory = VSAMemory(model)

# Add symbols - automatically samples and stores hypervectors
memory.add_many(["dog", "cat", "animal", "pet"])

# Dictionary-style access
dog = memory["dog"]
animal = memory["animal"]

# Check if symbol exists
if "dog" in memory:
    print(f"Memory contains {len(memory)} symbols")

# Bind concepts (circular convolution)
dog_is_animal = model.opset.bind(dog.vec, animal.vec)

# Bundle concepts (sum and normalize)
pets = model.opset.bundle(memory["dog"].vec, memory["cat"].vec)
```

### All Three Models

VSAX supports three VSA models, all with the same simple API:

```python
from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory

# FHRR: Complex hypervectors, exact unbinding
fhrr = create_fhrr_model(dim=512)

# MAP: Real hypervectors, approximate unbinding
map_model = create_map_model(dim=512)

# Binary: Discrete hypervectors, exact unbinding
binary = create_binary_model(dim=10000, bipolar=True)

# Same interface for all models!
for model in [fhrr, map_model, binary]:
    memory = VSAMemory(model)
    memory.add("concept")
    vec = memory["concept"]
```

### Advanced: Manual Model Creation

You can still create models manually if you need custom configuration:

```python
from vsax import VSAModel, ComplexHypervector, FHRROperations, sample_complex_random

model = VSAModel(
    dim=512,
    rep_cls=ComplexHypervector,
    opset=FHRROperations(),
    sampler=sample_complex_random
)
```

See [docs/design-spec.md](docs/design-spec.md) for complete technical specification.

## Development Status

Currently in **Iteration 3: VSAMemory + Factory Functions** ✅

### Completed

**Iteration 1** (v0.1.0): Foundation & Infrastructure ✅
- ✅ Core abstract classes (AbstractHypervector, AbstractOpSet)
- ✅ VSAModel dataclass
- ✅ Package structure
- ✅ Testing infrastructure (pytest, coverage)
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Documentation site (MkDocs)
- ✅ Development tooling (ruff, mypy)

**Iteration 2** (v0.2.0): All 3 Representations + All 3 OpSets ✅
- ✅ ComplexHypervector, RealHypervector, BinaryHypervector
- ✅ FHRROperations, MAPOperations, BinaryOperations
- ✅ Sampling utilities (sample_random, sample_complex_random, sample_binary_random)
- ✅ 175 comprehensive tests with 96% coverage
- ✅ Full integration tests for all model combinations

**Iteration 3** (v0.3.0): VSAMemory + Factory Functions ✅
- ✅ VSAMemory class - dictionary-style symbol management
- ✅ Factory functions (create_fhrr_model, create_map_model, create_binary_model)
- ✅ Utility functions (coerce_to_array, validation helpers)
- ✅ 230 tests with 89% coverage
- ✅ Comprehensive documentation guides

### Coming Next

**Iteration 4** (v0.4.0): First Usable Release 🎯
- ScalarEncoder for numeric values
- DictEncoder for structured data
- Complete working examples for all three models

See [todo.md](todo.md) for the complete development roadmap.

## Documentation

- [Full Documentation](https://vasanthsarathy.github.io/vsax/)
- [Getting Started](docs/getting-started.md)
- [Design Specification](docs/design-spec.md)
- [API Reference](https://vasanthsarathy.github.io/vsax/api/)
- [Contributing](CONTRIBUTING.md)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Development

Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=vsax --cov-report=term-missing
```

Type checking:
```bash
mypy vsax
```

Linting:
```bash
ruff check vsax tests
```

Build documentation:
```bash
mkdocs serve
```

## License

VSAX is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use VSAX in your research, please cite:

```bibtex
@software{vsax2025,
  title = {VSAX: Vector Symbolic Algebra for JAX},
  author = {Sarathy, Vasanth},
  year = {2025},
  url = {https://github.com/vasanthsarathy/vsax},
  version = {0.2.0}
}
```

## Acknowledgments

VSAX is built on [JAX](https://github.com/google/jax) and inspired by the VSA/HDC research community.