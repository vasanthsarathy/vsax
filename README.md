# VSAX: Vector Symbolic Algebra for JAX

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

VSAX is a GPU-accelerated, JAX-native Python library for Vector Symbolic Architectures (VSAs). It provides composable symbolic representations using hypervectors, algebraic operations for binding and bundling, and encoding strategies for symbolic and structured data.

## Features

- 🚀 **Three VSA Models**: FHRR, MAP, and Binary implementations *(coming in iteration 2)*
- ⚡ **GPU-Accelerated**: Built on JAX for high-performance computation
- 🧩 **Modular Architecture**: Clean separation between representations and operations
- 📊 **Encoders**: Scalar and dictionary encoders for structured data *(coming in iteration 4)*
- 💾 **Persistent Storage**: Save and load basis vectors *(coming in iteration 6)*
- 🔍 **Similarity Metrics**: Cosine, dot, and Hamming similarity *(coming in iteration 5)*
- 📚 **Comprehensive Documentation**: Full API docs and examples

## Installation

### From PyPI (Coming Soon)

```bash
pip install vsax
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
git clone https://github.com/yourusername/vsax.git
cd vsax

# Create virtual environment and install package
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

#### Using pip

```bash
git clone https://github.com/yourusername/vsax.git
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

```python
from vsax import VSAModel, AbstractHypervector, AbstractOpSet

# More examples coming in iteration 2+
# See docs/design-spec.md for complete technical specification
```

## Development Status

Currently in **Iteration 1: Foundation & Infrastructure** ✅

### Completed
- ✅ Core abstract classes (AbstractHypervector, AbstractOpSet)
- ✅ VSAModel dataclass
- ✅ Package structure
- ✅ Testing infrastructure (pytest, coverage)
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Documentation site (MkDocs)
- ✅ Development tooling (ruff, mypy)

### Coming Next

**Iteration 2** (v0.2.0): All 3 Representations + All 3 OpSets
- ComplexHypervector, RealHypervector, BinaryHypervector
- FHRROperations, MAPOperations, BinaryOperations
- Sampling utilities

**Iteration 3** (v0.3.0): VSAModel + VSAMemory
- Symbol table and memory management
- Factory functions for easy model creation

**Iteration 4** (v0.4.0): First Usable Release
- ScalarEncoder and DictEncoder
- Working examples for all three models

See [todo.md](todo.md) for the complete development roadmap.

## Documentation

- [Getting Started](docs/getting-started.md)
- [Design Specification](docs/design-spec.md)
- [API Reference](https://vsax.readthedocs.io) *(coming soon)*
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
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/vsax}
}
```

## Acknowledgments

VSAX is built on [JAX](https://github.com/google/jax) and inspired by the VSA/HDC research community.