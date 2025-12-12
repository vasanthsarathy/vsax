# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VSAX is a GPU-accelerated, JAX-native Python library for vector symbolic architectures (VSAs). It provides composable symbolic representations using hypervectors, algebraic operations for binding and bundling, and encoding strategies for symbolic and structured data.

## Core Architecture

### Central Components

1. **VSAModel** - Immutable dataclass container that defines the VSA algebra:
   - `dim: int` - dimensionality of all hypervectors
   - `rep_cls: Type[AbstractHypervector]` - representation class (e.g., ComplexHypervector)
   - `opset: AbstractOpSet` - operation strategies (bind, bundle, inverse)
   - `sampler: Callable[[int, int], jnp.ndarray]` - function for sampling raw vectors

2. **VSAMemory** - Symbol table and runtime memory for symbolic concepts:
   - Stores named hypervectors (basis symbols)
   - Dictionary-style access: `memory.add("apple")`, `memory["apple"]`
   - Uses VSAModel to sample and wrap vectors

3. **AbstractHypervector** - Base class for representations wrapping `jnp.ndarray`:
   - `.vec` - underlying vector
   - `.normalize()`, `.to_numpy()` methods
   - `.shape`, `.dtype` proxies

4. **AbstractOpSet** - Stateless, pure functional interface for symbolic algebra:
   - `bind(a, b)` - bind two hypervectors
   - `bundle(*args)` - bundle multiple hypervectors
   - `inverse(a)` - inverse of hypervector
   - `permute(a, shift)` - optional permutation

### Module Organization

- `vsax/representations/` - ComplexHypervector, BinaryHypervector, RealHypervector
- `vsax/ops/` - FHRROperations (FFT-based), MAPOperations (elementwise), BinaryOperations (XOR, majority)
- `vsax/encoders/` - ScalarEncoder, DictEncoder (and future: GraphEncoder, SequenceEncoder, TreeEncoder)
- `vsax/similarity/` - cosine_similarity, dot_similarity, hamming_similarity
- `vsax/sampling/` - sample_random, sample_circular
- `vsax/utils/` - coerce_vec, vmap_ops, pretty_repr
- `vsax/io/` - save_basis, load_basis (JSON serialization of named basis vectors)
- `tests/` - test coverage for all components

## Key Design Principles

1. **JAX-Native**: All operations must be compatible with JAX for GPU/TPU acceleration, differentiation, and JIT compilation
2. **Modular**: Representations and operation sets are completely decoupled and composable
3. **Functional**: Operation sets are stateless and pure functional
4. **NumPy-like API**: Should be as usable and expressive as NumPy or PyTorch
5. **Clean Separation**: VSAModel holds algebra definition but performs no operations; encoders and memory use the model

## Implementation Notes

### Vector Operations
- All operations in AbstractOpSet work directly on `jnp.ndarray` objects
- AbstractHypervector wraps arrays and provides convenience methods
- Future: implement `__jax_array__` and `__array__` for seamless operations

### Encoders
- Each encoder accepts a model and memory in constructor
- Implement `.fit()` and `.encode()` for consistency
- ScalarEncoder: powers basis vector by value (e.g., `basis_vec ** value`)
- DictEncoder: bundles binding of role-filler pairs

### Persistent Basis Vectors
- `save_basis(memory, path)` serializes named basis vectors to JSON
- `load_basis(memory, path)` loads into memory using model's `rep_cls`
- Enables reuse of symbolic spaces across sessions

## Development Workflow

### Package Management
- Use `pyproject.toml` with Poetry or Hatch for modern Python packaging
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Maintain `requirements.txt` or use Poetry's lock file for reproducibility

### Version Control
- Follow conventional commits specification for commit messages
- Use feature branches for development (`feature/`, `fix/`, `docs/`)
- Always push changes to remote after committing
- Create pull requests for major features
- Tag releases with version numbers (e.g., `v0.1.0`)

### Development Commands
Build/install:
```bash
pip install -e .              # Install in editable mode
pip install -e ".[dev]"       # Install with dev dependencies
```

Testing:
```bash
pytest                        # Run all tests
pytest tests/test_model.py    # Run specific test file
pytest -v -s                  # Verbose output with print statements
pytest --cov=vsax            # Run with coverage
```

Linting/formatting:
```bash
ruff check .                  # Lint code
ruff format .                 # Format code
mypy vsax                     # Type checking
```

Documentation:
```bash
mkdocs serve                  # Serve docs locally at http://127.0.0.1:8000
mkdocs build                  # Build static documentation site
```

Publishing:
```bash
python -m build              # Build distribution packages
twine check dist/*           # Verify package
twine upload dist/*          # Upload to PyPI
```

### Testing Strategy

Tests should validate:
- Model and memory initialization
- Scalar and dictionary encoding
- Similarity metrics correctness
- Save/load persistence
- Vector operations
- Batch encoding (vmap compatibility)
- Symbolic consistency (bind/unbind round-trips)

### Documentation

**README.md** should include:
- Project overview and key features
- Installation instructions
- Quick start examples
- Links to full documentation
- Citation information
- License and contributing guidelines

**Documentation site** (using MkDocs):
- Structure: `docs/` directory with markdown files
- Sections: Getting Started, API Reference, Tutorials, Examples, Theory
- Auto-generate API docs from docstrings using `mkdocstrings`
- Update documentation whenever API changes or new features are added
- Deploy to GitHub Pages or ReadTheDocs

### Code Quality Standards
- All public functions/classes must have docstrings (Google or NumPy style)
- Type hints required for all function signatures
- Maintain test coverage above 80%
- Use `ruff` for linting and formatting (replaces black, isort, flake8)
- Run `mypy` for static type checking

### Publishing to PyPI
1. Ensure version number is updated in `pyproject.toml` or `__version__`
2. Update CHANGELOG.md with version changes
3. Run full test suite and ensure all pass
4. Build distribution: `python -m build`
5. Verify package: `twine check dist/*`
6. Upload to TestPyPI first: `twine upload --repository testpypi dist/*`
7. Test install from TestPyPI
8. Upload to PyPI: `twine upload dist/*`
9. Create git tag: `git tag v0.1.0` and push tags
10. Create GitHub release with changelog
