# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Standard Development Workflow

**CRITICAL: Follow these steps for EVERY feature, update, or bug fix. Do NOT skip steps.**

### Step 1: Ideation and Planning
- Discuss the feature/update with the user
- Create a plan in todo.md or use the planning system
- Get user approval before proceeding

### Step 2: Implementation
- Write code following the architecture and design principles below
- Make changes as simple as possible, impacting minimal code
- Follow type safety and code quality standards
- **Package Management**: Use `uv add <package>` to add dependencies (NEVER `pip install`)

### Step 3: Testing
**ALWAYS use `uv run` for all commands:**
```bash
uv run pytest                           # Run all tests
uv run pytest --cov=vsax               # Run with coverage
uv run pytest tests/test_specific.py   # Run specific test
```

**NEVER use `pytest` or `python -m pytest` directly - ALWAYS use `uv run`**

### Step 4: Documentation Updates
Update ALL relevant documentation:
- [ ] Update `README.md` if public API changed
- [ ] Update `docs/index.md` (main documentation landing page)
- [ ] Update `CHANGELOG.md` with changes
- [ ] Update relevant guide files in `docs/guide/`
- [ ] Update relevant API reference files in `docs/api/`
- [ ] Update `docs/getting-started.md` if needed
- [ ] Update `docs/design-spec.md` if architecture changed
- [ ] **Review ALL documentation for consistency and version numbers**
- [ ] Build docs to verify: `uv run mkdocs build`

**Version Updates:**
- [ ] Update `vsax/__init__.py` `__version__`
- [ ] Update `pyproject.toml` version
- [ ] Update `tests/test_infrastructure.py` version assertion
- [ ] Update citation version in README.md and docs/index.md

### Step 5: Pre-commit Checks
**ALWAYS use `uv run` for all commands:**
```bash
uv run ruff check vsax tests    # Linting
uv run ruff format vsax tests   # Formatting
uv run mypy vsax                # Type checking
uv run pytest --cov=vsax        # Full test suite with coverage
```

**Fix ALL errors before proceeding. Do NOT commit if any checks fail.**

### Step 6: Publishing Scripts
- [ ] Review `publish.ps1` or publishing scripts
- [ ] Verify version numbers are updated in scripts
- [ ] Ensure CHANGELOG.md is current

### Step 7: Commit and Push
```bash
git add .
git commit -m "descriptive message

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
git push
```

**If CI/CD errors occur:**
- User will report the errors
- Go back to Step 2 and fix the issues
- Run through ALL steps again (2-8)
- Do NOT skip steps when fixing CI/CD errors

### Step 8: Documentation Deployment

**CRITICAL: Deploy documentation AFTER committing/pushing but BEFORE running publish.ps1**

This ensures users see correct documentation when the new version is published to PyPI.

**Pre-deployment Verification:**
- [ ] Verify `mkdocs.yml` has all new tutorials/guides in navigation
- [ ] Verify `docs/index.md` version numbers are updated
- [ ] Verify citation version is current in `README.md` and `docs/index.md`
- [ ] Build locally to test: `uv run mkdocs build`
- [ ] Fix any build errors before deploying

**Option 1: Versioned Documentation with Mike (Recommended for releases)**
```bash
# Deploy new version and set as latest
uv run mike deploy --push <version> latest

# Example for v1.2.0:
uv run mike deploy --push 1.2.0 latest

# Set as default version (optional)
uv run mike set-default --push 1.2.0
```

**Option 2: Simple Deployment (For quick updates, no versioning)**
```bash
# Deploy directly to gh-pages branch
uv run mkdocs gh-deploy --force
```

**Post-deployment Verification:**
- [ ] Visit documentation site (e.g., https://vsax.readthedocs.io or GitHub Pages URL)
- [ ] Verify version number displays correctly
- [ ] Verify new tutorials/guides appear in navigation
- [ ] Verify all links work
- [ ] Check that code examples render correctly

**IMPORTANT: After successful deployment, proceed to run `publish.ps1` to publish to PyPI**

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

**CRITICAL: ALWAYS use `uv` for package management and `uv run` for all Python commands**

Package Management:
```bash
uv add <package>                 # Add a new dependency (NOT pip install!)
uv add --dev <package>           # Add a dev dependency
uv remove <package>              # Remove a dependency
uv sync                          # Sync dependencies with lock file
```

**NEVER use `pip install` directly - ALWAYS use `uv add` to add dependencies**

Build/install:
```bash
uv pip install -e .              # Install in editable mode
uv pip install -e ".[dev]"       # Install with dev dependencies
```

Testing:
```bash
uv run pytest                        # Run all tests
uv run pytest tests/test_model.py    # Run specific test file
uv run pytest -v -s                  # Verbose output with print statements
uv run pytest --cov=vsax             # Run with coverage
```

Linting/formatting:
```bash
uv run ruff check vsax tests    # Lint code
uv run ruff format vsax tests   # Format code
uv run mypy vsax                # Type checking
```

Documentation:
```bash
uv run mkdocs serve             # Serve docs locally at http://127.0.0.1:8000
uv run mkdocs build             # Build static documentation site
```

Publishing:
```bash
.\publish.ps1                   # Use the publish script (handles version, build, upload)
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
