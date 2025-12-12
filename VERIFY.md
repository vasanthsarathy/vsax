# Iteration 1 Verification Guide

This guide will help you verify that Iteration 1 is set up correctly.

## Prerequisites

First, ensure `uv` is installed:

```bash
# Check if uv is installed
uv --version

# If not installed (Windows):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# If not installed (Unix/macOS):
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step 1: Verify File Structure

Run the verification script:

```bash
# This doesn't require dependencies, just checks files exist
python verify_setup.py
```

Expected output: All checkmarks (✓) with no missing files (✗)

## Step 2: Create Virtual Environment

```bash
# Create a virtual environment
uv venv

# Activate it
# On Windows:
.venv\Scripts\activate

# On Unix/macOS:
source .venv/bin/activate
```

## Step 3: Install Package

```bash
# Install VSAX in development mode with all dependencies
uv pip install -e ".[dev,docs]"
```

Expected output: Installation completes successfully with JAX, pytest, ruff, mypy, mkdocs, etc.

## Step 4: Test Imports

```bash
# Test that the package can be imported
python -c "import vsax; print(f'vsax v{vsax.__version__}')"
python -c "from vsax.core import AbstractHypervector, AbstractOpSet, VSAModel; print('All imports successful')"
```

Expected output:
```
vsax v0.1.0
All imports successful
```

## Step 5: Run Tests

```bash
# Run the test suite
pytest -v

# Run with coverage
pytest --cov=vsax --cov-report=term-missing
```

Expected output:
- All tests pass (8 tests)
- Coverage: 100% (minimal code in iteration 1)

## Step 6: Type Checking

```bash
# Run mypy type checker
mypy vsax --no-site-packages
```

Expected output: `Success: no issues found in X source files`

## Step 7: Linting

```bash
# Check code style with ruff
ruff check vsax tests

# Format code
ruff format vsax tests --check
```

Expected output: No issues found

## Step 8: Documentation

```bash
# Build and serve documentation
mkdocs serve
```

Expected output:
- Server starts at http://127.0.0.1:8000
- Open in browser and verify pages load correctly
- Check: Home, Getting Started, API Reference

## Step 9: Review Key Files

Manually review these critical files:

1. **vsax/core/base.py**: Abstract base classes
   - AbstractHypervector with `.vec`, `.normalize()`, etc.
   - AbstractOpSet with `bind()`, `bundle()`, `inverse()`

2. **vsax/core/model.py**: VSAModel dataclass
   - Frozen dataclass with dim, rep_cls, opset, sampler
   - Validation in `__post_init__`

3. **tests/test_infrastructure.py**: Infrastructure tests
   - JAX availability tests
   - Import tests
   - VSAModel creation tests

4. **pyproject.toml**: Package configuration
   - Dependencies: jax, numpy
   - Dev dependencies: pytest, mypy, ruff
   - Test configuration

## Verification Checklist

- [ ] All files exist (Step 1)
- [ ] Virtual environment created (Step 2)
- [ ] Dependencies installed (Step 3)
- [ ] Package imports successfully (Step 4)
- [ ] All tests pass with 100% coverage (Step 5)
- [ ] Type checking passes (Step 6)
- [ ] Linting passes (Step 7)
- [ ] Documentation builds and renders (Step 8)
- [ ] Code review looks good (Step 9)

## If Issues Occur

### Import Errors
- Ensure virtual environment is activated
- Reinstall: `uv pip install -e ".[dev,docs]"`

### JAX Not Found (on tests)
- JAX may not have CPU support on Windows by default
- Install JAX manually: `uv pip install jax[cpu]`

### Test Failures
- Check Python version: `python --version` (should be 3.9+)
- Ensure all dependencies installed
- Check for missing imports

### Type Checking Errors
- First run may require stub packages
- Install stubs: `uv pip install types-setuptools`

## Success Criteria

✅ **Iteration 1 is complete when:**
1. All 8 tests pass
2. Coverage is 100%
3. Type checking passes
4. Linting passes
5. Documentation builds
6. Package can be imported

## Next Steps

Once verification passes:
1. Commit all changes to git
2. Proceed to Iteration 2: Representations and Operations
3. Celebrate! 🎉
