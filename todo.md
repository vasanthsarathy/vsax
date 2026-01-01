# Fix FHRR Unbind Bug - Circular Deconvolution

**Date:** January 1, 2026
**Issue:** FHRR unbind method is missing division by |fft_b|² in circular deconvolution
**Current Version:** v1.3.0
**Target Version:** v1.3.1 (bug fix release)

---

## Problem Analysis

The unbind method in `vsax/ops/fhrr.py:177` incorrectly implements circular deconvolution:

**Current (WRONG):**
```python
return jnp.fft.ifft(fft_a * jnp.conj(fft_b))
```

**Correct:**
```python
return jnp.fft.ifft(fft_a * jnp.conj(fft_b) / (jnp.abs(fft_b)**2 + epsilon))
```

**Impact:**
- Norm explosion (16,196 vs 22.6)
- Poor similarity (~0.70 vs 1.0)
- Unbinding doesn't properly recover original vectors

---

## Implementation Plan

### Step 1: Fix unbind method
- [ ] Update `vsax/ops/fhrr.py` unbind method (line 177)
- [ ] Add epsilon parameter for numerical stability (1e-10)
- [ ] Update method docstring with corrected explanation
- [ ] Update comment explaining the mathematics

### Step 2: Run tests
- [ ] Run all FHRR tests: `uv run pytest tests/ops/test_fhrr.py -v`
- [ ] Verify all tests pass with corrected implementation
- [ ] Check that unbind similarity improves as expected
- [ ] Run full test suite: `uv run pytest`

### Step 3: Update documentation
- [ ] Update CHANGELOG.md with bug fix entry
- [ ] Update docs if unbind behavior is documented
- [ ] Add note about numerical stability

### Step 4: Version bump
- [ ] Update version to v1.3.1 in `vsax/__init__.py`
- [ ] Update version in `pyproject.toml`
- [ ] Update version in `tests/test_infrastructure.py`

### Step 5: Pre-commit checks
- [ ] Run linting: `uv run ruff check vsax tests`
- [ ] Run formatting: `uv run ruff format vsax tests`
- [ ] Run type checking: `uv run mypy vsax`
- [ ] Run full test suite with coverage: `uv run pytest --cov=vsax`

### Step 6: Commit and push
- [ ] Git commit with descriptive message
- [ ] Git push to remote

---

## Review Section

(To be filled after implementation)

### Changes Made:
- TBD

### Test Results:
- TBD

### Notes:
- TBD
