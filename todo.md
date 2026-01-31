# Quaternion Hypervectors (QHV) Implementation

**Date:** January 30, 2026
**Task:** Add Quaternion Hypervectors to VSAX following the design specification

---

## Overview

Quaternion Hypervectors provide:
- **Non-commutative binding** via Hamilton product (order-sensitive role/filler)
- **Unit quaternion constraint** (S³ manifold) for stability
- **Exact unbinding** for single bindings (left and right unbind needed)
- Vector shape: (D, 4) where D = number of quaternion coordinates

---

## Todo Items

### Phase 1: Core Quaternion Algebra
- [x] Create `vsax/ops/quaternion.py` with utility functions (qmul, qconj, qinverse, qnormalize)
- [x] Create `tests/ops/test_quaternion_algebra.py` with tests for Hamilton product

### Phase 2: Representation Class
- [x] Create `vsax/representations/quaternion_hv.py` with QuaternionHypervector class
- [x] Create `tests/representations/test_quaternion_hv.py` with tests

### Phase 3: Update Base Class
- [x] Add `unbind_left()` method to `vsax/core/base.py` AbstractOpSet

### Phase 4: Operation Set
- [x] Add `QuaternionOperations` class to `vsax/ops/quaternion.py`
- [x] Create `tests/ops/test_quaternion.py` with tests for operations

### Phase 5: Sampling Function
- [x] Add `sample_quaternion_random()` to `vsax/sampling/random.py`
- [x] Create `tests/sampling/test_quaternion.py` with tests

### Phase 6: Similarity Function
- [x] Create `vsax/similarity/quaternion.py` with quaternion_similarity
- [x] Create `tests/similarity/test_quaternion.py` with tests

### Phase 7: Factory Function
- [x] Add `create_quaternion_model()` to `vsax/core/factory.py`

### Phase 8: Exports
- [x] Update `vsax/representations/__init__.py`
- [x] Update `vsax/ops/__init__.py`
- [x] Update `vsax/sampling/__init__.py`
- [x] Update `vsax/similarity/__init__.py`
- [x] Update `vsax/core/__init__.py`
- [x] Update `vsax/__init__.py`

### Phase 9: Integration Tests
- [x] Create `tests/core/test_quaternion_model.py`

### Phase 10: Documentation
- [x] Update README.md
- [x] Create `docs/guide/quaternion.md`
- [x] Update `docs/guide/representations.md`
- [x] Create `docs/api/ops/quaternion.md`
- [x] Create `docs/api/representations/quaternion.md`
- [x] Update `mkdocs.yml` navigation
- [x] Update CHANGELOG.md

### Phase 11: Verification
- [x] Run all tests: `uv run pytest` - 747 passed
- [x] Run linting: `uv run ruff check vsax tests` - No errors
- [x] Run formatting: `uv run ruff format vsax tests` - Formatted
- [x] Run type checking: `uv run mypy vsax` - No issues
- [x] Build docs: `uv run mkdocs build` - Success

---

## Review Section

### Implementation Summary

**New Files Created (10):**
1. `vsax/ops/quaternion.py` - Quaternion algebra functions + QuaternionOperations class
2. `vsax/representations/quaternion_hv.py` - QuaternionHypervector representation class
3. `vsax/similarity/quaternion.py` - quaternion_similarity function
4. `tests/ops/test_quaternion_algebra.py` - 25 tests for Hamilton product, conjugate, inverse
5. `tests/ops/test_quaternion.py` - 24 tests for QuaternionOperations
6. `tests/representations/test_quaternion_hv.py` - 17 tests for QuaternionHypervector
7. `tests/sampling/test_quaternion.py` - 11 tests for quaternion sampling
8. `tests/similarity/test_quaternion.py` - 11 tests for quaternion similarity
9. `tests/core/test_quaternion_model.py` - 17 integration tests
10. `docs/guide/quaternion.md` - Complete user guide with examples

**Modified Files (11):**
1. `vsax/core/base.py` - Added `unbind_left()` method to AbstractOpSet
2. `vsax/core/factory.py` - Added `create_quaternion_model()` factory function
3. `vsax/sampling/random.py` - Added `sample_quaternion_random()` function
4. `vsax/representations/__init__.py` - Export QuaternionHypervector
5. `vsax/ops/__init__.py` - Export QuaternionOperations
6. `vsax/sampling/__init__.py` - Export sample_quaternion_random
7. `vsax/similarity/__init__.py` - Export quaternion_similarity
8. `vsax/core/__init__.py` - Export create_quaternion_model
9. `vsax/__init__.py` - Export all new public API
10. `README.md` - Added quaternion documentation and examples
11. `CHANGELOG.md` - Documented v1.4.0 changes
12. `mkdocs.yml` - Added quaternion guide and API pages
13. `docs/guide/representations.md` - Added quaternion section
14. `docs/api/ops/quaternion.md` - API reference page
15. `docs/api/representations/quaternion.md` - API reference page

### Key Features Implemented

1. **Non-Commutative Binding:**
   - Hamilton product: `bind(x, y) ≠ bind(y, x)`
   - Natural for role/filler, subject/object relationships

2. **Left and Right Unbinding:**
   - `unbind(z, y)` - Right-unbind: recover x from z = bind(x, y)
   - `unbind_left(x, z)` - Left-unbind: recover y from z = bind(x, y)
   - Both achieve >99% similarity recovery

3. **Unit Quaternion Constraint:**
   - All quaternions normalized to S³ manifold
   - Binding preserves unit length

4. **Full Integration:**
   - Works with VSAMemory
   - Works with all existing encoders
   - Consistent API with other VSA models

### Test Statistics

- **New quaternion tests:** 105
- **Total tests:** 747
- **Test coverage:** 94.65%
- **All checks passing:** ruff, mypy, pytest

### Backward Compatibility

- No breaking changes
- `unbind_left()` added to base class with default implementation
- Existing code continues to work unchanged

---

# Add Sandwich Product to Quaternions

**Date:** January 31, 2026
**Task:** Add sandwich product operation for rotor transformations

## Tasks

- [x] Add `sandwich()` and `sandwich_unit()` functions to `vsax/ops/quaternion.py`
- [x] Add convenience method to `QuaternionOperations` class
- [x] Export new functions in `vsax/ops/__init__.py`
- [x] Add tests for sandwich product in `tests/ops/test_quaternion.py`
- [x] Run tests and verify all pass (33 passed)
- [x] Run type checking and linting (mypy, ruff pass)
- [x] Update `docs/guide/quaternion.md` with sandwich product section
- [x] Update `docs/api/ops/quaternion.md` with sandwich function docs
- [x] Update `README.md` with sandwich product in key features
- [x] Update `CHANGELOG.md` with sandwich product under Unreleased

## Review

### Changes Made

**Modified Files:**
1. `vsax/ops/quaternion.py` - Added `sandwich()` and `sandwich_unit()` functions, plus convenience methods on `QuaternionOperations` class
2. `vsax/ops/__init__.py` - Exported `sandwich` and `sandwich_unit`
3. `tests/ops/test_quaternion.py` - Added 10 new tests for sandwich product
4. `docs/guide/quaternion.md` - Added sandwich product section with examples
5. `docs/api/ops/quaternion.md` - Added sandwich function documentation
6. `README.md` - Added sandwich product to quaternion key features
7. `CHANGELOG.md` - Added sandwich product under Unreleased

### New Functions

- `sandwich(rotor, v)` - Applies rotor * v * rotor^-1 transformation
- `sandwich_unit(rotor, v)` - Efficient version for unit quaternions using conjugate

### Tests Added

- `test_sandwich_identity` - Identity rotor preserves vector
- `test_sandwich_unit_identity` - Same for unit version
- `test_sandwich_inverse_roundtrip` - sandwich(q, sandwich(q^-1, v)) == v
- `test_sandwich_unit_equivalent` - Both versions give same result for unit quaternions
- `test_sandwich_composition` - sandwich(q2, sandwich(q1, v)) == sandwich(q2*q1, v)
- `test_sandwich_preserves_unit_length` - Output stays normalized
- `test_sandwich_ops_method` - Class method matches standalone function
- `test_sandwich_unit_ops_method` - Same for unit version
- `test_sandwich_different_from_bind` - Sandwich != simple binding
- `test_sandwich_single_quaternion` - Works on single quaternions

### Verification

- All 33 quaternion tests pass
- mypy: no issues
- ruff: all checks passed
