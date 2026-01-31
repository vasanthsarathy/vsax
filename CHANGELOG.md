# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.1] - 2026-01-31

### Added
- **Sandwich Product for Quaternion Hypervectors:**
  - `sandwich(rotor, v)` - Apply rotor transformation: `rotor * v * rotor⁻¹`
  - `sandwich_unit(rotor, v)` - Efficient version for unit quaternions using conjugate
  - Convenience methods added to `QuaternionOperations` class
  - Useful for learning state transformations from (s, action, s') traces
  - Properties: identity preservation, round-trip, composition
  - 10 new tests for sandwich product operations
  - Exported from `vsax.ops`

## [1.4.0] - 2026-01-30

### Added
- **Quaternion Hypervectors (QHV) - Non-Commutative VSA:**
  - `QuaternionHypervector` - Unit quaternion representation on S³ manifold
  - `QuaternionOperations` - Hamilton product-based operations (non-commutative binding)
  - `sample_quaternion_random()` - Sample random unit quaternion hypervectors
  - `quaternion_similarity()` - Similarity metric for quaternion vectors
  - `create_quaternion_model()` - Factory function for quaternion VSA model

- **Non-Commutative Binding:**
  - `bind(x, y) ≠ bind(y, x)` - Order matters in quaternion binding
  - Enables natural encoding of role/filler, subject/object relationships
  - Hamilton product: `q = a + bi + cj + dk` with `i² = j² = k² = ijk = -1`

- **Left and Right Unbinding:**
  - `unbind(z, y)` - Right-unbind: recover x from z = bind(x, y) given y
  - `unbind_left(x, z)` - Left-unbind: recover y from z = bind(x, y) given x
  - Both achieve >99% similarity recovery for exact unbinding
  - `unbind_left()` added to `AbstractOpSet` base class (backward compatible)

- **Documentation:**
  - User guide: `docs/guide/quaternion.md` - Complete tutorial with examples
  - API reference: `docs/api/ops/quaternion.md`, `docs/api/representations/quaternion.md`
  - Updated `docs/guide/representations.md` with quaternion section

- **Tests:**
  - 105 new tests for quaternion algebra, operations, representation, sampling, similarity
  - Integration tests for quaternion model with VSAMemory
  - Tests for non-commutative binding, left/right unbinding, bundling
  - Total: 747 tests with 94.65% coverage

### Technical Details
- Storage format: `(D, 4)` where D is dimensionality, 4 is quaternion components (a, b, c, d)
- Unit quaternion constraint ensures stability on S³
- Hamilton product is associative but non-commutative
- Binding preserves unit length for unit quaternions

### Notes
- No breaking changes from v1.3.1
- Quaternion VSA complements existing FHRR, MAP, and Binary models
- Use when binding order matters (role/filler, subject/object relationships)

## [1.3.1] - 2026-01-01

### Fixed
- **CRITICAL BUG FIX: FHRR unbinding and inverse operations corrected**
  - **Issue:** Both `unbind()` and `inverse()` methods were missing division by |FFT(b)|² in circular deconvolution
  - **Was:** `ifft(fft(a) * conj(fft(b)))` (mathematically incorrect)
  - **Now:** `ifft(fft(a) * conj(fft(b)) / (|fft(b)|² + ε))` (mathematically correct)
  - **Impact:**
    - Norm explosion eliminated (was 16,196, now 22.6)
    - Unbinding similarity dramatically improved (was ~0.70, now >0.99)
    - Proper recovery of original vectors after bind-unbind operations
  - **Mathematical explanation:** For proper circular deconvolution:
    - If C = A ⊙ B (element-wise multiplication in frequency domain = circular convolution)
    - Then to recover A: A = C ⊙ conj(B) / |B|²
    - Simply multiplying by conjugate without magnitude normalization is incorrect
  - **Numerical stability:** Added epsilon (1e-10) to prevent division by zero
  - **Files changed:**
    - `vsax/ops/fhrr.py`: Updated `unbind()` and `inverse()` methods
    - `tests/ops/test_fhrr.py`: Updated test expectations to match corrected behavior
  - **All tests pass:** 642 tests passing with 94.35% coverage

### Notes
- This is a bug fix release that corrects fundamental mathematics in FHRR circular deconvolution
- Users should see immediate improvement in unbinding accuracy
- No breaking API changes - existing code will automatically benefit from the fix

## [1.3.0] - 2025-12-30

### Added
- **Unbinding Operations:**
  - `unbind()` method added to `AbstractOpSet` for explicit unbinding interface
  - `FHRROperations.unbind()` - Optimized FFT-based circular deconvolution
  - `BinaryOperations.unbind()` - Leverages XOR self-inverse property
  - `MAPOperations` uses default implementation from AbstractOpSet
  - Provides clearer API: `ops.unbind(c, b)` vs `ops.bind(c, ops.inverse(b))`

- **FHRR Sampling Function:**
  - `sample_fhrr_random()` - Generates real-valued vectors with conjugate symmetry
  - Ensures IFFT produces real results (no complex artifacts)
  - Enforces frequency-domain properties: `F[k] = conj(F[D-k])`
  - DC and Nyquist components are real-valued
  - Enables >99% unbinding accuracy for FHRR operations
  - Samples unit-magnitude phasors in frequency domain

- **Comprehensive Test Coverage:**
  - 30+ new tests validating unbinding mathematical correctness
  - `test_unbind_perfect_with_fhrr_vectors()` - Demonstrates >99% accuracy
  - `test_inverse_frequency_domain_conjugate()` - Validates correct mathematics
  - `test_sample_fhrr_random_conjugate_symmetry()` - Validates sampling properties
  - Tests for FHRR, MAP, and Binary unbinding operations
  - Round-trip accuracy tests, chain unbinding tests

### Fixed
- **BREAKING: FHRR inverse operation corrected (CRITICAL MATHEMATICAL FIX)**
  - **Was:** `conj(a)` (time-domain conjugate, mathematically incorrect)
  - **Now:** `ifft(conj(fft(a)))` (frequency-domain conjugate, mathematically correct)
  - **Impact:** FHRR unbinding accuracy improves from ~50-70% to >99% (with proper vectors)
  - **Why this is correct:** For circular convolution unbinding, the inverse must use
    frequency-domain conjugate: `inverse(b) = ifft(conj(fft(b)))`
  - **Migration:** No code changes required - unbinding automatically improves.
    For best results, use `sample_fhrr_random()` instead of `sample_complex_random()`
  - This fixes fundamental circular convolution mathematics that should have been
    correct from the start. The old implementation gave poor unbinding results.

### Changed
- **FHRR test expectations updated:**
  - General complex phasor vectors: ~70% unbinding similarity (acceptable)
  - Proper FHRR vectors (`sample_fhrr_random`): >99% unbinding similarity (ideal)
  - Tests now document the difference between general complex vectors and
    FHRR-specific vectors with conjugate symmetry

### Notes
- **Breaking change justification:** The old inverse behavior was mathematically
  wrong for FHRR circular convolution. Users may see different (better) results,
  but this is the correct behavior.
- **Recommended migration:** Use `sample_fhrr_random()` for FHRR applications to
  achieve optimal >99% unbinding accuracy. Old `sample_complex_random()` still
  works but achieves lower accuracy (~70%) due to lack of conjugate symmetry.

## [1.2.1] - 2025-01-24

### Added
- **Documentation:**
  - User Guide for Fractional Power Encoding (`docs/guide/fpe.md`)
  - User Guide for Spatial Semantic Pointers (`docs/guide/spatial.md`)
  - User Guide for Vector Function Architecture (`docs/guide/vfa.md`)
  - API Reference for FractionalPowerEncoder (`docs/api/encoders/fpe.md`)
  - API Reference for spatial module (`docs/api/spatial/index.md`)
  - API Reference for VFA module (`docs/api/vfa/index.md`)
  - Step 8 in CLAUDE.md for documentation deployment workflow

### Fixed
- Navigation links in mkdocs.yml now include FPE, SSP, and VFA guides
- Documentation now properly displays all v1.2.0 features

## [1.2.0] - 2025-01-24

### Added
- **Fractional Power Encoding (FPE) - Phase 1:**
  - `fractional_power()` method added to FHRROperations for continuous value encoding
  - `FractionalPowerEncoder` - General encoder for fractional power encoding (`v^r`)
  - Enables continuous spatial and function representations
  - True fractional powers via phase rotation: `exp(i*θ)^r = exp(i*r*θ)`
  - Properties: continuous, compositional, invertible
  - Full type safety: requires ComplexHypervector (FHRR) model
  - 28 new tests for FHRR operations, 26 tests for FractionalPowerEncoder

- **Spatial Semantic Pointers (SSP) - Phase 2:**
  - `vsax.spatial` module for continuous spatial encoding (Komer et al. 2019)
  - `SpatialSemanticPointers` - 1D, 2D, 3D spatial encoding with `X^x ⊗ Y^y` representation
  - `SSPConfig` - Configuration for spatial dimensions and axis names
  - Spatial utilities:
    - `create_spatial_scene()` - Bundle multiple object-location pairs
    - `similarity_map_2d()` - Generate 2D similarity heatmaps
    - `plot_ssp_2d_scene()` - Matplotlib visualization
    - `region_query()` - Find objects within spatial regions
  - Query operations:
    - "What is at (x, y)?" - Query objects at locations
    - "Where is X?" - Query locations of objects
  - Scene transformations:
    - `shift_scene()` - Translate entire scene
  - Location decoding with grid search
  - 30 tests for SSP core, 17 tests for utilities (100% & 74% coverage)

- **Vector Function Architecture (VFA) - Phase 3:**
  - `vsax.vfa` module for function approximation in RKHS (Frady et al. 2021)
  - **Kernel Abstraction:**
    - `KernelType` enum (UNIFORM, GAUSSIAN, LAPLACE, CUSTOM)
    - `KernelConfig` - Configure kernel type, bandwidth, dimensionality
    - `sample_kernel_basis()` - Sample basis vectors for function encoding
    - 26 tests with 95% coverage
  - **VectorFunctionEncoder:**
    - Function encoding from samples using regularized least squares
    - Point and batch evaluation: `evaluate_1d()`, `evaluate_batch()`
    - Function arithmetic: `add_functions()` with alpha/beta coefficients
    - Function shifting: `shift_function()` for f(x) → f(x - shift)
    - Function convolution: `convolve_functions()`
    - 29 tests with 100% coverage
  - **VFA Applications:**
    - `DensityEstimator` - Kernel density estimation with fit()/evaluate()
    - `NonlinearRegressor` - Scikit-learn-like API with fit()/predict()/score()
    - `ImageProcessor` - Image encoding/decoding/blending
    - 30 tests with 100% coverage
  - Represents functions as `f(x) ≈ Σ α_i * z_i^x` in RKHS
  - Compact hypervector representation of entire functions

- **Examples (Phase 4):**
  - **Spatial Examples:**
    - `examples/spatial/ssp_1d_line.py` - 1D spatial encoding tutorial
    - `examples/spatial/ssp_2d_navigation.py` - 2D navigation with landmarks
  - **VFA Examples:**
    - `examples/vfa/density_estimation.py` - Kernel density estimation demo
    - `examples/vfa/nonlinear_regression.py` - Function regression with multiple test functions
    - `examples/vfa/image_processing.py` - Image encoding/decoding/blending

### Documentation
- Updated README.md with FPE/SSP/VFA features and examples
- Added Quick Start sections for Spatial Semantic Pointers and VFA
- Updated citation version to 1.2.0
- Added new feature highlights to feature list

### Technical Details
- **Total new tests:** 85 (26 kernels + 29 function encoder + 30 applications)
- **VFA module coverage:** 95-100% across all submodules
- **Foundations:** Komer et al. 2019 (SSP), Frady et al. 2021 (VFA)
- **Requirements:** Requires ComplexHypervector (FHRR) model for FPE/SSP/VFA
- **JAX-native:** All operations GPU-accelerated and JIT-compatible

### Summary
This release adds continuous spatial encoding and function approximation capabilities to VSAX:
- **FPE** enables encoding continuous values as fractional powers
- **SSP** provides spatial semantic pointers for navigation and spatial reasoning
- **VFA** enables function approximation in RKHS for density estimation, regression, and image processing
- All features build on ComplexHypervector (FHRR) representations
- Comprehensive examples and 85 new tests ensure reliability

No breaking changes from v1.1.0.

## [1.1.0] - 2025-01-20

### Added
- **Clifford-Inspired Operator Layer (Phase 1 - Core Infrastructure):**
  - `AbstractOperator` - Base interface for all operators
  - `CliffordOperator` - Phase-based operator for FHRR hypervectors
  - `OperatorKind` enum - Semantic typing (SPATIAL, SEMANTIC, TEMPORAL, etc.)
  - `OperatorMetadata` - Metadata dataclass for operator information
  - Exact inversion: `op.inverse().apply(op.apply(v))` recovers original (similarity > 0.999)
  - Compositional algebra: Combine operators with `compose()`
  - FHRR-compatible: Phase rotations for complex hypervectors
  - Full type safety with mypy compliance
  - 23 comprehensive tests with 96% coverage for CliffordOperator
  - 410 total tests with 94% coverage

- **Pre-defined Operators (Phase 2):**
  - **8 Spatial Operators**: Reproducible factory functions for spatial relations
    - `create_left_of()`, `create_right_of()`
    - `create_above()`, `create_below()`
    - `create_in_front_of()`, `create_behind()`
    - `create_near()`, `create_far()`
    - `create_spatial_operators()` - Create all spatial operators at once
  - **8 Semantic Role Operators**: Reproducible factory functions for semantic roles
    - `create_agent()`, `create_patient()`, `create_theme()`, `create_experiencer()`
    - `create_instrument()`, `create_location()`, `create_goal()`, `create_source()`
    - `create_semantic_operators()` - Create all semantic operators at once
  - All operators are reproducible: same dimension → same parameters
  - Inverse pairs work perfectly (e.g., LEFT_OF/RIGHT_OF)
  - Full integration with VSAMemory and encoders
  - 40 new comprehensive tests
  - 450 total tests with 95% coverage

- **Examples and Documentation (Phase 3):**
  - **4 Complete Runnable Examples** (1,157 lines total):
    - `examples/operators/spatial_reasoning.py` - Spatial operators demo (213 lines)
    - `examples/operators/semantic_roles.py` - Semantic role labeling (318 lines)
    - `examples/operators/typed_graph.py` - Knowledge graph reasoning (302 lines)
    - `examples/operators/operator_algebra.py` - Algebraic properties (324 lines)
  - **Comprehensive Documentation**:
    - Tutorial 10: Clifford Operators - Complete hands-on tutorial
    - User Guide: docs/guide/operators.md - When and how to use operators
    - API Reference: docs/api/operators/index.md - Full API documentation
  - All examples demonstrate real-world use cases with clear output

### Technical Details
- Operators represent "what happens" (transformations, relations, actions)
- Hypervectors represent "what exists" (concepts, objects, symbols)
- Phase-based implementation: `apply(v) = v * exp(i * params)`
- Composition via phase addition: associative and commutative
- Immutable design with frozen dataclasses
- JAX-native for GPU acceleration and JIT compilation

### Fixed
- Cross-platform test compatibility: Adjusted similarity thresholds to account for numerical precision differences between Python versions (3.9 vs 3.14)

### Documentation
- Updated README.md with operators example and pre-defined operators
- Added operators to feature list with usage examples
- Updated test coverage numbers (387 → 450 tests, 94% → 95% coverage)
- All examples verified to run on Windows without encoding issues

No breaking changes from v1.0.0.

## [1.0.0] - 2025-01-20

### Changed
- **Production Release:** VSAX is now production-ready and stable
- Development status upgraded from "Alpha" to "Production/Stable"
- Version updated to 1.0.0 across all package files
- All core features complete with 94% test coverage (387 tests)
- Comprehensive documentation with 9 tutorials and complete API reference
- Full type safety with mypy compliance
- Ready for production use in research and applications

### Summary
This release marks VSAX as feature-complete and production-ready. The library provides:
- Three complete VSA models (FHRR, MAP, Binary)
- Five core encoders (Scalar, Sequence, Set, Dict, Graph)
- Resonator networks for factorization
- GPU acceleration and device management
- Persistence and I/O
- Comprehensive documentation and tutorials

No breaking changes from v0.7.2.

## [0.7.2] - 2025-01-16

### Added
- **GPU Device Management:**
  - `get_device_info()` - Get information about available JAX devices
  - `print_device_info()` - Print detailed device information
  - `ensure_gpu()` - Check if GPU is available and warn if not
  - `get_array_device()` - Get the device where an array is stored
  - `benchmark_operation()` - Benchmark operations on specific devices
  - `compare_devices()` - Compare CPU vs GPU performance
  - `print_benchmark_results()` - Pretty-print benchmark results
- Comprehensive GPU usage guide (docs/guide/gpu_usage.md)
- GPU benchmarking section in MNIST tutorial
- 10 new tests for device utilities
- 387 total tests with 94% coverage

### Changed
- Updated package version from 0.7.1 to 0.7.2
- Enhanced documentation with GPU device management guides

## [0.7.1] - 2025-01-15

### Fixed
- Fixed mypy type errors in resonator module
- Made FHRR resonator test platform-agnostic to handle numerical precision differences across platforms
- Added proper type annotations and type narrowing for Union return types

## [0.7.0] - 2025-01-15

### Added
- **Resonator Networks:**
  - `CleanupMemory` - Codebook projection for finding nearest vectors
  - `Resonator` - Iterative factorization algorithm for VSA composites
  - Supports all 3 VSA models (Binary, FHRR, MAP)
  - Factorization of 2-3 factor composites
  - Superposition initialization from Frady et al. (2020)
  - Convergence detection and history tracking
  - Batch processing with `factorize_batch()`
- Comprehensive examples (examples/resonator_tree_search.py)
- User guide for resonator networks (docs/guide/resonator.md)
- API documentation (docs/api/resonator/index.md)
- 38 new tests for resonator module
- 377 total tests with 96% coverage

### Changed
- Updated package version from 0.6.0 to 0.7.0
- Enhanced documentation with resonator network guides

## [0.6.0] - 2025-01-14

### Added
- **I/O and Persistence:**
  - `save_basis()` - Save VSAMemory to JSON file
  - `load_basis()` - Load VSAMemory from JSON file
  - JSON serialization for all 3 VSA models (FHRR, MAP, Binary)
  - Round-trip vector preservation with exact accuracy
  - Dimension and representation type validation
  - Human-readable JSON format
- Comprehensive persistence example (examples/persistence.py)
- User guide for persistence (docs/guide/persistence.md)
- API documentation for I/O (docs/api/io/index.md)
- 20 new tests for save/load functionality
- 339 total tests with 96% coverage

### Changed
- Updated package version from 0.5.0 to 0.6.0
- Enhanced documentation with persistence guides

## [0.5.0] - 2025-01-XX

### Added
- **Similarity metrics:**
  - `cosine_similarity()` - Cosine similarity for all vector types
  - `dot_similarity()` - Unnormalized dot product similarity
  - `hamming_similarity()` - Hamming similarity for binary vectors
- **Batch operations with JAX vmap:**
  - `vmap_bind()` - GPU-accelerated batch binding
  - `vmap_bundle()` - GPU-accelerated batch bundling
  - `vmap_similarity()` - Efficient similarity search across batches
- **Visualization utilities:**
  - `pretty_repr()` - Pretty-print hypervectors
  - `format_similarity_results()` - Format similarity search results
- Comprehensive examples: similarity_search.py, batch_operations.py
- User guides for similarity metrics and batch operations
- Pre-commit check scripts (check.sh, check.ps1)
- 319 tests with 95%+ coverage

### Changed
- Updated package version from 0.4.2 to 0.5.0
- Enhanced documentation with similarity and batch operation guides
- Added ruff ignore rules for scientific notation (uppercase X, Y)

## [0.4.0] - 2025-01-XX

### Added
- **5 Core Encoders:**
  - `ScalarEncoder` - Numeric values with power encoding
  - `SequenceEncoder` - Ordered sequences (lists, tuples)
  - `SetEncoder` - Unordered collections (sets)
  - `DictEncoder` - Key-value pairs (dictionaries)
  - `GraphEncoder` - Graph structures (edge lists)
- `AbstractEncoder` - Base class for custom encoders
- Complete integration examples for all 3 VSA models
- Custom encoder examples (DateEncoder, ColorEncoder)
- 280+ tests with 92%+ coverage

### Changed
- Updated package version from 0.3.0 to 0.4.0
- Enhanced documentation with encoder guides and examples

## [0.3.0] - 2025-01-XX

### Added
- `VSAMemory` - Dictionary-style symbol management and basis storage
- Factory functions:
  - `create_fhrr_model()` - One-line FHRR model creation
  - `create_map_model()` - One-line MAP model creation
  - `create_binary_model()` - One-line Binary model creation
- Utility functions:
  - `coerce_to_array()` - Convert hypervectors to arrays
  - `coerce_many()` - Batch conversion
- 230 tests with 89% coverage
- Comprehensive user guides for models and memory

### Changed
- Updated package version from 0.2.0 to 0.3.0
- Simplified API with factory functions

## [0.2.0] - 2025-01-XX

### Added
- **All three VSA representations:**
  - `ComplexHypervector` - phase-based representation for FHRR
  - `RealHypervector` - continuous real-valued representation for MAP
  - `BinaryHypervector` - bipolar {-1, +1} or binary {0, 1} representation
- **Complete operation sets:**
  - `FHRROperations` - FFT-based circular convolution, complex conjugate inverse
  - `MAPOperations` - element-wise multiplication and mean
  - `BinaryOperations` - XOR binding, majority voting bundling
- **Sampling functions:**
  - `sample_random()` - normal distribution sampling for real vectors
  - `sample_complex_random()` - unit-magnitude complex vectors with random phases
  - `sample_binary_random()` - bipolar or binary random vectors
- Comprehensive test suite with 175 tests and 96% coverage
- Full integration tests for all model combinations
- Type-safe implementations with complete mypy compliance

### Changed
- Updated package version from 0.1.0 to 0.2.0
- Enhanced documentation with operation details

## [0.1.0] - 2025-XX-XX

### Added
- Project foundation and infrastructure
- Abstract base classes (AbstractHypervector, AbstractOpSet)
- VSAModel dataclass for defining VSA algebras
- Package structure for vsax library
- Abstract interfaces for VSA components
- Full testing infrastructure with pytest
- CI/CD pipeline with GitHub Actions
- Documentation site with MkDocs
- Development tooling (ruff, mypy, pytest-cov)
- GitHub workflows for CI and publishing

[Unreleased]: https://github.com/vasanthsarathy/vsax/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/vasanthsarathy/vsax/compare/v0.7.2...v1.0.0
[0.7.2]: https://github.com/vasanthsarathy/vsax/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/vasanthsarathy/vsax/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/vasanthsarathy/vsax/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/vasanthsarathy/vsax/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/vasanthsarathy/vsax/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/vasanthsarathy/vsax/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/vasanthsarathy/vsax/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/vasanthsarathy/vsax/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/vasanthsarathy/vsax/releases/tag/v0.1.0
