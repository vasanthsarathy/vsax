# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Technical Details
- Operators represent "what happens" (transformations, relations, actions)
- Hypervectors represent "what exists" (concepts, objects, symbols)
- Phase-based implementation: `apply(v) = v * exp(i * params)`
- Composition via phase addition: associative and commutative
- Immutable design with frozen dataclasses
- JAX-native for GPU acceleration and JIT compilation

### Documentation
- Updated README.md with operators example
- Added operators to feature list
- Updated test coverage numbers (387 → 410 tests)

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
