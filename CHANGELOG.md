# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-XX-XX

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

[Unreleased]: https://github.com/yourusername/vsax/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yourusername/vsax/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/vsax/releases/tag/v0.1.0
