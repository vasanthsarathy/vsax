# API Reference

Complete API documentation for VSAX v0.2.0.

## Core Components

- **[Base Classes](core/base.md)** - Abstract interfaces (AbstractHypervector, AbstractOpSet)
- **[VSAModel](core/model.md)** - Immutable model container

## Representations

- **[ComplexHypervector](representations/complex.md)** - Complex-valued phase-based representation
- **[RealHypervector](representations/real.md)** - Real-valued continuous representation
- **[BinaryHypervector](representations/binary.md)** - Binary/bipolar discrete representation

## Operations

- **[FHRROperations](ops/fhrr.md)** - FFT-based circular convolution
- **[MAPOperations](ops/map.md)** - Element-wise multiply and mean
- **[BinaryOperations](ops/binary.md)** - XOR and majority voting

## Sampling

- **[Sampling Functions](sampling.md)** - Random vector generation

## Coming Soon

- **Encoders** (Iteration 4) - ScalarEncoder, DictEncoder
- **Similarity** (Iteration 5) - Cosine, dot, Hamming similarity
- **I/O** (Iteration 6) - Save and load basis vectors

## Quick Links

- [Getting Started](../getting-started.md)
- [User Guide](../guide/representations.md)
- [Examples](../examples/fhrr.md)
