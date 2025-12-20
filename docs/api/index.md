# API Reference

Complete API documentation for VSAX v1.0.0.

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

## Memory & Utilities

- **[VSAMemory](core/memory.md)** - Symbol table and basis management
- **[Factory Functions](core/factory.md)** - Easy model creation

## Encoders

- **[ScalarEncoder](encoders/scalar.md)** - Encode numeric values
- **[SequenceEncoder](encoders/sequence.md)** - Encode ordered sequences
- **[SetEncoder](encoders/set.md)** - Encode unordered collections
- **[DictEncoder](encoders/dict.md)** - Encode key-value pairs
- **[GraphEncoder](encoders/graph.md)** - Encode graph structures
- **[AbstractEncoder](encoders/base.md)** - Base class for custom encoders

## Similarity

- **[Similarity Functions](similarity/index.md)** - Cosine, dot, Hamming similarity

## Resonator Networks

- **[CleanupMemory & Resonator](resonator/index.md)** - Codebook projection and iterative factorization

## I/O & Persistence

- **[Save/Load Functions](io/index.md)** - JSON serialization for basis vectors

## Utilities

- **[Batch Operations](utils/index.md)** - vmap_bind, vmap_bundle, vmap_similarity
- **[Visualization](utils/index.md)** - pretty_repr, format_similarity_results


## Quick Links

- [Getting Started](../getting-started.md)
- [User Guide](../guide/representations.md)
- [Examples](../examples/fhrr.md)
