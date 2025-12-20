# VSAX: Vector Symbolic Algebra for JAX

VSAX is a GPU-accelerated, JAX-native Python library for Vector Symbolic Architectures (VSAs). It provides composable symbolic representations using hypervectors, algebraic operations for binding and bundling, and encoding strategies for symbolic and structured data.

## Features

- 🚀 **Three VSA Models**: FHRR, MAP, and Binary implementations ✅
- 🔧 **Clifford Operators**: Exact, compositional, invertible transformations ✅ **NEW in v1.1.0**
- ⚡ **GPU-Accelerated**: Built on JAX for high-performance computation
- 🧩 **Modular Architecture**: Clean separation between representations and operations
- 🧬 **Complete Representations**: Complex, Real, and Binary hypervectors ✅
- ⚙️ **Full Operation Sets**: FFT-based FHRR, MAP, and XOR/majority Binary ops ✅
- 🎲 **Random Sampling**: Sampling utilities for all representation types ✅
- 💯 **Type-Safe**: Full type annotations with mypy support
- ✅ **Well-Tested**: 410 tests with 94% coverage
- 🔍 **Similarity Metrics**: Cosine, dot, and Hamming similarity
- ⚡ **Batch Operations**: GPU-accelerated vmap operations
- 💾 **I/O & Persistence**: Save/load basis vectors to JSON

## Installation

### From PyPI (Recommended)

```bash
pip install vsax
```

### From Source

```bash
git clone https://github.com/yourusername/vsax.git
cd vsax

# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

## Quick Example

### Simple API (v0.5.0)

```python
from vsax import create_fhrr_model, VSAMemory, DictEncoder
from vsax.similarity import cosine_similarity
from vsax.utils import vmap_bind
import jax.numpy as jnp

# Create model with factory function
model = create_fhrr_model(dim=512)

# Create memory for symbols
memory = VSAMemory(model)
memory.add_many(["subject", "action", "dog", "cat", "animal", "run", "jump"])

# Access and manipulate symbols
dog = memory["dog"]
animal = memory["animal"]

# Bind two concepts (circular convolution)
dog_is_animal = model.opset.bind(dog.vec, animal.vec)

# Bundle multiple concepts (sum and normalize)
pets = model.opset.bundle(memory["dog"].vec, memory["cat"].vec)

# NEW: Similarity search
similarity = cosine_similarity(memory["dog"], memory["cat"])
print(f"Dog-Cat similarity: {similarity:.3f}")

# NEW: Batch operations (GPU-accelerated)
nouns = jnp.stack([memory["dog"].vec, memory["cat"].vec])
verbs = jnp.stack([memory["run"].vec, memory["jump"].vec])
actions = vmap_bind(model.opset, nouns, verbs)  # Parallel binding!

# NEW: Encoders
encoder = DictEncoder(model, memory)
sentence = encoder.encode({"subject": "dog", "action": "run"})
```

### MAP Model (Real Hypervectors)

```python
from vsax import RealHypervector, MAPOperations, sample_random

model = VSAModel(
    dim=512,
    rep_cls=RealHypervector,
    opset=MAPOperations(),
    sampler=sample_random
)

# Element-wise multiplication for binding
# Element-wise mean for bundling
```

### Binary Model (Bipolar Hypervectors)

```python
from vsax import BinaryHypervector, BinaryOperations, sample_binary_random

model = VSAModel(
    dim=512,
    rep_cls=BinaryHypervector,
    opset=BinaryOperations(),
    sampler=sample_binary_random
)

# XOR binding (exact unbinding)
# Majority voting for bundling
```

## Development Status

**Current**: Iteration 6 Complete ✅

### Completed

**Iteration 1** (v0.1.0): Foundation & Infrastructure ✅
- ✅ Core abstract classes (AbstractHypervector, AbstractOpSet)
- ✅ VSAModel dataclass
- ✅ Package structure
- ✅ Testing infrastructure (pytest, coverage)
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Documentation site (MkDocs)

**Iteration 2** (v0.2.0): Core Algebras ✅
- ✅ All 3 representations (Complex, Real, Binary)
- ✅ All 3 operation sets (FHRR, MAP, Binary)
- ✅ Sampling utilities
- ✅ 175 comprehensive tests with 96% coverage
- ✅ Full integration tests

**Iteration 3** (v0.3.0): Models & Memory ✅
- ✅ VSAMemory for symbol storage
- ✅ Factory functions for easy model creation
- ✅ Integration utilities
- ✅ 230 tests with 89% coverage

**Iteration 4** (v0.4.0): First Usable Release ✅
- ✅ 5 Core Encoders (Scalar, Sequence, Set, Dict, Graph)
- ✅ AbstractEncoder base class
- ✅ Complete working examples for all 3 models
- ✅ Custom encoder examples
- ✅ 280+ tests with 92%+ coverage

**Iteration 5** (v0.5.0): Similarity Metrics & Utilities ✅
- ✅ Cosine, dot, and Hamming similarity functions
- ✅ Batch operations with JAX vmap (vmap_bind, vmap_bundle, vmap_similarity)
- ✅ Visualization utilities (pretty_repr, format_similarity_results)
- ✅ GPU-accelerated similarity search
- ✅ 319 tests with 95%+ coverage

**Iteration 6** (v0.6.0): I/O & Persistence ✅
- ✅ save_basis() and load_basis() functions
- ✅ JSON serialization for all 3 models
- ✅ Round-trip vector preservation
- ✅ Dimension and type validation
- ✅ 339 tests with 96% coverage

**Iteration 7** (v1.0.0): Production Release ✅
- ✅ Complete API documentation
- ✅ 9 tutorial notebooks with real datasets
- ✅ Production-ready v1.0.0 release
- ✅ 387 tests with 94% coverage

**Iteration 8** (v1.1.0): Clifford Operators ✅ **NEW**
- ✅ Clifford-inspired operator layer for exact reasoning
- ✅ Phase-based transformations for FHRR hypervectors
- ✅ Exact inversion (similarity > 0.999)
- ✅ Compositional algebra with compose()
- ✅ 410 tests with 94% coverage

## Documentation

- [Getting Started](getting-started.md) - Installation and first steps
- [Tutorials](tutorials/index.md) - Hands-on tutorials with real datasets
  - [MNIST Classification](tutorials/01_mnist_classification.md) - Image classification with VSA
  - [Knowledge Graph Reasoning](tutorials/02_knowledge_graph.md) - Multi-hop reasoning
  - [Clifford Operators](tutorials/10_clifford_operators.md) - Exact transformations ✨ **NEW in v1.1.0**
  - And 6 more tutorials covering analogies, word embeddings, edge computing, and more!
- [User Guide](guide/operators.md) - Detailed guides for all components
  - [Operators Guide](guide/operators.md) - Using Clifford operators ✨ **NEW**
- [Examples](examples/fhrr.md) - Working examples for all three models
- [API Reference](api/index.md) - Complete API documentation
  - [Operators API](api/operators/index.md) - Operators module reference ✨ **NEW**
- [Design Specification](design-spec.md) - Technical design details

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/yourusername/vsax/blob/main/CONTRIBUTING.md) for guidelines.

## License

VSAX is released under the MIT License. See [LICENSE](https://github.com/yourusername/vsax/blob/main/LICENSE) for details.

## Citation

If you use VSAX in your research, please cite:

```bibtex
@software{vsax2025,
  title = {VSAX: Vector Symbolic Algebra for JAX},
  author = {Sarathy, Vasanth},
  year = {2025},
  version = {1.1.0},
  url = {https://github.com/vasanthsarathy/vsax}
}
```
