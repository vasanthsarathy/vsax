# VSAX vs Other HDC/VSA Libraries

This document explains VSAX's design philosophy and how it compares to other open-source hyperdimensional computing libraries.

## TL;DR - When to Use VSAX

**Choose VSAX if you want:**
- ✅ JAX-native functional programming with automatic GPU acceleration
- ✅ Clean separation between representations and operations
- ✅ Composable, modular architecture for research and prototyping
- ✅ Strong theoretical grounding (implements canonical VSA models)
- ✅ Type-safe, well-documented API with 94% test coverage
- ✅ Resonator networks for factorization
- ✅ Seamless integration with JAX ecosystem (jit, vmap, grad)

**Choose alternatives if you need:**
- ❌ PyTorch integration → **torchhd**
- ❌ Production ML classifiers with 150+ datasets → **torchhd**
- ❌ Biomedical/medical informatics focus → **hdlib**
- ❌ Advanced boolean operations and circuit compilation → **PyBHV**
- ❌ Custom CUDA kernels → **hdtorch**

---

## VSAX's Design Philosophy

VSAX is built on three core principles:

### 1. **JAX-Native Functional Programming**

Unlike PyTorch-based libraries (torchhd, hdtorch), VSAX is built entirely on JAX:

```python
# JAX provides automatic differentiation, JIT compilation, and vectorization
from jax import jit, vmap, grad
import jax.numpy as jnp

# VSAX operations are pure functions
result = model.opset.bind(a, b)  # Functional, composable

# Automatic GPU acceleration - no explicit device management
@jit
def fast_encoding(vectors):
    return vmap(model.opset.bundle)(vectors)
```

**Why JAX?**
- **Functional purity**: No hidden state, easier to reason about
- **Automatic transformations**: `jit`, `vmap`, `grad` work out of the box
- **Research-friendly**: Designed for ML research at Google/DeepMind
- **NumPy-like API**: Familiar interface, minimal learning curve

### 2. **Modular Architecture**

VSAX cleanly separates concerns:

```python
# Representations (data)
ComplexHypervector, RealHypervector, BinaryHypervector

# Operations (algorithms)
FHRROperations, MAPOperations, BinaryOperations

# Model (composition)
VSAModel(dim, rep_cls, opset, sampler)
```

This is different from torchhd's integrated approach where models are classes with built-in operations.

**Benefit**: Mix and match components:
- Try different operations with the same representation
- Swap representations without changing code
- Easy to add new VSA models

### 3. **Simplicity and Clarity**

VSAX prioritizes **understanding** over **features**:

- **3 canonical VSA models** (FHRR, MAP, Binary) implemented correctly
- **Clear abstractions**: Every operation has a mathematical meaning
- **Comprehensive tutorials**: Learn VSA concepts, not just API calls
- **Theory-first**: Based on foundational papers (Plate, Gayler, Kanerva)

---

## Feature Comparison

### Supported VSA Models

| Library | FHRR | MAP | Binary | HRR | Others |
|---------|------|-----|--------|-----|--------|
| **VSAX** | ✅ | ✅ | ✅ | ❌ | - |
| **torchhd** | ✅ | ✅ | ✅ (BSC) | ✅ | B-SBC, CGR, MCR, VTB |
| **hdlib** | ❓ | ❓ | ❓ | ❓ | General VSA |
| **PyBHV** | ❌ | ❌ | ✅ | ❌ | Boolean only |
| **hdtorch** | ❓ | ❓ | ✅ | ❓ | Focus on CUDA ops |

**VSAX focuses on quality over quantity**: 3 well-implemented models vs 8+ models with varying documentation.

### Core Operations

| Feature | VSAX | torchhd | hdlib | PyBHV | hdtorch |
|---------|------|---------|-------|-------|---------|
| Binding | ✅ | ✅ | ✅ | ✅ (XOR) | ✅ |
| Bundling | ✅ | ✅ | ✅ | ✅ (Majority) | ✅ |
| Permutation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Similarity | ✅ | ✅ | ✅ | ✅ | ✅ |
| Resonator Networks | ✅ | ❌ | ❌ | ❌ | ❌ |
| Memory/Cleanup | ✅ | ✅ | ✅ | ❌ | ❌ |

**VSAX unique feature**: Full implementation of resonator networks for factorization (from Frady et al. 2020).

### Encoders

| Feature | VSAX | torchhd | hdlib | PyBHV | hdtorch |
|---------|------|---------|-------|-------|---------|
| Scalar | ✅ | ✅ (Level, Thermometer) | ✅ | ❌ | ✅ |
| Sequence | ✅ | ✅ | ✅ | ❌ | ✅ |
| Set | ✅ | ✅ (Multiset) | ✅ | ❌ | ❌ |
| Dict/Record | ✅ | ❌ | ❌ | ❌ | ❌ |
| Graph | ✅ | ✅ | ✅ | ✅ | ❌ |
| Tree | ❌ | ✅ | ❌ | ❌ | ❌ |
| FSA | ❌ | ✅ | ❌ | ❌ | ❌ |

**VSAX strength**: Clean, extensible encoder API with `AbstractEncoder` base class.

### Machine Learning

| Feature | VSAX | torchhd | hdlib | PyBHV | hdtorch |
|---------|------|---------|-------|-------|---------|
| Classification | ❌ | ✅ (9+ types) | ✅ | ✅ | ✅ (Basic) |
| Built-in Datasets | ❌ | ✅ (150+) | ✅ (Some) | ❌ | ❌ |
| Online Learning | ❌ | ✅ (OnlineHD) | ❌ | ❌ | ❌ |
| Neural Integration | ❌ | ✅ (NeuralHD) | ❌ | ❌ | ❌ |
| Regression | ❌ | ❌ | ✅ | ❌ | ❌ |
| Clustering | ❌ | ❌ | ✅ | ❌ | ❌ |

**Biggest VSAX gap**: No built-in classifiers or ML workflows (yet).

**torchhd is the clear winner** for production ML applications.

### Performance & Hardware

| Feature | VSAX | torchhd | hdlib | PyBHV | hdtorch |
|---------|------|---------|-------|-------|---------|
| GPU Support | ✅ (JAX auto) | ✅ (PyTorch) | ❌ | ✅ (PyTorch backend) | ✅ (Custom CUDA) |
| CPU Fallback | ✅ | ✅ | ✅ | ✅ | ❌ |
| Batch Operations | ✅ (vmap) | ✅ | ✅ | ✅ | ✅ |
| JIT Compilation | ✅ (JAX) | ✅ (TorchScript) | ❌ | ❌ | ✅ |
| Custom Kernels | ❌ | ❌ | ❌ | ✅ (C++) | ✅ (CUDA) |

**VSAX uses JAX's automatic GPU dispatch** - no manual device management.

### Developer Experience

| Feature | VSAX | torchhd | hdlib | PyBHV | hdtorch |
|---------|------|---------|-------|-------|---------|
| Type Hints | ✅ (Full) | ✅ | ❓ | ❌ | ❓ |
| Test Coverage | 94% | ❓ | ❓ | ❓ | ❓ |
| Documentation | ✅ | ✅ | ✅ (Wiki) | ✅ | ✅ |
| Tutorials | ✅ (3 deep) | ✅ | ✅ | ✅ (Examples) | ✅ |
| Examples | ✅ | ✅ | ✅ | ✅ (Many) | ✅ |

**VSAX prioritizes code quality**: Type-safe, well-tested, thoroughly documented.

---

## Detailed Library Comparison

### torchhd: The Production ML Library

**Best for**: Machine learning applications, classification tasks, production deployment

**Strengths**:
- **Comprehensive**: 8 VSA models, 9+ classifiers, 150+ datasets
- **Production-ready**: Battle-tested with active community (346 stars)
- **PyTorch integration**: Seamless with existing PyTorch workflows
- **Rich structures**: Graph, Tree, FSA, HashTable implementations
- **Well-documented**: Extensive tutorials and examples

**Weaknesses**:
- **Complexity**: Large API surface, steeper learning curve
- **PyTorch-coupled**: Hard to use without PyTorch knowledge
- **Less modular**: Models are monolithic classes

**When to choose over VSAX**:
- You need production ML classifiers
- You're already using PyTorch
- You want ready-made datasets
- You need advanced structures (Tree, FSA)

### hdlib: The Biomedical Specialist

**Best for**: Biomedical applications, bioinformatics, medical informatics

**Strengths**:
- **Domain focus**: Proven in cancer classification, metagenomics
- **Versatile**: Classification, regression, clustering, feature selection
- **Academic backing**: Peer-reviewed publications
- **Easy install**: PyPI and conda-forge

**Weaknesses**:
- **Less clear**: VSA model support not well documented
- **No GPU**: CPU-only implementation
- **Older codebase**: Less active maintenance

**When to choose over VSAX**:
- You're working in bioinformatics/medical AI
- You need regression or clustering
- You want proven biomedical applications

### PyBHV: The Boolean Specialist

**Best for**: Boolean operations, symbolic reasoning, theoretical research

**Strengths**:
- **Research framework**: Expression simplification, circuit compilation
- **Multiple backends**: Python, C++, NumPy, PyTorch with bit-packing
- **Rich metrics**: Comprehensive distance and similarity measures
- **Symbolic computing**: Law-based testing and optimization
- **Memory efficient**: 8x compression with bit-packing

**Weaknesses**:
- **Boolean only**: No support for real or complex hypervectors
- **Narrow focus**: Limited to binary VSA
- **Complex API**: Many abstraction levels

**When to choose over VSAX**:
- You only need boolean/binary hypervectors
- You want circuit compilation or logic synthesis
- You need bit-level optimization
- You're doing theoretical VSA research

### hdtorch: The CUDA Accelerator

**Best for**: Custom GPU kernels, maximum performance

**Strengths**:
- **Custom CUDA**: Hand-optimized GPU kernels
- **Performance**: Fastest for supported operations
- **Educational**: Clear tutorials on CUDA implementation

**Weaknesses**:
- **Limited scope**: Fewer features than torchhd or VSAX
- **CUDA required**: No CPU fallback
- **Less mature**: Smaller community

**When to choose over VSAX**:
- You need maximum GPU performance
- You want to learn CUDA kernel programming
- You're willing to trade features for speed

---

## What Makes VSAX Unique?

### 1. **JAX-First Design**

VSAX is the **only JAX-native VSA library**:

```python
# Automatic GPU acceleration
model = create_fhrr_model(dim=512)  # Works on GPU if available

# JIT compilation for speed
@jit
def encode_batch(items):
    return vmap(encoder.encode)(items)

# Automatic differentiation (future: differentiable VSA)
gradient = grad(lambda x: similarity(x, target))
```

**Why this matters**:
- JAX is the future of ML research (used by Google, DeepMind)
- Functional programming = easier reasoning
- Better for research and prototyping

### 2. **Clean Theoretical Foundation**

VSAX implements the **canonical VSA models** from foundational papers:

- **FHRR**: Plate (1995) - Complex-valued circular convolution
- **MAP**: Gayler (1998) - Multiply-Add-Permute
- **Binary**: Kanerva (1996) - Binary Spatter Codes

Each implementation is **mathematically correct** and **well-documented**.

### 3. **Resonator Networks**

VSAX is the **only library with full resonator support**:

```python
# Factorize compositional structures
resonator = Resonator(model, codebooks=[subjects, relations, objects])
factors = resonator.factorize(composite_vector)
# ['dog', 'isA', 'mammal']
```

Based on Frady et al. (2020), resonators enable:
- Decoding compositional structures
- Iterative refinement with convergence
- Multi-factor factorization

### 4. **Tutorial-Driven Documentation**

VSAX teaches **VSA concepts**, not just API:

1. **MNIST Classification**: Learn encoding and prototypes
2. **Knowledge Graphs**: Understand binding and bundling
3. **Kanerva's Analogies**: Master mappings and transformations

Each tutorial implements **foundational papers** with full code.

### 5. **Research-Friendly Architecture**

VSAX makes it **easy to experiment**:

```python
# Try different operations with same representation
model1 = VSAModel(dim=512, rep_cls=ComplexHypervector,
                  opset=FHRROperations(), sampler=sample_complex_random)

model2 = VSAModel(dim=512, rep_cls=ComplexHypervector,
                  opset=MAPOperations(), sampler=sample_complex_random)

# Same API, different algebra!
```

---

## What VSAX Doesn't (Yet) Do

We're honest about gaps:

### ❌ Machine Learning Classifiers

**Missing**:
- No built-in classifiers (Centroid, AdaptHD, OnlineHD, etc.)
- No datasets
- No training loops

**Workaround**: Build your own with VSAX primitives:
```python
# Manual centroid classifier
prototypes = {label: bundle(class_examples) for label, class_examples in data}
prediction = max(prototypes, key=lambda l: similarity(query, prototypes[l]))
```

**Future**: v1.0+ will add classifiers

### ❌ Advanced Structures

**Missing**:
- Tree encoders
- Finite State Automata
- HashTable structures

**Workaround**: Use GraphEncoder as building block

**Future**: May add in v1.x based on demand

### ❌ Additional VSA Models

**Missing**:
- HRR (original Plate model without FFT)
- BSC variants (Sparse Block Codes)
- CGR, MCR, VTB from recent research

**Reason**: We prioritize **depth** (correct implementation, documentation, tests) over **breadth**

**Future**: May add models with strong theoretical foundation

### ❌ Production Optimization

**Missing**:
- Custom CUDA kernels (like hdtorch)
- Bit-packing (like PyBHV)
- Quantization/compression

**Reason**: JAX provides good-enough performance for research

**Future**: Optimization in later versions if needed

---

## Choosing the Right Library

### Use VSAX if you:
- 🎓 Want to **learn VSA deeply** with tutorial-driven examples
- 🔬 Are doing **research** and need flexibility
- 🧮 Prefer **functional programming** and JAX
- 📐 Value **theoretical correctness** over feature count
- 🧩 Need **compositional operations** (resonators, mappings)
- 💻 Want **type-safe, well-tested** code

### Use torchhd if you:
- 🏭 Need **production ML** with classifiers and datasets
- 🔥 Are already using **PyTorch**
- 📊 Want **many VSA models** to experiment with
- 🚀 Need **battle-tested** software (350+ stars)
- 🎯 Are building **classification systems**

### Use hdlib if you:
- 🧬 Work in **bioinformatics** or **medical AI**
- 📈 Need **regression** or **clustering**
- 📚 Want **proven biomedical applications**
- 🐍 Prefer simple Python without GPU

### Use PyBHV if you:
- 🔲 Only need **boolean hypervectors**
- ⚡ Want **bit-level optimization**
- 🧠 Are doing **symbolic reasoning** research
- 🔧 Need **circuit compilation**

### Use hdtorch if you:
- ⚙️ Need **custom CUDA kernels**
- 🏎️ Want **maximum GPU performance**
- 🎓 Want to **learn CUDA** programming

---

## VSAX Roadmap: Closing the Gaps

### v1.0.0 (Future)
- ✅ Basic classifiers (Centroid, kNN)
- ✅ Common datasets (MNIST, CIFAR-10)
- ✅ Training utilities

### v1.1.0 (Future)
- ✅ Tree and FSA encoders
- ✅ Additional VSA models (HRR, BSC variants)

### v2.0.0 (Future)
- ✅ Advanced classifiers (OnlineHD, AdaptHD)
- ✅ Performance optimizations
- ✅ Production tooling

**Guiding principle**: Maintain simplicity and theoretical clarity while adding practical features.

---

## Contributing to VSAX

We welcome contributions! Priority areas:

1. **Classifiers**: Implement standard HDC classifiers
2. **Datasets**: Add benchmark datasets with encoders
3. **Examples**: More domain applications (NLP, robotics, etc.)
4. **VSA Models**: Add models with theoretical grounding
5. **Performance**: Optimize hot paths while keeping API clean

See [CONTRIBUTING.md](https://github.com/vasanthsarathy/vsax/blob/main/CONTRIBUTING.md) for guidelines.

---

## Conclusion

**VSAX is a research-oriented, JAX-native VSA library** that prioritizes:
- ✨ **Clarity** over completeness
- 🧮 **Theory** over features
- 🔬 **Research** over production

If you need **production ML** → choose **torchhd**
If you need **biomedical apps** → choose **hdlib**
If you need **boolean operations** → choose **PyBHV**
If you need **custom CUDA** → choose **hdtorch**

**If you want to understand VSA deeply and build novel approaches** → choose **VSAX** ✨

---

## References

- **torchhd**: https://github.com/hyperdimensional-computing/torchhd
- **hdlib**: https://github.com/cumbof/hdlib
- **PyBHV**: https://github.com/Adam-Vandervorst/PyBHV
- **hdtorch**: https://hdtorch.readthedocs.io/en/latest/

---

*Last updated: 2025-01-16*
