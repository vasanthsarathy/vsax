# VSAX vs Other HDC/VSA Libraries

This document explains VSAX's design philosophy and how it compares to other open-source hyperdimensional computing libraries.

## TL;DR - When to Use VSAX

**Choose VSAX if you want:**
- ✅ JAX-native functional programming with automatic GPU acceleration
- ✅ Clean separation between representations and operations
- ✅ Composable, modular architecture for research and prototyping
- ✅ **Clifford Operators** for exact compositional reasoning
- ✅ **Spatial Semantic Pointers (SSP)** for continuous spatial encoding
- ✅ **Vector Function Architecture (VFA)** for function encoding and RKHS operations
- ✅ Full resonator networks with iterative convergence
- ✅ Type-safe, well-documented API with 94% test coverage
- ✅ Seamless integration with JAX ecosystem (jit, vmap, grad)

**Choose alternatives if you need:**
- ❌ PyTorch integration → **torchhd**
- ❌ Production ML classifiers with 160+ datasets → **torchhd**
- ❌ 8+ VSA model variants → **torchhd**
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
- **Theory-first**: Based on foundational papers (Plate, Gayler, Kanerva, Komer, Frady)

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
| Memory/Cleanup | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Resonator Networks** | ✅ Full | ✅ Single-step | ❌ | ❌ | ❌ |

**Note on Resonators**: Torchhd provides a single-step `resonator()` function for factorization, while VSAX implements full resonator networks with iterative convergence, cleanup memory, and multi-factor factorization (Frady et al. 2020).

### Advanced Capabilities (v1.2.0+)

| Feature | VSAX | torchhd | hdlib | PyBHV | hdtorch |
|---------|------|---------|-------|-------|---------|
| **Clifford Operators** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Spatial Semantic Pointers (SSP)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Vector Function Architecture (VFA)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Fractional Power Encoding** | ✅ | ✅ (Kernel-based) | ❌ | ❌ | ❌ |

**VSAX unique features (v1.2.0)**:
- **Clifford Operators**: Phase-based binding with exact invertibility (similarity > 0.999 vs 0.3-0.6)
- **Spatial Semantic Pointers**: Full 2D/3D scene encoding with object-location binding, bidirectional queries, scene transformations
- **Vector Function Architecture**: Complete RKHS function encoding with density estimation, nonlinear regression, image processing applications
- **Integrated FPE**: Fractional power encoding integrated with SSP and VFA for continuous spatial and functional representation

**Torchhd's FractionalPower**: Kernel-based continuous value encoding (sinc, gaussian kernels) - different scope from VSAX's integrated FPE/SSP/VFA system.

### Encoders

| Feature | VSAX | torchhd | hdlib | PyBHV | hdtorch |
|---------|------|---------|-------|-------|---------|
| Scalar | ✅ | ✅ (Level, Thermometer, Circular) | ✅ | ❌ | ✅ |
| Sequence | ✅ | ✅ | ✅ | ❌ | ✅ |
| Set | ✅ | ✅ (Multiset) | ✅ | ❌ | ❌ |
| Dict/Record | ✅ | ❌ | ❌ | ❌ | ❌ |
| Graph | ✅ | ✅ | ✅ | ✅ | ❌ |
| Tree | ❌ | ✅ | ❌ | ❌ | ❌ |
| FSA | ❌ | ✅ | ❌ | ❌ | ❌ |
| **FractionalPowerEncoder** | ✅ | ✅ | ❌ | ❌ | ❌ |

**VSAX strength**: Clean, extensible encoder API with `AbstractEncoder` base class.

### Machine Learning

| Feature | VSAX | torchhd | hdlib | PyBHV | hdtorch |
|---------|------|---------|-------|-------|---------|
| Classification | ❌ | ✅ (10+ types) | ✅ | ✅ | ✅ (Basic) |
| Built-in Datasets | ❌ | ✅ (160+) | ✅ (Some) | ❌ | ❌ |
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
| Test Coverage | 94% | 85% | ❓ | ❓ | ❓ |
| Documentation | ✅ | ✅ | ✅ (Wiki) | ✅ | ✅ |
| Tutorials | ✅ (11) | ✅ (Many) | ✅ | ✅ (Examples) | ✅ |
| Examples | ✅ | ✅ | ✅ | ✅ (Many) | ✅ |

**VSAX prioritizes code quality**: Type-safe, well-tested, thoroughly documented.

---

## Detailed Library Comparison

### torchhd: The Production ML Library

**Best for**: Machine learning applications, classification tasks, production deployment

**Strengths**:
- **Comprehensive**: 8 VSA models, 10+ classifiers, 160+ datasets
- **Production-ready**: Battle-tested with active community (350+ stars)
- **PyTorch integration**: Seamless with existing PyTorch workflows
- **Rich structures**: Graph, Tree, FSA, HashTable implementations
- **Well-documented**: Extensive tutorials and examples
- **Kernel-based FPE**: FractionalPower embedding with sinc/gaussian kernels

**Weaknesses**:
- **Complexity**: Large API surface, steeper learning curve
- **PyTorch-coupled**: Hard to use without PyTorch knowledge
- **Less modular**: Models are monolithic classes
- **No SSP/VFA**: Missing spatial semantic pointers and vector function architecture
- **No Clifford operators**: Traditional unbinding only

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

### 3. **Clifford Operators for Exact Compositional Reasoning**

VSAX is the **only library with Clifford operators**:

```python
from vsax.operators import CliffordOperator

# Create phase-based binding operator
op = CliffordOperator(model, memory, "LEFT_OF")

# Exact invertibility: similarity > 0.999
bound = op.bind("cup", "plate")
retrieved = op.unbind(bound, "cup")  # Returns "plate" with >0.999 similarity
```

Based on Aerts et al. (2007), Clifford operators enable:
- Near-perfect unbinding (>0.999 vs traditional 0.3-0.6)
- Exact compositional reasoning
- Precise spatial and semantic relations

### 4. **Spatial Semantic Pointers (SSP)**

VSAX is the **only library with complete SSP implementation**:

```python
from vsax.spatial import SpatialSemanticPointers, SSPConfig

# Configure 2D spatial encoding
config = SSPConfig(dim=512, num_axes=2)
ssp = SpatialSemanticPointers(model, memory, config)

# Encode scene: apple at (3.5, 2.1), banana at (1.0, 4.0)
scene = ssp.create_scene({
    "apple": [3.5, 2.1],
    "banana": [1.0, 4.0]
})

# Query: what is at (3.5, 2.1)?
result = ssp.query_location(scene, [3.5, 2.1])  # Returns "apple"

# Query: where is the banana?
coords = ssp.query_object(scene, "banana")  # Returns [1.0, 4.0]

# Transform: shift entire scene by (2, 2)
shifted = ssp.shift_scene(scene, [2.0, 2.0])
```

Based on Komer et al. (2019), SSPs enable:
- Continuous spatial encoding (no discretization)
- Object-location binding
- Bidirectional queries (what/where)
- Global scene transformations
- 2D/3D spatial reasoning

### 5. **Vector Function Architecture (VFA)**

VSAX is the **only library with full VFA implementation**:

```python
from vsax.vfa import VectorFunctionEncoder
from vsax.vfa.applications import DensityEstimator, NonlinearRegressor

# Encode function in RKHS
vfa = VectorFunctionEncoder(model, memory)
x = jnp.linspace(0, 2*jnp.pi, 50)
y = jnp.sin(x)
f_hv = vfa.encode_function_1d(x, y)

# Evaluate at new points
y_pred = vfa.evaluate_1d(f_hv, 1.5)

# Function arithmetic
g_hv = vfa.encode_function_1d(x, jnp.cos(x))
h_hv = vfa.add_functions(f_hv, g_hv)  # h = sin + cos

# Applications
estimator = DensityEstimator(model, memory)
estimator.fit(data_samples)
density = estimator.evaluate_batch(query_points)
```

Based on Frady et al. (2021), VFA enables:
- Functions as first-class symbolic objects
- RKHS representation: $f(x) \approx \langle \alpha, z^x \rangle$
- Function arithmetic (add, scale, shift, convolve)
- Kernel density estimation
- Nonlinear regression
- Image processing

### 6. **Full Resonator Networks**

VSAX implements **complete resonator networks** (not just single-step):

```python
from vsax.resonator import ResonatorNetwork

# Create resonator with multiple codebooks
resonator = ResonatorNetwork(
    model,
    codebooks=[subjects, relations, objects],
    max_iterations=15
)

# Factorize compositional structure
composite = bind3(dog_hv, isA_hv, mammal_hv)
factors, convergence = resonator.factorize(composite)
# Returns: [dog_hv, isA_hv, mammal_hv] with convergence metrics
```

Based on Frady et al. (2020), resonator networks provide:
- Iterative convergence (not single-step like torchhd)
- Multi-factor factorization
- Cleanup memory integration
- Convergence tracking

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

**Future**: v2.0+ will add classifiers

### ❌ Advanced Structures

**Missing**:
- Tree encoders
- Finite State Automata
- HashTable structures

**Workaround**: Use GraphEncoder as building block

**Future**: May add in v2.x based on demand

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
- 🧩 Need **compositional operations** (Clifford operators, SSP, VFA, resonators)
- 🌐 Need **continuous spatial encoding** or **function representation**
- 💻 Want **type-safe, well-tested** code
- 🔍 Need **exact unbinding** (Clifford operators)

### Use torchhd if you:
- 🏭 Need **production ML** with classifiers and datasets
- 🔥 Are already using **PyTorch**
- 📊 Want **many VSA models** to experiment with (8+ models)
- 🚀 Need **battle-tested** software (350+ stars)
- 🎯 Are building **classification systems**
- 📚 Want **extensive pre-built benchmarks**

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

### v1.2.0 (Current)
- ✅ Clifford Operators
- ✅ Fractional Power Encoding
- ✅ Spatial Semantic Pointers (SSP)
- ✅ Vector Function Architecture (VFA)
- ✅ Resonator Networks

### v2.0.0 (Future)
- ✅ Basic classifiers (Centroid, kNN)
- ✅ Common datasets (MNIST, CIFAR-10)
- ✅ Training utilities

### v2.1.0 (Future)
- ✅ Tree and FSA encoders
- ✅ Additional VSA models (HRR, BSC variants)

### v3.0.0 (Future)
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
- 🎯 **Depth** over breadth

**Unique strengths** (as of v1.2.0):
- Only library with Clifford Operators for exact compositional reasoning
- Only library with complete Spatial Semantic Pointers implementation
- Only library with Vector Function Architecture (RKHS function encoding)
- Full resonator networks with iterative convergence
- JAX-native for research and GPU acceleration

If you need **production ML** → choose **torchhd**
If you need **biomedical apps** → choose **hdlib**
If you need **boolean operations** → choose **PyBHV**
If you need **custom CUDA** → choose **hdtorch**

**If you want advanced VSA capabilities (Clifford, SSP, VFA) and to understand VSA deeply** → choose **VSAX** ✨

---

## References

- **torchhd**: https://github.com/hyperdimensional-computing/torchhd
- **hdlib**: https://github.com/cumbof/hdlib
- **PyBHV**: https://github.com/Adam-Vandervorst/PyBHV
- **hdtorch**: https://hdtorch.readthedocs.io/en/latest/

---

*Last updated: 2025-01-25 (v1.2.1)*
