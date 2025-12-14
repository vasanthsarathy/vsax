# Technical Specification: VSAX - Vector Symbolic Algebra Library

## Overview
VSAX is a GPU-accelerated, JAX-native Python library for vector symbolic architectures (VSAs). It provides composable symbolic representations using hypervectors, algebraic operations for binding and bundling, and encoding strategies for symbolic and structured data. The library is designed to be modular, efficient, and extensible.

---

## Core Objectives
- Enable definition of VSA models combining representations (e.g., complex, real, binary) with algebraic operation sets (e.g., FHRR, MAP).
- Support encoding of symbolic data (scalars, dictionaries, graphs) using model-defined operations.
- Maintain a persistent, accessible store of basis hypervectors.
- Be fully compatible with JAX for high-performance, differentiable, GPU/TPU computation.
- Provide a clean API and separation of concerns.
- Be as usable and expressive as NumPy or PyTorch for symbolic computation.

---

## Architecture

### 1. VSAModel
- `dim: int` — dimensionality of all hypervectors
- `rep_cls: Type[AbstractHypervector]` — the representation class (e.g. ComplexHypervector)
- `opset: AbstractOpSet` — operation strategies (bind, bundle, inverse)
- `sampler: Callable[[int, int], jnp.ndarray]` — function for sampling raw vectors

➡️ Immutable dataclass container for algebra definition. No ops. Used by encoders and memory.

### 2. VSAMemory
- Stores named hypervectors (basis symbols)
- Uses `VSAModel` to sample and wrap vectors
- Supports dictionary-style access:
  - `memory.add("apple")`
  - `memory["apple"]`
- Methods:
  - `add(name: str)`
  - `add_many(names: list[str])`
  - `get(name: str)` → returns representation-wrapped vector

➡️ Symbol table + runtime memory for symbolic concepts.

### 3. AbstractHypervector
- Base class for representations
- Wraps a single `jnp.ndarray` with:
  - `.vec`: the underlying vector
  - `.normalize()`
  - `.to_numpy()`
  - `.shape`, `.dtype` proxies
- Future: implement `__jax_array__` and `__array__` for seamless ops

➡️ Allows clean vector math and JAX compatibility.

### 4. AbstractOpSet
Defines symbolic operations over `jnp.ndarray`s:
- `bind(a, b)`
- `bundle(*args)`
- `inverse(a)`
- `permute(a, shift)` (optional)

➡️ Stateless, pure functional interface for algebra.

### 5. Encoders
Classes that convert structured data into hypervectors:

#### ScalarEncoder
- Input: `name: str`, `value: float`
- Output: powered basis vector (e.g. `basis_vec ** value`)

#### DictEncoder
- Input: `{role: filler}`
- Output: bundled binding of role-filler pairs

➡️ Each encoder accepts a model and memory. Add `.fit()`, `.encode()` for consistency.

### 6. Similarity Metrics
Located in `vsax/similarity/`
- `cosine_similarity(a, b)`
- `dot_similarity(a, b)`
- `hamming_similarity(a, b)`

➡️ Independent of model. Uses `.vec` or coerces inputs.

### 7. I/O
#### `save_basis(memory, path)`
- JSON serialization of named basis vectors

#### `load_basis(memory, path)`
- Load into a memory from disk using the model's `rep_cls`

➡️ Reuse persistent symbolic spaces across sessions.

### 8. Resonator Networks
#### `CleanupMemory`
- Codebook projection for nearest vector retrieval
- Input: query vector
- Output: closest symbol from codebook or None if below threshold

#### `Resonator`
- Iterative factorization of VSA composites
- Decomposes `s = a ⊙ b ⊙ c` into factors from known codebooks
- Superposition initialization (Frady et al. 2020)
- Supports 2-3 factor composites
- Works with all 3 VSA models

➡️ Enables decoding of complex VSA data structures.

### 9. Vector Utilities
- `vsax.utils.coerce_vec()` — ensure input is `jnp.ndarray`
- `vsax.utils.vmap_bind()` — batch version of bind
- `vsax.utils.vmap_bundle()` — batch version of bundle
- `vsax.utils.pretty_repr()` — printing shape/type of vectors

➡️ Improves usability and debugging.

---

## Representations
Located in `vsax/representations/`
- `ComplexHypervector` — phase-based encoding, useful for FHRR
- `BinaryHypervector` — elementwise ±1 or 0/1 vectors
- `RealHypervector` — continuous valued vectors

Each wraps a `jnp.ndarray` and conforms to `AbstractHypervector`.

---

## Operation Sets
Located in `vsax/ops/`
- `FHRROperations` — FFT-based circular convolution
- `MAPOperations` — elementwise multiplication and mean
- `BinaryOperations` — XOR, majority

➡️ Functional, stateless ops working directly on `jnp.ndarray`s.

---

## Sampling
Located in `vsax/sampling/`
- `sample_random(dim, n)` — random normal
- `sample_circular(dim, n)` — structured circular sampling

➡️ Used by `VSAModel` for basis vector generation.

---

## Test Coverage
Located in `tests/`
- `test_model_memory_init()`
- `test_scalar_encoding()`
- `test_dict_encoding()`
- `test_similarity_metrics()`
- `test_save_load()`
- `test_vector_ops()`
- `test_batch_encoding()` (planned)

➡️ Validates representation correctness and symbolic consistency.

---

## Usage Examples

### Example 1: Basic symbolic binding
```python
a = memory["apple"]
b = memory["fruit"]
encoded = model.opset.bind(a.vec, b.vec)
```

### Example 2: Scalar Encoding
```python
encoder = ScalarEncoder(model, memory)
memory.add("temperature")
vec = encoder.encode("temperature", 23.5)
```

### Example 3: Dictionary Encoding (role-filler)
```python
encoder = DictEncoder(model, memory)
memory.add_many(["subject", "predicate", "object", "dog", "is_a", "animal"])
vec = encoder.encode({"subject": "dog", "predicate": "is_a", "object": "animal"})
```

### Example 4: Similarity
```python
similarity = cosine_similarity(vec, memory["dog"])
```

### Example 5: Save and Load Basis
```python
save_basis(memory, "./basis.json")
new_memory = VSAMemory(model)
load_basis(new_memory, "./basis.json")
```

### Example 6: Batch Operations
```python
from vsax.utils.batch import vmap_bind
X = jnp.stack([a.vec, b.vec, c.vec])
Y = jnp.stack([x.vec, y.vec, z.vec])
batch_result = vmap_bind(model.opset, X, Y)
```

### Example 7: Resonator Networks
```python
from vsax import CleanupMemory, Resonator

# Create codebooks
letters = CleanupMemory(["alpha", "beta"], memory)
numbers = CleanupMemory(["one", "two"], memory)

# Create resonator
resonator = Resonator([letters, numbers], model.opset)

# Factorize composite
composite = model.opset.bind(memory["alpha"].vec, memory["one"].vec)
factors = resonator.factorize(composite)  # ["alpha", "one"]
```

### Example 8: Access .vec automatically
```python
# Future sugar
bind(a, b)  # Automatically unwraps .vec if needed
```

---

## Extensibility Plan

### Completed ✅
- ✅ `GraphEncoder`, `SequenceEncoder`, `SetEncoder`, `DictEncoder`, `ScalarEncoder`
- ✅ Dictionary-style access to `VSAMemory`
- ✅ `vmap`/`jit`-friendly versions of bind/bundle
- ✅ Save/load basis vectors (I/O)
- ✅ Resonator networks for factorization
- ✅ Similarity metrics (cosine, dot, Hamming)
- ✅ Batch operations (vmap_bind, vmap_bundle, vmap_similarity)

### Future Enhancements
- Add `TreeEncoder` for hierarchical structures
- Add `QuaternionHypervector`, `FourierHypervector`
- Add coercion logic to auto-handle `.vec`
- Implement `__jax_array__` on representations for seamless ops
- Streamlit UI for interactive symbolic exploration
- CLI tools for inspecting memory
- Registries for custom representations and opsets
- Multi-factor resonator support (4+ factors)

---

## Summary
VSAX (v0.7.1) provides a principled, modular, and efficient system for symbolic reasoning with hypervectors. It is built for researchers and developers interested in neurosymbolic AI, cognitive modeling, and high-performance semantic encoding systems.

**Key Features:**
- Three complete VSA models (FHRR, MAP, Binary)
- Five core encoders for structured data
- Resonator networks for factorization and decoding
- Similarity metrics and batch operations
- I/O persistence for basis vectors
- GPU acceleration via JAX

Usability is prioritized with a clean, NumPy-style API, factory functions, batch operations, and JAX-native performance — enabling symbolic algebra at scale.

