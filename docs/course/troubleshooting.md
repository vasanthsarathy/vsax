# Troubleshooting Common VSA Issues

This guide covers common problems you might encounter while learning VSA and using VSAX.

## Low Similarity Issues

### Problem: "My similarities are all around 0.5"

**Possible Causes:**
1. Vectors aren't normalized
2. Dimension is too low
3. Using wrong similarity metric

**Solutions:**

```python
# Solution 1: Normalize vectors
vec_normalized = vec.normalize()

# Solution 2: Increase dimension
model = create_fhrr_model(dim=2048)  # Instead of dim=512

# Solution 3: Use cosine similarity (default)
from vsax.similarity import cosine_similarity
sim = cosine_similarity(vec1.vec, vec2.vec)
```

**Why this happens:** In high dimensions, random vectors are nearly orthogonal. Similarity ~0 is expected for unrelated vectors. If you're seeing ~0.5, vectors might not be properly normalized or dimension is insufficient.

---

### Problem: "Unbinding doesn't recover the original vector"

**Possible Causes:**
1. Using MAP with too many binding operations (error accumulates)
2. Not using inverse correctly
3. Dimension too low for task complexity

**Solutions:**

```python
# Solution 1: Switch to FHRR for exact unbinding
model = create_fhrr_model(dim=2048)  # FHRR has exact inverse

# Solution 2: Use inverse operation correctly
retrieved = model.opset.unbind(bound, key.vec)

# Solution 3: Increase dimension
model = create_fhrr_model(dim=4096)  # More capacity
```

**Why this happens:** MAP uses approximate unbinding (division), which accumulates error. FHRR uses complex conjugate for exact inversion.

---

### Problem: "Similarity to bundled vector is too low"

**Possible Causes:**
1. Too many vectors bundled (capacity exceeded)
2. Vectors not normalized before bundling
3. Dimension insufficient for bundle size

**Solutions:**

```python
# Solution 1: Check capacity (rule of thumb: bundle < sqrt(dim))
max_bundle_size = int(math.sqrt(dim))  # For dim=2048, max ~45 vectors

# Solution 2: Normalize before bundling
normalized_vecs = [v.normalize() for v in vecs]
bundled = model.opset.bundle(*[v.vec for v in normalized_vecs])

# Solution 3: Increase dimension
model = create_fhrr_model(dim=8192)  # More capacity
```

**Capacity formula:** For FHRR and MAP, bundling capacity ≈ √d. For Binary, capacity is higher but still finite.

---

## Encoding Issues

### Problem: "ScalarEncoder not working for continuous values"

**Possible Causes:**
1. Using MAP or Binary (ScalarEncoder requires FHRR for fractional powers)
2. Values out of range
3. Scale not set appropriately

**Solutions:**

```python
# Solution 1: Use FHRR model
model = create_fhrr_model(dim=2048)  # ComplexHypervector required

# Solution 2: Normalize values to appropriate range
from vsax.encoders import ScalarEncoder
encoder = ScalarEncoder(model, memory, scale=100.0)  # Adjust scale

# Solution 3: Check value range
value = (value - min_val) / (max_val - min_val)  # Normalize to [0, 1]
encoded = encoder.encode("basis", value * scale)
```

**Why this happens:** Fractional powers require complex numbers. MAP/Binary don't support fractional exponents.

---

### Problem: "DictEncoder produces low similarity for similar dicts"

**Possible Causes:**
1. Key-value pairs order matters (shouldn't)
2. Not enough shared keys
3. Value encodings are too dissimilar

**Solutions:**

```python
# Check overlap
dict1_keys = set(dict1.keys())
dict2_keys = set(dict2.keys())
overlap = dict1_keys & dict2_keys
print(f"Shared keys: {len(overlap)} / {len(dict1_keys)}")

# For continuous values, use ScalarEncoder
encoder = DictEncoder(model, memory, value_encoder=ScalarEncoder)
```

**Why this happens:** DictEncoder bundles role-filler bindings. Similarity depends on shared keys and value similarity.

---

## Performance Issues

### Problem: "VSAX is running slowly on GPU"

**Possible Causes:**
1. Data not moved to GPU
2. Using CPU NumPy instead of JAX
3. Not using JIT compilation

**Solutions:**

```python
# Solution 1: Ensure JAX uses GPU
import jax
print(jax.devices())  # Should show GPU

# Solution 2: Use JAX arrays, not NumPy
import jax.numpy as jnp
vec = jnp.array(data)  # Not np.array()

# Solution 3: JIT compile operations
from jax import jit

@jit
def encode_batch(values):
    return model.opset.bundle(*[encoder.encode("basis", v).vec for v in values])
```

See [GPU Usage Guide](../guide/gpu_usage.md) for details.

---

### Problem: "Out of memory errors"

**Possible Causes:**
1. Dimension too high for available memory
2. Batch size too large
3. Memory leak in loop

**Solutions:**

```python
# Solution 1: Reduce dimension
model = create_fhrr_model(dim=2048)  # Instead of 8192

# Solution 2: Process in smaller batches
batch_size = 32  # Instead of 1000
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    process_batch(batch)

# Solution 3: Clear memory explicitly
import gc
gc.collect()
```

---

## Model Selection Issues

### Problem: "Not sure which model to use"

**Decision Framework:**

| Requirement | Model | Why |
|-------------|-------|-----|
| Exact unbinding required | FHRR | Complex conjugate gives exact inverse |
| Speed is critical | Binary | XOR is fastest operation |
| Memory constrained | Binary | 1 bit per element vs 64 bits (float) |
| Continuous encoding needed | FHRR | Supports fractional powers |
| Simplicity preferred | MAP | Easiest to understand (multiply/add) |
| Hardware deployment | Binary | Lowest memory, fastest on edge devices |

**Still unsure?** Start with FHRR. It's the most versatile.

---

## Installation Issues

### Problem: "ImportError: cannot import name 'create_fhrr_model'"

**Solution:**

```bash
# Ensure latest version
pip install --upgrade vsax

# Or install from source
git clone https://github.com/vasanthsarathy/vsax.git
cd vsax
pip install -e ".[dev]"
```

---

### Problem: "JAX CUDA errors"

**Solution:**

```bash
# Install JAX with CUDA support
pip install --upgrade "jax[cuda12]"

# Check JAX can see GPU
python -c "import jax; print(jax.devices())"
```

See [JAX installation docs](https://jax.readthedocs.io/en/latest/installation.html) for platform-specific instructions.

---

## Conceptual Confusion

### Problem: "When to use binding vs bundling?"

**Simple Rule:**

- **Binding**: Combine concepts into a NEW concept (dissimilar to inputs)
  - Example: "cup" ⊗ "red" = "red cup" (different from both "cup" and "red")

- **Bundling**: Create a PROTOTYPE/SET (similar to inputs)
  - Example: "cat" ⊕ "dog" ⊕ "mouse" = "small mammals" (similar to each)

**Still confused?** Think of binding as multiplication and bundling as addition.

---

### Problem: "What's the difference between operators and hypervectors?"

| Hypervector | Operator |
|-------------|----------|
| Represents a concept | Represents a transformation |
| Created randomly | Learned from examples |
| Used in: memory, encoding | Used in: relations, roles |
| Example: "cup", "plate" | Example: LEFT_OF, AGENT |

**Use hypervectors for:** Things, concepts, symbols
**Use operators for:** Relations between things, transformations

See [Clifford Operators Tutorial](../tutorials/10_clifford_operators.md) for details.

---

## Still Stuck?

### Resources

1. **[Getting Started Guide](../getting-started.md)** - Basic usage examples
2. **[User Guide](../guide/models.md)** - API reference
3. **[Tutorials](../tutorials/index.md)** - Application recipes
4. **[Design Spec](../design-spec.md)** - Architecture details

### Get Help

- **GitHub Discussions**: [Ask questions](https://github.com/vasanthsarathy/vsax/discussions)
- **GitHub Issues**: [Report bugs](https://github.com/vasanthsarathy/vsax/issues)
- **Email**: Contact the maintainers (see repo)

### Debug Checklist

When debugging any VSA issue:

- [ ] Check vector dimensions match
- [ ] Verify vectors are normalized
- [ ] Confirm using correct model (FHRR vs MAP vs Binary)
- [ ] Check similarity metric is appropriate
- [ ] Review capacity limits (bundle size < √dim)
- [ ] Ensure JAX is using intended device (CPU/GPU)
- [ ] Verify VSAX version is latest

---

**Found a solution not listed here?**

Please contribute! Submit a PR adding your solution to this troubleshooting guide.

[Back to Course Overview](index.md)
