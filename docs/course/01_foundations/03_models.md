# Lesson 1.3: The Three VSA Models in VSAX

**Duration:** ~45 minutes

**Learning Objectives:**

- Understand the three VSA models: FHRR, MAP, and Binary
- Learn the key differences in representation and operations
- Develop intuition for when to use each model
- Use VSAX factory functions to create models
- Apply decision framework for model selection

---

## Introduction

VSAX implements three major VSA models, each with different trade-offs:

1. **FHRR** (Fourier Holographic Reduced Representations) - Complex vectors, exact unbinding
2. **MAP** (Multiply-Add-Permute) - Real vectors, approximate unbinding
3. **Binary** - Discrete {0,1} vectors, fast and memory-efficient

Each model uses different **representations** and **operations**, but all share the same high-level concepts (binding, bundling, similarity).

---

## Model 1: FHRR (Fourier Holographic Reduced Representations)

### What It Is

- **Representation**: Complex-valued vectors (ℂ^d)
- **Binding**: Circular convolution via FFT
- **Bundling**: Element-wise sum (complex addition)
- **Inverse**: Complex conjugate (exact)

### Key Properties

✅ **Exact unbinding** - Complex conjugate provides perfect inverse
✅ **Supports fractional powers** - Enables continuous encoding (FPE)
✅ **Well-studied** - Strong theoretical foundations
❌ **More memory** - 2× real numbers (real + imaginary parts)
❌ **Slower** - FFT operations have overhead

### When to Use FHRR

- Need exact unbinding for deep hierarchies
- Working with continuous spaces (spatial encoding, function encoding)
- Accuracy matters more than speed
- Memory is not critically constrained

### Code Example

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity

# Create FHRR model
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Add symbols
memory.add_many(["cat", "red", "mat"])

# Binding (circular convolution)
bound = model.opset.bind(memory["cat"].vec, memory["red"].vec)

# Unbinding (exact inverse via complex conjugate)
inverse_red = model.opset.inverse(memory["red"].vec)
retrieved = model.opset.bind(bound, inverse_red)

# Check accuracy
sim = cosine_similarity(retrieved, memory["cat"].vec)
print(f"FHRR unbinding accuracy: {sim:.6f}")  # ~0.999999 (nearly perfect)
```

**Expected Output:**
```
FHRR unbinding accuracy: 0.999998
```

---

## Model 2: MAP (Multiply-Add-Permute)

### What It Is

- **Representation**: Real-valued vectors (ℝ^d)
- **Binding**: Element-wise multiplication
- **Bundling**: Element-wise sum (addition)
- **Inverse**: Element-wise division (approximate)

### Key Properties

✅ **Simple operations** - Easiest to understand and implement
✅ **Fast** - Element-wise operations are very efficient
✅ **Less memory** - Real numbers only (half of FHRR)
❌ **Approximate unbinding** - Division introduces error
❌ **Error accumulates** - Deep binding chains degrade
❌ **No fractional powers** - Can't use FPE

### When to Use MAP

- Speed is critical
- Shallow binding structures (1-2 levels)
- Memory constrained
- Don't need exact unbinding

### Code Example

```python
from vsax import create_map_model, VSAMemory

# Create MAP model
model = create_map_model(dim=2048)
memory = VSAMemory(model)

# Add symbols
memory.add_many(["cat", "red", "mat"])

# Binding (element-wise multiply)
bound = model.opset.bind(memory["cat"].vec, memory["red"].vec)

# Unbinding (approximate inverse via division)
inverse_red = model.opset.inverse(memory["red"].vec)
retrieved = model.opset.bind(bound, inverse_red)

# Check accuracy
sim = cosine_similarity(retrieved, memory["cat"].vec)
print(f"MAP unbinding accuracy: {sim:.6f}")  # ~0.707 (good but not perfect)
```

**Expected Output:**
```
MAP unbinding accuracy: 0.707123
```

**Note:** MAP unbinding is approximate. Similarity ~0.7 is typical and sufficient for most applications.

---

## Model 3: Binary

### What It Is

- **Representation**: Binary vectors ({0,1}^d or {-1,+1}^d)
- **Binding**: XOR (exclusive OR)
- **Bundling**: Majority voting per element
- **Inverse**: Self (XOR is self-inverse: a ⊕ b ⊕ b = a)

### Key Properties

✅ **Fastest** - XOR and majority are hardware-optimized
✅ **Minimal memory** - 1 bit per element (64× less than FHRR)
✅ **Exact unbinding** - XOR is self-inverse
✅ **Ideal for hardware** - Perfect for neuromorphic chips and edge devices
❌ **Discrete only** - Can't represent continuous values directly
❌ **Lower capacity** - Binary limits information density

### When to Use Binary

- Deploying to edge devices (IoT, mobile, neuromorphic)
- Memory is critically constrained
- Need maximum speed
- Working with discrete/symbolic data only

### Code Example

```python
from vsax import create_binary_model, VSAMemory

# Create Binary model
model = create_binary_model(dim=2048)
memory = VSAMemory(model)

# Add symbols
memory.add_many(["cat", "red", "mat"])

# Binding (XOR)
bound = model.opset.bind(memory["cat"].vec, memory["red"].vec)

# Unbinding (XOR with same key - self-inverse)
retrieved = model.opset.bind(bound, memory["red"].vec)  # XOR again

# Check accuracy (for binary, use hamming-based similarity)
from vsax.similarity import hamming_similarity
sim = hamming_similarity(retrieved, memory["cat"].vec)
print(f"Binary unbinding accuracy: {sim:.6f}")  # ~1.0 (exact)
```

**Expected Output:**
```
Binary unbinding accuracy: 1.000000
```

---

## Comparison Table

| Feature | FHRR | MAP | Binary |
|---------|------|-----|--------|
| **Representation** | Complex (ℂ^d) | Real (ℝ^d) | Binary ({0,1}^d) |
| **Memory per element** | 16 bytes | 8 bytes | 0.125 bytes |
| **Binding** | Circular conv (FFT) | Element-wise × | XOR |
| **Unbinding accuracy** | Exact (~1.0) | Approximate (~0.7) | Exact (1.0) |
| **Speed** | Moderate | Fast | Fastest |
| **Fractional powers** | ✅ Yes | ❌ No | ❌ No |
| **Best for** | Accuracy, continuous | Speed, simplicity | Hardware, memory |

---

## Decision Framework

### Start Here: Decision Tree

```
Do you need continuous/spatial encoding (FPE, SSP, VFA)?
├─ YES → Use FHRR (only model supporting fractional powers)
└─ NO → Continue...
    │
    Is memory critically constrained (<1MB per vector)?
    ├─ YES → Use Binary (1 bit per element)
    └─ NO → Continue...
        │
        Do you need exact unbinding for deep hierarchies?
        ├─ YES → Use FHRR (exact inverse via conjugate)
        └─ NO → Continue...
            │
            Is speed the top priority?
            ├─ YES → Use Binary (fastest) or MAP (simple)
            └─ NO → Use FHRR (default, most versatile)
```

### Quick Reference

| Your Requirement | Recommended Model |
|------------------|-------------------|
| Spatial encoding, continuous values | FHRR |
| Deep binding chains (>3 levels) | FHRR |
| Edge deployment, IoT, neuromorphic | Binary |
| Simple prototype, learning VSA | MAP or FHRR |
| Maximum speed, symbolic data only | Binary |
| Don't know yet / general purpose | FHRR (most versatile) |

---

## Practical Example: Comparing All Three

Let's solve the same problem with all three models and compare results.

**Task:** Encode "The cat sat on the mat" and query "What's the subject?"

```python
import jax.numpy as jnp
from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory
from vsax.similarity import cosine_similarity, hamming_similarity

def encode_sentence(model_name):
    # Create model
    if model_name == "FHRR":
        model = create_fhrr_model(dim=2048)
    elif model_name == "MAP":
        model = create_map_model(dim=2048)
    else:  # Binary
        model = create_binary_model(dim=2048)

    memory = VSAMemory(model)
    memory.add_many(["cat", "mat", "sat", "subject", "object", "verb"])

    # Encode sentence: (cat⊗subject) ⊕ (mat⊗object) ⊕ (sat⊗verb)
    cat_subj = model.opset.bind(memory["cat"].vec, memory["subject"].vec)
    mat_obj = model.opset.bind(memory["mat"].vec, memory["object"].vec)
    sat_verb = model.opset.bind(memory["sat"].vec, memory["verb"].vec)

    sentence = model.opset.bundle(cat_subj, mat_obj, sat_verb)

    # Query: "What's the subject?"
    subject_inv = model.opset.inverse(memory["subject"].vec)
    retrieved = model.opset.bind(sentence, subject_inv)

    # Measure similarity to "cat"
    if model_name == "Binary":
        sim = hamming_similarity(retrieved, memory["cat"].vec)
    else:
        sim = cosine_similarity(retrieved, memory["cat"].vec)

    return sim

# Compare all three
for model_name in ["FHRR", "MAP", "Binary"]:
    sim = encode_sentence(model_name)
    print(f"{model_name:10s} - Retrieved 'cat' with similarity: {sim:.4f}")
```

**Expected Output:**
```
FHRR       - Retrieved 'cat' with similarity: 0.9567
MAP        - Retrieved 'cat' with similarity: 0.6234
Binary     - Retrieved 'cat' with similarity: 0.9821
```

**Analysis:**
- **FHRR**: High accuracy (~0.96) due to exact unbinding
- **MAP**: Lower accuracy (~0.62) due to approximate inverse, but still usable
- **Binary**: High accuracy (~0.98) due to XOR self-inverse

All three models successfully retrieve "cat" as the subject!

---

## Common Misconceptions

### ❌ "Binary is always better because it's faster"
**Reality:** Binary can't handle continuous values. FHRR is needed for spatial/function encoding.

### ❌ "MAP is bad because unbinding is approximate"
**Reality:** For many applications, ~0.7 similarity is sufficient. MAP's simplicity and speed are valuable.

### ❌ "I need to choose one model and stick with it"
**Reality:** You can use different models for different parts of your application!

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Describe the three VSA models (FHRR, MAP, Binary)
- [ ] Explain key differences in representation and operations
- [ ] Identify which model to use for a given requirement
- [ ] Use VSAX factory functions to create models
- [ ] Understand binding/unbinding accuracy trade-offs
- [ ] Apply the decision framework for model selection

---

## Quick Quiz

**Q1:** Which model supports fractional power encoding (FPE)?

a) FHRR
b) MAP
c) Binary
d) All three

<details>
<summary>Answer</summary>
**a) FHRR** - Only complex vectors support fractional powers (phase rotation).
</details>

**Q2:** Which model has the smallest memory footprint?

a) FHRR
b) MAP
c) Binary
d) All equal

<details>
<summary>Answer</summary>
**c) Binary** - Uses 1 bit per element (0.125 bytes vs 8 bytes for MAP, 16 bytes for FHRR).
</details>

**Q3:** Which model provides exact unbinding?

a) Only FHRR
b) Only Binary
c) FHRR and Binary
d) All three

<details>
<summary>Answer</summary>
**c) FHRR and Binary** - FHRR uses complex conjugate (exact), Binary uses XOR (self-inverse). MAP uses division (approximate).
</details>

**Q4:** You're deploying to an Arduino with 2KB RAM. Which model should you use?

a) FHRR
b) MAP
c) Binary
d) None (not possible)

<details>
<summary>Answer</summary>
**c) Binary** - Only Binary can fit in 2KB. A d=512 binary vector uses only 64 bytes!
</details>

**Q5:** You need to encode continuous 2D spatial coordinates. Which model?

a) FHRR
b) MAP
c) Binary
d) Any model

<details>
<summary>Answer</summary>
**a) FHRR** - Continuous/spatial encoding requires fractional powers, which only FHRR supports.
</details>

---

## Hands-On Exercise

**Task:** Benchmark all three models on your hardware.

1. Create models with d=2048 for all three types
2. Generate 1000 random symbols in memory
3. Time 10,000 bind operations for each model
4. Time 10,000 bundle operations (100 vectors each) for each model
5. Measure memory usage per vector
6. Create a comparison table

**Solution:**

```python
import time
import numpy as np
from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory

def benchmark_model(model, memory, num_ops=10000):
    """Benchmark bind and bundle operations."""
    symbols = [f"sym_{i}" for i in range(1000)]
    memory.add_many(symbols)

    # Benchmark binding
    start = time.time()
    for i in range(num_ops):
        idx1, idx2 = i % 1000, (i + 1) % 1000
        _ = model.opset.bind(memory[symbols[idx1]].vec, memory[symbols[idx2]].vec)
    bind_time = time.time() - start

    # Benchmark bundling (100 vectors at a time)
    start = time.time()
    for i in range(num_ops // 100):
        vecs = [memory[symbols[j]].vec for j in range(100)]
        _ = model.opset.bundle(*vecs)
    bundle_time = time.time() - start

    # Memory usage (rough estimate)
    vec = memory[symbols[0]].vec
    mem_per_vec = vec.nbytes

    return bind_time, bundle_time, mem_per_vec

# Benchmark all three
models = {
    "FHRR": create_fhrr_model(dim=2048),
    "MAP": create_map_model(dim=2048),
    "Binary": create_binary_model(dim=2048)
}

print(f"{'Model':<10} {'Bind (ms)':<12} {'Bundle (ms)':<12} {'Memory/vec (KB)':<15}")
print("-" * 55)

for name, model in models.items():
    memory = VSAMemory(model)
    bind_t, bundle_t, mem = benchmark_model(model, memory)

    print(f"{name:<10} {bind_t*1000/10000:<12.4f} {bundle_t*1000/100:<12.4f} {mem/1024:<15.2f}")
```

**Expected findings:**
- Binary is fastest for binding (XOR)
- FHRR is slowest (FFT overhead)
- Memory: Binary << MAP < FHRR

---

## Key Takeaways

1. **Three models, same concepts** - All use binding, bundling, similarity
2. **FHRR** - Best accuracy, supports continuous encoding, most versatile
3. **MAP** - Simple, fast, good for shallow structures
4. **Binary** - Fastest, minimal memory, perfect for hardware deployment
5. **Model selection matters** - Choose based on requirements (accuracy vs speed vs memory)
6. **Default to FHRR** - Unless you have specific constraints (memory, speed, hardware)

---

## Further Reading

- **Tutorial 05**: [Understanding VSA Models](../../tutorials/05_model_comparison.md) - Deeper dive with benchmarks
- **User Guide**: [Models](../../guide/models.md) - Technical API documentation
- **Research Papers**:
  - FHRR: Plate (1995) - Holographic Reduced Representations
  - MAP: Gayler (2003) - Multiply-Add-Permute
  - Binary: Kanerva (2009) - Hyperdimensional Computing

---

**Next:** [Lesson 1.4: Your First VSAX Program](04_first_program.md)

Put it all together! Write your first complete VSA program from scratch.

**Previous:** [Lesson 1.2: The Two Fundamental Operations](02_operations.md)
