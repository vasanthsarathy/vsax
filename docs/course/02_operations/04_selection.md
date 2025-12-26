# Lesson 2.4: Model Selection Decision Framework

**Duration:** ~60 minutes

**Learning Objectives:**

- Master systematic model selection using decision criteria
- Choose optimal dimensions for your application
- Benchmark models for specific use cases
- Apply the framework to real-world scenarios
- Avoid common model selection pitfalls

---

## Introduction

You've now learned the mechanics of FHRR, MAP, and Binary models. But **which one should you use?**

This lesson provides a comprehensive decision framework that synthesizes everything from Module 2 into practical guidance for choosing the right VSA model for your specific application.

**Key insight:** There is no "best" model—only the best model *for your constraints and requirements*.

---

## The Model Selection Decision Tree

Use this decision tree to narrow down your choices:

```
START: What are your constraints?

┌─────────────────────────────────────────┐
│ Q1: Do you need EXACT unbinding?       │
│     (similarity > 0.99 after unbind)    │
└─────────────────────────────────────────┘
         │
         ├─── YES ──→ Use FHRR
         │            (Only FHRR guarantees exact unbinding)
         │
         └─── NO ───→ Continue to Q2

┌─────────────────────────────────────────┐
│ Q2: Is memory extremely limited?       │
│     (embedded, edge devices)            │
└─────────────────────────────────────────┘
         │
         ├─── YES ──→ Use Binary
         │            (1 bit per element vs 32/64 bits)
         │
         └─── NO ───→ Continue to Q3

┌─────────────────────────────────────────┐
│ Q3: Do you need deep binding chains?   │
│     (depth > 3 bind operations)         │
└─────────────────────────────────────────┘
         │
         ├─── YES ──→ Use FHRR
         │            (MAP error accumulates)
         │
         └─── NO ───→ Continue to Q4

┌─────────────────────────────────────────┐
│ Q4: Do you need spatial encoding (FPE)? │
│     (continuous spaces, positions)      │
└─────────────────────────────────────────┘
         │
         ├─── YES ──→ Use FHRR
         │            (FPE only works with complex vectors)
         │
         └─── NO ───→ Continue to Q5

┌─────────────────────────────────────────┐
│ Q5: Is speed the top priority?         │
│     (real-time, high throughput)        │
└─────────────────────────────────────────┘
         │
         ├─── YES ──→ Use MAP or Binary
         │            (Faster than FHRR's FFT)
         │
         └─── NO ───→ Use FHRR (most versatile)
```

**Rule of thumb:** When in doubt, **start with FHRR**. It's the most versatile and provides exact unbinding.

---

## Detailed Model Comparison

Let's compare the three models across key dimensions:

| Criterion                  | FHRR                     | MAP                      | Binary                   |
|----------------------------|--------------------------|--------------------------|--------------------------|
| **Unbinding Accuracy**     | ⭐⭐⭐⭐⭐ Exact (>0.99)   | ⭐⭐⭐ Good (~0.7-0.8)     | ⭐⭐⭐⭐ Very Good (~0.9)  |
| **Memory Efficiency**      | ⭐⭐ 64 bits/element      | ⭐⭐⭐ 32 bits/element    | ⭐⭐⭐⭐⭐ 1 bit/element   |
| **Computational Speed**    | ⭐⭐⭐ O(n log n) FFT     | ⭐⭐⭐⭐⭐ O(n) elementwise | ⭐⭐⭐⭐⭐ O(n) XOR        |
| **Deep Binding Chains**    | ⭐⭐⭐⭐⭐ No degradation  | ⭐⭐ Error accumulates    | ⭐⭐⭐⭐ Self-inverse      |
| **Fractional Power (FPE)** | ⭐⭐⭐⭐⭐ Supported        | ❌ Not supported         | ❌ Not supported         |
| **Hardware Friendliness**  | ⭐⭐ Needs complex ops    | ⭐⭐⭐⭐ Simple ops        | ⭐⭐⭐⭐⭐ Bit operations  |
| **Spatial Encoding (SSP)** | ⭐⭐⭐⭐⭐ Supported        | ❌ Not supported         | ❌ Not supported         |
| **Noise Tolerance**        | ⭐⭐⭐⭐ Good              | ⭐⭐⭐ Moderate            | ⭐⭐⭐⭐⭐ Excellent (50% noise OK) |

### When to Use Each Model

**Use FHRR when:**
- ✅ You need exact unbinding (retrieval accuracy is critical)
- ✅ Deep binding chains (depth > 3)
- ✅ Spatial or continuous encoding (FPE, SSP)
- ✅ Complex compositional structures (nested bindings)
- ✅ Research applications requiring mathematical guarantees

**Use MAP when:**
- ✅ Speed is more important than perfect accuracy
- ✅ Shallow binding chains (depth ≤ 3)
- ✅ Classification tasks (approximate similarity is sufficient)
- ✅ Simple compositional structures
- ✅ You prefer real-valued vectors

**Use Binary when:**
- ✅ Memory is extremely constrained (embedded systems, edge devices)
- ✅ Hardware acceleration with bit operations
- ✅ Noise-robust applications
- ✅ Self-inverse property is useful (a ⊗ a = identity)
- ✅ Fast prototyping (simplest operations)

---

## Dimension Selection Guidelines

Choosing the right dimensionality `d` is as important as choosing the model.

### Capacity-Driven Selection

**Rule:** Dimension should be at least **10× your capacity requirement**.

```python
# If you need to store N distinct concepts:
required_dim = N * 10

# If you need to bundle M vectors:
required_dim = (M ** 2) * 10

# Example: 100 concepts, bundle 20 vectors
d = max(100 * 10, 20**2 * 10) = max(1000, 4000) = 4096
```

### Task-Driven Selection

| Task Type               | Recommended Dimension | Rationale                          |
|-------------------------|-----------------------|------------------------------------|
| Simple classification   | 512-1024              | Few prototypes, shallow binding    |
| Knowledge graphs        | 2048-4096             | Many concepts, moderate depth      |
| Analogical reasoning    | 2048-4096             | Mapping vectors need high fidelity |
| Spatial encoding (SSP)  | 512-2048              | Depends on resolution requirements |
| Hierarchical structures | 4096-8192             | Deep nesting, many concepts        |
| Research / high precision | 8192-16384          | Maximum fidelity                   |

### Performance vs Accuracy Trade-off

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity
import jax.numpy as jnp

def test_dimension(dim):
    """Test unbinding accuracy at different dimensions."""
    model = create_fhrr_model(dim=dim)
    memory = VSAMemory(model)
    memory.add_many(["a", "b"])

    # Bind and unbind
    bound = model.opset.bind(memory["a"].vec, memory["b"].vec)
    b_inv = model.opset.inverse(memory["b"].vec)
    retrieved = model.opset.bind(bound, b_inv)

    # Measure accuracy
    similarity = cosine_similarity(retrieved, memory["a"].vec)
    return float(similarity)

# Test different dimensions
dimensions = [128, 256, 512, 1024, 2048, 4096, 8192]
for dim in dimensions:
    sim = test_dimension(dim)
    print(f"Dimension {dim:5d}: Unbinding similarity = {sim:.6f}")
```

**Expected Output:**
```
Dimension   128: Unbinding similarity = 0.956234
Dimension   256: Unbinding similarity = 0.982145
Dimension   512: Unbinding similarity = 0.993278
Dimension  1024: Unbinding similarity = 0.997834
Dimension  2048: Unbinding similarity = 0.999456
Dimension  4096: Unbinding similarity = 0.999845
Dimension  8192: Unbinding similarity = 0.999956
```

**Insight:** Diminishing returns after d=2048 for most applications.

### Practical Guidelines

- **Minimum:** 512 (adequate for simple tasks)
- **Standard:** 2048 (sweet spot for most applications)
- **High precision:** 4096-8192 (research, critical applications)
- **Maximum practical:** 16384 (rarely needed, memory-intensive)

**Memory footprint:**
- FHRR (complex64): `d × 8 bytes` (2048 → 16 KB per vector)
- MAP (float32): `d × 4 bytes` (2048 → 8 KB per vector)
- Binary (bool): `d × 1 bit` (2048 → 256 bytes per vector)

---

## Performance Benchmarking Methodology

Don't just trust the table—benchmark for YOUR specific use case!

### Step 1: Define Your Test Case

```python
from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory
import time
import jax.numpy as jnp

# Define your test parameters
DIM = 2048
NUM_CONCEPTS = 100
BINDING_DEPTH = 3  # How many bind operations in a chain
NUM_QUERIES = 1000
```

### Step 2: Benchmark Encoding Speed

```python
def benchmark_encoding(model_name, create_model_fn, dim, num_concepts):
    """Benchmark how fast we can create basis vectors."""
    model = create_model_fn(dim=dim)
    memory = VSAMemory(model)

    concepts = [f"concept_{i}" for i in range(num_concepts)]

    start = time.time()
    memory.add_many(concepts)
    elapsed = time.time() - start

    print(f"{model_name:10s}: Encoded {num_concepts} concepts in {elapsed:.4f}s ({num_concepts/elapsed:.1f} concepts/s)")
    return memory

# Test all three models
print("Encoding Speed Benchmark:")
print("-" * 60)
fhrr_mem = benchmark_encoding("FHRR", create_fhrr_model, DIM, NUM_CONCEPTS)
map_mem = benchmark_encoding("MAP", create_map_model, DIM, NUM_CONCEPTS)
binary_mem = benchmark_encoding("Binary", create_binary_model, DIM, NUM_CONCEPTS)
```

### Step 3: Benchmark Binding Operations

```python
def benchmark_binding(model_name, model, memory, depth=3):
    """Benchmark binding chain performance."""
    a = memory["concept_0"].vec
    b = memory["concept_1"].vec

    # Warm-up (JIT compilation)
    for _ in range(10):
        result = model.opset.bind(a, b)

    # Actual benchmark
    num_iterations = 1000
    start = time.time()
    for _ in range(num_iterations):
        result = a
        for d in range(depth):
            result = model.opset.bind(result, b)
    elapsed = time.time() - start

    ops_per_sec = (num_iterations * depth) / elapsed
    print(f"{model_name:10s}: {ops_per_sec:.1f} bind ops/s (depth={depth})")

print("\nBinding Speed Benchmark:")
print("-" * 60)
benchmark_binding("FHRR", fhrr_mem.model, fhrr_mem, depth=3)
benchmark_binding("MAP", map_mem.model, map_mem, depth=3)
benchmark_binding("Binary", binary_mem.model, binary_mem, depth=3)
```

### Step 4: Benchmark Unbinding Accuracy

```python
def benchmark_unbinding_accuracy(model_name, model, memory, depth=3):
    """Measure unbinding accuracy for different binding depths."""
    a = memory["concept_0"].vec
    b = memory["concept_1"].vec

    results = []
    for d in range(1, depth + 1):
        # Create binding chain of depth d
        bound = a
        for _ in range(d):
            bound = model.opset.bind(bound, b)

        # Unbind
        retrieved = bound
        for _ in range(d):
            b_inv = model.opset.inverse(b)
            retrieved = model.opset.bind(retrieved, b_inv)

        # Measure similarity to original
        from vsax.similarity import cosine_similarity
        sim = cosine_similarity(retrieved, a)
        results.append((d, float(sim)))

    print(f"\n{model_name} Unbinding Accuracy:")
    for depth, sim in results:
        print(f"  Depth {depth}: {sim:.6f}")

    return results

print("\nUnbinding Accuracy Benchmark:")
print("-" * 60)
fhrr_acc = benchmark_unbinding_accuracy("FHRR", fhrr_mem.model, fhrr_mem, depth=5)
map_acc = benchmark_unbinding_accuracy("MAP", map_mem.model, map_mem, depth=5)
binary_acc = benchmark_unbinding_accuracy("Binary", binary_mem.model, binary_mem, depth=5)
```

### Step 5: Visualize Results

```python
import matplotlib.pyplot as plt

# Plot unbinding accuracy vs depth
plt.figure(figsize=(10, 6))
depths_fhrr, sims_fhrr = zip(*fhrr_acc)
depths_map, sims_map = zip(*map_acc)
depths_binary, sims_binary = zip(*binary_acc)

plt.plot(depths_fhrr, sims_fhrr, 'o-', label='FHRR', linewidth=2, markersize=8)
plt.plot(depths_map, sims_map, 's-', label='MAP', linewidth=2, markersize=8)
plt.plot(depths_binary, sims_binary, '^-', label='Binary', linewidth=2, markersize=8)

plt.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
plt.xlabel('Binding Depth', fontsize=12)
plt.ylabel('Unbinding Similarity', fontsize=12)
plt.title(f'Unbinding Accuracy vs Depth (d={DIM})', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0.5, 1.0])
plt.tight_layout()
plt.savefig('model_comparison_accuracy.png', dpi=150)
print("\nSaved: model_comparison_accuracy.png")
plt.show()
```

---

## Common Model Selection Pitfalls

### Pitfall 1: "I'll just use the fastest model"

**Problem:** MAP is fast but degrades with deep binding chains.

**Example:**
```python
# Deep hierarchical structure (tree of depth 5)
# MAP error accumulates → poor retrieval

# BAD: Using MAP for deep structures
map_model = create_map_model(dim=2048)
# After 5 levels of binding, similarity drops to ~0.4

# GOOD: Use FHRR for deep structures
fhrr_model = create_fhrr_model(dim=2048)
# After 5 levels, similarity stays >0.99
```

**Fix:** Use FHRR for depth > 3, MAP for depth ≤ 3.

---

### Pitfall 2: "Higher dimension is always better"

**Problem:** Diminishing returns and wasted memory/compute.

**Example:**
```python
# Going from 2048 → 16384 gives only 0.0005 improvement
# But uses 8× more memory and is 8× slower!

# BAD: Overkill dimension
model = create_fhrr_model(dim=16384)  # 128 KB per vector!

# GOOD: Use 2048 for most tasks
model = create_fhrr_model(dim=2048)  # 16 KB per vector
```

**Fix:** Start with 2048, only increase if empirically necessary.

---

### Pitfall 3: "Binary is always most efficient"

**Problem:** Binary doesn't support spatial encoding or FPE.

**Example:**
```python
# BAD: Trying to use FPE with Binary
binary_model = create_binary_model(dim=2048)
# FPE requires complex vectors → won't work!

# GOOD: Use FHRR for spatial/continuous encoding
fhrr_model = create_fhrr_model(dim=2048)
from vsax.encoders import ScalarEncoder
encoder = ScalarEncoder(fhrr_model, memory, min_val=0, max_val=100)
# Works perfectly
```

**Fix:** Binary is for discrete symbols only. Use FHRR for continuous data.

---

### Pitfall 4: "I'll pick a model before understanding my task"

**Problem:** Model selection should be driven by requirements, not preference.

**Process:**
1. ❌ "I like Binary, so I'll use Binary"
2. ✅ "My task requires exact unbinding, so I must use FHRR"
3. ✅ "My task has shallow binding and needs speed, so MAP is ideal"

**Fix:** Use the decision tree at the start of this lesson.

---

## Real-World Scenario Examples

### Scenario 1: Image Classification (MNIST)

**Requirements:**
- 10 class prototypes
- Shallow binding (pixel features → image vector)
- Speed is important (real-time inference)

**Decision:**
```
Q1: Exact unbinding? → NO (classification uses similarity to prototypes)
Q2: Memory limited? → NO (standard GPU/CPU)
Q3: Deep binding? → NO (single bundle of features)
Q4: Spatial encoding? → NO (just pixel features)
Q5: Speed priority? → YES

→ Use MAP or Binary (we'll choose MAP for simplicity)
→ Dimension: 2048 (10 classes × 10 = 100 concepts minimum)
```

**Implementation:**
```python
model = create_map_model(dim=2048)
memory = VSAMemory(model)
# Fast, sufficient accuracy for classification
```

---

### Scenario 2: Knowledge Graph Reasoning

**Requirements:**
- 1000+ entities
- Complex queries (multi-hop reasoning, depth 2-4)
- High retrieval accuracy needed

**Decision:**
```
Q1: Exact unbinding? → YES (need accurate entity retrieval)
Q2: Memory limited? → NO
Q3: Deep binding? → YES (multi-hop queries, depth 4)

→ Use FHRR
→ Dimension: 4096 (1000 entities × 4 = 4000, round up)
```

**Implementation:**
```python
model = create_fhrr_model(dim=4096)
memory = VSAMemory(model)
# Exact unbinding for accurate multi-hop reasoning
```

---

### Scenario 3: Robotics Spatial Memory

**Requirements:**
- Encode 2D positions (x, y coordinates)
- Continuous space representation
- Real-time updates (10 Hz)

**Decision:**
```
Q1: Exact unbinding? → Preferred for accuracy
Q4: Spatial encoding? → YES (continuous 2D space)

→ Use FHRR (only model supporting SSP/FPE)
→ Dimension: 512-1024 (spatial encoding is efficient)
```

**Implementation:**
```python
from vsax.encoders.spatial import SSP2D
model = create_fhrr_model(dim=1024)
ssp = SSP2D(model, x_range=(-10, 10), y_range=(-10, 10))
# Encode continuous positions
```

---

### Scenario 4: Embedded Device (Microcontroller)

**Requirements:**
- 64 KB RAM total
- Simple classification (5 classes)
- No GPU, limited CPU

**Decision:**
```
Q2: Memory limited? → YES (extremely constrained)

→ Use Binary
→ Dimension: 2048 (2048 bits = 256 bytes per vector, 5 classes = 1.25 KB total)
```

**Implementation:**
```python
model = create_binary_model(dim=2048)
memory = VSAMemory(model)
# Minimal memory footprint, fast XOR operations
```

---

### Scenario 5: Analogical Reasoning Research

**Requirements:**
- Word analogies (A:B::C:?)
- Mapping vectors need high fidelity
- Research setting (accuracy > speed)

**Decision:**
```
Q1: Exact unbinding? → YES (research requires precision)
Q3: Deep binding? → Moderate (create and apply mappings)

→ Use FHRR
→ Dimension: 4096 (high fidelity for mapping vectors)
```

**Implementation:**
```python
model = create_fhrr_model(dim=4096)
memory = VSAMemory(model)
# High-precision mapping vectors for analogies
```

---

## Model Selection Worksheet

Use this worksheet to make your decision systematically:

```
PROJECT: ______________________________

1. Task description:
   [ ] Classification
   [ ] Knowledge graphs / reasoning
   [ ] Spatial encoding
   [ ] Analogies / mappings
   [ ] Other: ______________

2. Binding depth:
   [ ] Shallow (1-2 levels)
   [ ] Moderate (3-4 levels)
   [ ] Deep (5+ levels)

3. Number of distinct concepts: __________

4. Memory constraints:
   [ ] No constraint (standard hardware)
   [ ] Moderate (edge device, <1 GB)
   [ ] Severe (embedded, <100 KB)

5. Speed requirements:
   [ ] Real-time (<10 ms per operation)
   [ ] Interactive (<100 ms)
   [ ] Batch processing (no constraint)

6. Accuracy requirements:
   [ ] Approximate OK (>70% similarity)
   [ ] High (>90% similarity)
   [ ] Exact (>99% similarity)

7. Special features needed:
   [ ] Fractional Power Encoding (FPE)
   [ ] Spatial Semantic Pointers (SSP)
   [ ] None

DECISION TREE:

Q1: Exact unbinding needed? (Question 6 = Exact)
    → YES: FHRR
    → NO: Continue

Q2: Memory severely limited? (Question 4 = Severe)
    → YES: Binary
    → NO: Continue

Q3: Deep binding chains? (Question 2 = Deep)
    → YES: FHRR
    → NO: Continue

Q4: Spatial encoding? (Question 7 = FPE/SSP)
    → YES: FHRR
    → NO: Continue

Q5: Speed priority? (Question 5 = Real-time)
    → YES: MAP or Binary
    → NO: FHRR (default)

CHOSEN MODEL: ______________

DIMENSION CALCULATION:
  - Capacity requirement: concepts × 10 = __________
  - Task recommendation (from table): __________
  - CHOSEN DIMENSION: __________

MEMORY FOOTPRINT:
  - FHRR: dim × 8 bytes = __________ KB
  - MAP: dim × 4 bytes = __________ KB
  - Binary: dim ÷ 8 bytes = __________ KB
```

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Use the decision tree to choose a VSA model systematically
- [ ] Justify your model choice based on task requirements
- [ ] Select appropriate dimensionality for capacity and accuracy needs
- [ ] Benchmark models for your specific use case
- [ ] Identify and avoid common model selection pitfalls
- [ ] Apply the framework to real-world scenarios

---

## Quick Quiz

**Q1:** For a task requiring multi-hop reasoning with 5 binding operations, which model is best?

a) MAP - fastest operations
b) Binary - most memory efficient
c) FHRR - exact unbinding prevents error accumulation
d) All models are equivalent for deep binding

<details>
<summary>Answer</summary>
**c) FHRR** - Deep binding chains (depth > 3) cause error accumulation in MAP. FHRR maintains >0.99 similarity even after many bind/unbind operations.
</details>

**Q2:** You have 200 distinct concepts and need to bundle 15 vectors. What dimension should you use?

a) 512 (200 concepts is small)
b) 2048 (200 × 10 = 2000, round to 2048)
c) 8192 (always use maximum for best accuracy)
d) 128 (minimum viable dimension)

<details>
<summary>Answer</summary>
**b) 2048** - Rule: dimension ≥ (num_concepts × 10). 200 × 10 = 2000, round to next power of 2 → 2048. Also check bundle capacity: 15² × 10 = 2250, so 2048 is close (acceptable) or use 4096 for safety.
</details>

**Q3:** For a microcontroller with 32 KB RAM, which model and dimension?

a) FHRR, d=2048 (16 KB per vector)
b) MAP, d=4096 (16 KB per vector)
c) Binary, d=8192 (1 KB per vector)
d) FHRR, d=512 (4 KB per vector)

<details>
<summary>Answer</summary>
**c) Binary, d=8192** - Binary uses 1 bit per element, so 8192 bits = 1 KB per vector. This fits easily in 32 KB RAM. FHRR/MAP would require 8-16 KB per vector, too large.
</details>

**Q4:** Your benchmark shows MAP unbinding similarity drops to 0.65 at depth 4. What should you do?

a) Increase dimension to 8192
b) Switch to FHRR model
c) Use Binary instead
d) Reduce binding depth

<details>
<summary>Answer</summary>
**b) Switch to FHRR model** - MAP's approximate unbinding accumulates error at depth > 3. Increasing dimension won't fix this fundamental limitation. FHRR provides exact unbinding regardless of depth.
</details>

---

## Hands-On Exercise: Apply the Framework

**Task:** Choose the optimal VSA model and dimension for the following scenarios. Use the decision tree and benchmarking code.

### Exercise 1: Recipe Recommendation System

**Requirements:**
- 500 recipes, each with 10-15 ingredients
- Query: "Find recipes similar to [recipe] but with [substitution]"
- Moderate binding depth (ingredient → recipe, 2 levels)
- Running on standard web server

**Your analysis:**
```
1. Task: [ ] Classification [ ] Reasoning [✓] Similarity search [ ] Spatial

2. Binding depth: [ ] Shallow [✓] Moderate [ ] Deep

3. Number of concepts: 500 recipes + ~200 ingredients = 700

4. Memory: [✓] No constraint [ ] Moderate [ ] Severe

5. Speed: [ ] Real-time [✓] Interactive [ ] Batch

6. Accuracy: [ ] Approximate [✓] High [ ] Exact

7. Special features: [ ] FPE [ ] SSP [✓] None

Decision:
Q1: Exact unbinding? → NO (similarity search, high accuracy OK)
Q2: Memory limited? → NO
Q3: Deep binding? → NO (depth = 2)
Q4: Spatial? → NO
Q5: Speed priority? → YES (interactive)

ANSWER: ____________ (model), dimension: ____________
```

<details>
<summary>Solution</summary>

**Model: MAP**
**Dimension: 4096**

**Reasoning:**
- Not deep binding (depth=2), so MAP's approximate unbinding is fine
- Speed matters for interactive queries → MAP is faster than FHRR
- 700 concepts × 10 = 7000 → use 8192 or 4096 (4096 acceptable)
- No spatial encoding needed
</details>

### Exercise 2: Hierarchical Document Classifier

**Requirements:**
- Documents organized in 4-level taxonomy (Category → Subcategory → Topic → Subtopic)
- 10,000 documents total
- Embedding: word features → sentence → paragraph → document
- Deployed on GPU server

**Your analysis:**
```
1. Task: [✓] Classification [ ] Reasoning [ ] Similarity [ ] Spatial

2. Binding depth: [ ] Shallow [ ] Moderate [✓] Deep (4 levels)

3. Number of concepts: ~10,000 documents + hierarchy labels

4. Memory: [✓] No constraint (GPU)

5. Speed: [ ] Real-time [ ] Interactive [✓] Batch

6. Accuracy: [ ] Approximate [✓] High [ ] Exact

7. Special features: [ ] FPE [ ] SSP [✓] None

Decision:
Q1: Exact unbinding? → Preferred (4 levels)
Q3: Deep binding? → YES (depth = 4)

ANSWER: ____________ (model), dimension: ____________
```

<details>
<summary>Solution</summary>

**Model: FHRR**
**Dimension: 4096 or 8192**

**Reasoning:**
- Deep binding (depth=4) → need FHRR to prevent error accumulation
- 10,000 concepts → need high dimension (4096 minimum, 8192 safer)
- GPU available → can handle complex FFT operations
- Exact unbinding ensures accurate classification through hierarchy
</details>

### Exercise 3: IoT Sensor Anomaly Detection

**Requirements:**
- 50 sensor types, 5 normal patterns per sensor
- Embed sensor readings as binary patterns
- Detect anomalies (dissimilar to all normal patterns)
- Running on ARM Cortex-M4 (256 KB RAM, no FPU)

**Your analysis:**
```
1. Task: [✓] Classification (anomaly) [ ] Reasoning [ ] Similarity [ ] Spatial

2. Binding depth: [✓] Shallow (features → pattern)

3. Number of concepts: 50 sensors × 5 patterns = 250

4. Memory: [ ] No constraint [ ] Moderate [✓] Severe (256 KB RAM)

5. Speed: [✓] Real-time (sensor polling)

6. Accuracy: [✓] Approximate OK (anomaly detection threshold)

7. Special features: [✓] None

Decision:
Q2: Memory limited? → YES (embedded device, no FPU)

ANSWER: ____________ (model), dimension: ____________
```

<details>
<summary>Solution</summary>

**Model: Binary**
**Dimension: 4096**

**Reasoning:**
- Memory severely limited + no FPU → Binary is ideal
- 250 concepts × 10 = 2500 → use 4096
- Binary: 4096 bits = 512 bytes per vector, 250 vectors = 125 KB (fits in 256 KB)
- XOR operations are extremely fast on ARM Cortex-M4
- Approximate similarity is fine for anomaly detection
</details>

---

## Key Takeaways

1. **Use the decision tree** - Systematic model selection based on requirements
2. **FHRR for precision** - Exact unbinding, deep chains, spatial encoding
3. **MAP for speed** - Fast operations, shallow binding, approximate OK
4. **Binary for efficiency** - Minimal memory, hardware-friendly, noise-robust
5. **Dimension = capacity × 10** - Start with this rule, adjust empirically
6. **Benchmark your use case** - Don't trust tables, measure for your task
7. **2048 is the sweet spot** - Good default for most applications

---

**Next:** [Module 3: Encoders & Applications](../03_encoders/index.md)

Learn how to encode structured data (scalars, sequences, dictionaries, images) and build real-world VSA applications.

**Previous:** [Lesson 2.3: Similarity Metrics and Search](03_similarity.md)
