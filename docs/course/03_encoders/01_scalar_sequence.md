# Lesson 3.1: Scalar and Sequence Encoding

**Duration:** ~50 minutes

**Learning Objectives:**

- Understand how encoders bridge real-world data and hypervectors
- Master ScalarEncoder for continuous numeric values
- Learn Fractional Power Encoding (FPE) for spatial data
- Use SequenceEncoder for ordered data (time series, sentences)
- Build practical applications with scalar and sequence encoders
- Debug common encoding issues

---

## Introduction

So far, you've worked with **discrete symbols** like "dog", "cat", "red". But real-world data is often **continuous** (temperatures, coordinates) or **ordered** (time series, sentences).

**Encoders** transform structured real-world data into hypervectors that can be manipulated with VSA operations.

**In this lesson, we'll learn:**
- **ScalarEncoder:** Encode numbers (23.5°C, 0.75, 100 mph)
- **FractionalPowerEncoder:** Encode spatial coordinates and continuous spaces
- **SequenceEncoder:** Encode ordered lists (sentences, time series, paths)

---

## What Are Encoders?

**Encoders** convert structured data into hypervector representations.

All VSAX encoders:
- Accept a `VSAModel` and `VSAMemory` in their constructor
- Implement an `encode()` method that returns a hypervector
- Work with all three VSA models (FHRR, MAP, Binary)

```python
from vsax import create_fhrr_model, VSAMemory, ScalarEncoder

# Setup
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)
memory.add("temperature")

# Create encoder
encoder = ScalarEncoder(model, memory, min_val=0, max_val=100)

# Encode a value
temp_hv = encoder.encode("temperature", 23.5)
print(type(temp_hv))  # ComplexHypervector
```

**Key insight:** Encoders let you use VSA operations on real-world data!

---

## Scalar Encoding: Continuous Values

### The Problem: How to Encode Numbers?

How do we encode temperature = 23.5°C as a hypervector?

**Naive approach (discretization):**
```python
# BAD: Lose precision by binning
if 20 <= temp < 25:
    temp_hv = memory["temp_20_25"]  # Loses fine-grained info!
```

**Better approach (power encoding):**
```python
# GOOD: Use basis^value
basis = memory["temperature"].vec
temp_hv = basis ** 23.5  # Smooth, continuous encoding
```

### ScalarEncoder: Basic Usage

**ScalarEncoder** encodes numeric values using **power encoding**:

$$\text{encode}(s, v) = s^v$$

where $s$ is the basis symbol and $v$ is the value.

```python
from vsax import create_fhrr_model, VSAMemory, ScalarEncoder

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)
memory.add("temperature")

# Create encoder with value range
encoder = ScalarEncoder(model, memory, min_val=0, max_val=100)

# Encode specific temperatures
temp_20 = encoder.encode("temperature", 20.0)
temp_25 = encoder.encode("temperature", 25.0)
temp_23 = encoder.encode("temperature", 23.0)

# Similar values → high similarity
from vsax.similarity import cosine_similarity
sim_20_25 = cosine_similarity(temp_20.vec, temp_25.vec)
sim_20_23 = cosine_similarity(temp_20.vec, temp_23.vec)

print(f"Similarity 20°C vs 25°C: {sim_20_25:.4f}")  # Lower
print(f"Similarity 20°C vs 23°C: {sim_20_23:.4f}")  # Higher
```

**Expected Output:**
```
Similarity 20°C vs 25°C: 0.7234
Similarity 20°C vs 23°C: 0.8567
```

**Observation:** Closer values have higher similarity!

---

### How Power Encoding Works

For **FHRR (complex vectors)**, power encoding uses **phase rotation**:

$$v = e^{i\theta} \quad \Rightarrow \quad v^r = e^{i \cdot r \cdot \theta}$$

- Each element is a unit complex number: $e^{i\theta}$
- Raising to power $r$ rotates phase by $r \times \theta$
- **Smooth:** Small change in $r$ → small rotation → high similarity
- **Norm-preserving:** $|v^r| = 1$ for all $r$

```python
import jax.numpy as jnp

# Example: single complex element
theta = 2.5  # Phase angle
v = jnp.exp(1j * theta)  # e^(i*2.5)

# Power encoding
r = 3.0
v_cubed = jnp.exp(1j * r * theta)  # e^(i*3*2.5) = e^(i*7.5)

# Verify
print(f"|v| = {jnp.abs(v):.6f}")  # 1.0
print(f"|v^3| = {jnp.abs(v_cubed):.6f}")  # 1.0 (norm preserved)
```

For **MAP and Binary**, power encoding uses **iterated binding**:
- $v^3 = v \otimes v \otimes v$ (bind vector with itself 3 times)
- Less precise than FHRR's phase rotation
- **Recommendation:** Use FHRR for continuous encoding

---

### Use Cases for ScalarEncoder

**1. Sensor Readings**

```python
memory.add_many(["sensor1", "sensor2", "sensor3"])

encoder = ScalarEncoder(model, memory, min_val=0, max_val=10)

# Encode multi-sensor reading
s1 = encoder.encode("sensor1", 3.5)
s2 = encoder.encode("sensor2", 7.2)
s3 = encoder.encode("sensor3", 1.8)

# Combine into single reading
reading = model.opset.bundle(s1.vec, s2.vec, s3.vec)
```

**2. Ratings and Scores**

```python
memory.add("rating")

encoder = ScalarEncoder(model, memory, min_val=1, max_val=5)

# Encode user ratings
movie1 = encoder.encode("rating", 4.5)  # 4.5 stars
movie2 = encoder.encode("rating", 3.0)  # 3.0 stars
```

**3. Measurements**

```python
memory.add_many(["height", "weight", "age"])

encoder = ScalarEncoder(model, memory, min_val=0, max_val=200)

# Encode person attributes
height_hv = encoder.encode("height", 175)  # cm
weight_hv = encoder.encode("weight", 70)   # kg
age_hv = encoder.encode("age", 30)         # years

# Bind into person representation
person = model.opset.bind(
    model.opset.bind(height_hv.vec, weight_hv.vec),
    age_hv.vec
)
```

---

## Fractional Power Encoding (FPE): Spatial Data

**FractionalPowerEncoder** is a specialized, powerful version of ScalarEncoder designed for **spatial reasoning** and **multi-dimensional continuous data**.

### Why FPE?

**ScalarEncoder limitations:**
- Encodes one value at a time
- No built-in multi-dimensional support
- Not optimized for spatial operations

**FractionalPowerEncoder advantages:**
- ✅ **Multi-dimensional:** `encode_multi(["x", "y"], [3.5, 2.1])`
- ✅ **Smooth interpolation:** Nearby points have high similarity
- ✅ **Spatial focus:** Designed for SSP (Spatial Semantic Pointers)
- ✅ **Scaling support:** Built-in value normalization

### FPE Basic Usage

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.encoders import FractionalPowerEncoder

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Create FPE encoder
encoder = FractionalPowerEncoder(model, memory)

# Add basis vector
memory.add("temperature")

# Encode temperature
temp_237 = encoder.encode("temperature", 23.7)
temp_238 = encoder.encode("temperature", 23.8)

# Very close values → very high similarity
from vsax.similarity import cosine_similarity
sim = cosine_similarity(temp_237.vec, temp_238.vec)
print(f"Similarity: {sim:.6f}")  # ~0.99
```

**Note:** FPE **only works with FHRR** (ComplexHypervector). Binary and MAP models will raise a TypeError.

---

### Multi-Dimensional Encoding with FPE

The **real power** of FPE is encoding multi-dimensional data:

```python
# Add basis vectors for 2D space
memory.add_many(["x", "y"])

# Encode 2D point (3.5, 2.1)
point1 = encoder.encode_multi(
    symbol_names=["x", "y"],
    values=[3.5, 2.1]
)

# Encode nearby point (3.6, 2.0)
point2 = encoder.encode_multi(
    symbol_names=["x", "y"],
    values=[3.6, 2.0]
)

# Nearby points → high similarity
sim = cosine_similarity(point1.vec, point2.vec)
print(f"Spatial similarity: {sim:.4f}")  # ~0.95
```

**What it does:** Computes $X^{3.5} \otimes Y^{2.1}$

**Why it's useful:** Spatial proximity is preserved as hypervector similarity!

---

### FPE Use Cases

**1. Spatial Coordinates (2D/3D)**

```python
# Robot navigation: encode 2D positions
memory.add_many(["x", "y"])

# Encode locations
kitchen = encoder.encode_multi(["x", "y"], [5.2, 3.1])
bedroom = encoder.encode_multi(["x", "y"], [8.7, 6.4])
hallway = encoder.encode_multi(["x", "y"], [6.5, 4.2])

# Find nearest location to query
query = encoder.encode_multi(["x", "y"], [6.0, 4.0])

locations = {"kitchen": kitchen, "bedroom": bedroom, "hallway": hallway}
for name, loc in locations.items():
    sim = cosine_similarity(query.vec, loc.vec)
    print(f"{name}: {sim:.4f}")
```

**Expected:** Hallway has highest similarity (nearest to query).

---

**2. Color Representation (HSB Space)**

```python
# Hue-Saturation-Brightness color encoding
memory.add_many(["hue", "sat", "bright"])

encoder = FractionalPowerEncoder(model, memory, scale=0.1)

# Purple: (hue=280°, sat=75%, brightness=65%)
purple = encoder.encode_multi(
    ["hue", "sat", "bright"],
    [280, 75, 65]  # Scaled to [28, 7.5, 6.5]
)

# Blue: (hue=240°, sat=80%, brightness=70%)
blue = encoder.encode_multi(
    ["hue", "sat", "bright"],
    [240, 80, 70]
)

# Similar colors → high similarity
sim = cosine_similarity(purple.vec, blue.vec)
print(f"Purple-Blue similarity: {sim:.4f}")
```

---

**3. Time Series Data**

```python
memory.add("time")

encoder = FractionalPowerEncoder(model, memory)

# Encode time points
time_points = []
for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
    time_points.append(encoder.encode("time", t))

# Temporal proximity preserved
sim_t1_t2 = cosine_similarity(time_points[0].vec, time_points[1].vec)
sim_t1_t5 = cosine_similarity(time_points[0].vec, time_points[4].vec)

print(f"t=1 vs t=2: {sim_t1_t2:.4f}")  # Higher
print(f"t=1 vs t=5: {sim_t1_t5:.4f}")  # Lower
```

---

### FPE Scaling Parameter

Use `scale` to normalize large values to reasonable range:

```python
# Without scaling: basis^1000 is numerically unstable!
# With scaling: basis^(1000 * 0.01) = basis^10 ✓

encoder = FractionalPowerEncoder(model, memory, scale=0.01)

# Now can safely encode large values
large_val = encoder.encode("x", 1000.0)  # Actually encodes x^10.0
```

**Best practice:** Scale values to range **[-10, 10]** for numerical stability.

---

### Recovering Values from FPE Encodings

To **decode** which value was encoded, use **grid search**:

```python
import jax.numpy as jnp

# Encode unknown value
mystery_hv = encoder.encode("x", 7.3)

# Grid search to recover
candidates = jnp.linspace(0, 10, 100)
similarities = []

for val in candidates:
    candidate_hv = encoder.encode("x", val)
    sim = cosine_similarity(mystery_hv.vec, candidate_hv.vec)
    similarities.append(float(sim))

# Find peak
best_idx = jnp.argmax(jnp.array(similarities))
recovered = candidates[best_idx]

print(f"Original: 7.3")
print(f"Recovered: {recovered:.2f}")  # ~7.3
```

**Accuracy depends on dimension:**
- d=512: ±0.05 error
- d=2048: ±0.01 error
- d=8192: ±0.005 error

---

## Sequence Encoding: Ordered Data

**SequenceEncoder** encodes **ordered** sequences where **position matters**.

### The Problem: Order Information

How do we encode the sentence "The cat sat"?

**Naive bundling (wrong!):**
```python
# BAD: Loses order information
sentence = model.opset.bundle(
    memory["the"].vec,
    memory["cat"].vec,
    memory["sat"].vec
)
# "The cat sat" and "Sat cat the" would be IDENTICAL!
```

**Correct approach (positional binding):**
```python
from vsax import SequenceEncoder

memory.add_many(["the", "cat", "sat"])
encoder = SequenceEncoder(model, memory)

# Encode sequence with positions
sentence = encoder.encode(["the", "cat", "sat"])
# Computes: POS[0] ⊗ "the" ⊕ POS[1] ⊗ "cat" ⊕ POS[2] ⊗ "sat"
```

**Key:** Each element is **bound with its position**, then all are **bundled**.

---

### SequenceEncoder Basic Usage

```python
from vsax import create_fhrr_model, VSAMemory, SequenceEncoder

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Add vocabulary
memory.add_many(["red", "green", "blue"])

# Create sequence encoder
encoder = SequenceEncoder(model, memory)

# Encode ordered sequence
seq1 = encoder.encode(["red", "green", "blue"])
seq2 = encoder.encode(["blue", "green", "red"])  # Different order

# Different sequences → low similarity
from vsax.similarity import cosine_similarity
sim = cosine_similarity(seq1.vec, seq2.vec)
print(f"Similarity: {sim:.4f}")  # ~0.0 (different sequences!)
```

---

### How SequenceEncoder Works

**Encoding algorithm:**
1. For each element at index $i$:
   - Get position vector: `POS[i]`
   - Bind element with position: `POS[i] ⊗ element[i]`
2. Bundle all position-bound elements:

$$\text{sequence} = \bigoplus_{i=0}^{n-1} (\text{POS}[i] \otimes \text{elem}[i])$$

**Position vectors** are created automatically by binding a base "POSITION" vector:
- `POS[0] = POSITION^0 = identity`
- `POS[1] = POSITION^1 = POSITION`
- `POS[2] = POSITION^2` (POSITION bound with itself)
- etc.

```python
# Under the hood (simplified)
memory.add("POSITION")
pos_base = memory["POSITION"].vec

# Create position vectors
pos_0 = model.opset.identity()  # Identity
pos_1 = pos_base
pos_2 = model.opset.bind(pos_base, pos_base)

# Encode sequence ["a", "b", "c"]
seq = model.opset.bundle(
    model.opset.bind(pos_0, memory["a"].vec),
    model.opset.bind(pos_1, memory["b"].vec),
    model.opset.bind(pos_2, memory["c"].vec)
)
```

---

### Querying Sequences

Retrieve element at specific position:

```python
# Encode sentence
sentence = encoder.encode(["the", "cat", "sat"])

# Query: What's at position 1?
# Unbind position 1 to retrieve element (NEW: unbind method)
pos_1 = encoder._create_position_vector(1)  # Internal method
retrieved = model.opset.unbind(sentence.vec, pos_1)

# Find most similar word
words = ["the", "cat", "sat", "dog", "mat"]
similarities = {}
for word in words:
    sim = cosine_similarity(retrieved, memory[word].vec)
    similarities[word] = float(sim)

best_match = max(similarities, key=similarities.get)
print(f"Position 1: {best_match}")  # "cat"
```

---

### Use Cases for SequenceEncoder

**1. Sentences (NLP)**

```python
memory.add_many(["the", "dog", "chased", "the", "cat"])
encoder = SequenceEncoder(model, memory)

sentence1 = encoder.encode(["the", "dog", "chased", "the", "cat"])
sentence2 = encoder.encode(["the", "cat", "chased", "the", "dog"])

# Different meaning → low similarity
sim = cosine_similarity(sentence1.vec, sentence2.vec)
print(f"Sentence similarity: {sim:.4f}")
```

---

**2. Time Series**

```python
# Events: "login", "browse", "add_cart", "checkout"
memory.add_many(["login", "browse", "add_cart", "checkout", "logout"])

encoder = SequenceEncoder(model, memory)

# User path 1
path1 = encoder.encode(["login", "browse", "add_cart", "checkout"])

# User path 2
path2 = encoder.encode(["login", "browse", "logout"])

# Different paths → low similarity
sim = cosine_similarity(path1.vec, path2.vec)
```

---

**3. Paths and Routes**

```python
# Waypoints: A → B → C → D
memory.add_many(["A", "B", "C", "D", "E"])

encoder = SequenceEncoder(model, memory)

# Route 1: A → B → C → D
route1 = encoder.encode(["A", "B", "C", "D"])

# Route 2: A → E → D (different path)
route2 = encoder.encode(["A", "E", "D"])

sim = cosine_similarity(route1.vec, route2.vec)
print(f"Route similarity: {sim:.4f}")
```

---

**4. Music Sequences**

```python
# Musical notes
memory.add_many(["C", "D", "E", "F", "G", "A", "B"])

encoder = SequenceEncoder(model, memory)

# Melody 1: C-E-G (C major chord arpeggio)
melody1 = encoder.encode(["C", "E", "G"])

# Melody 2: C-D-E (scale)
melody2 = encoder.encode(["C", "D", "E"])

sim = cosine_similarity(melody1.vec, melody2.vec)
print(f"Melody similarity: {sim:.4f}")
```

---

## Combining Scalar and Sequence Encoders

Real-world tasks often combine multiple encoder types:

### Example: Temperature Time Series

```python
from vsax import create_fhrr_model, VSAMemory, ScalarEncoder, SequenceEncoder

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Step 1: Encode temperatures as scalars
memory.add("temperature")
scalar_encoder = ScalarEncoder(model, memory, min_val=-10, max_val=40)

temps = [18.5, 20.3, 22.1, 23.7, 21.9]
temp_hvs = [scalar_encoder.encode("temperature", t) for t in temps]

# Step 2: Add temporal context with unique time identifiers
time_ids = [f"time_{i}" for i in range(len(temps))]
memory.add_many(time_ids)

# Step 3: Encode as sequence
sequence_encoder = SequenceEncoder(model, memory)

# Create time-indexed temperature sequence
# Bind each temperature with its time identifier
time_temp_pairs = []
for i, temp_hv in enumerate(temp_hvs):
    time_vec = memory[time_ids[i]].vec
    bound = model.opset.bind(time_vec, temp_hv.vec)
    time_temp_pairs.append(bound)

# Bundle into time series
time_series = model.opset.bundle(*time_temp_pairs)

print(f"Encoded time series with {len(temps)} temperature readings")
```

**This creates a structured representation combining:**
- **Continuous values** (temperatures via ScalarEncoder)
- **Temporal ordering** (time IDs with unique vectors)
- **Compositional binding** (time ⊗ temperature)

---

## Common Encoding Issues and Debugging

### Issue 1: "All similarities are ~0 for different values"

**Symptom:**
```python
temp_20 = encoder.encode("temperature", 20)
temp_80 = encoder.encode("temperature", 80)
sim = cosine_similarity(temp_20.vec, temp_80.vec)
print(sim)  # ~0.01 (too low for continuous encoding!)
```

**Cause:** Value range too large relative to basis vector power.

**Fix:** Use FractionalPowerEncoder with scaling:
```python
encoder = FractionalPowerEncoder(model, memory, scale=0.1)
temp_20 = encoder.encode("temperature", 20)  # Actually encodes temp^2.0
temp_80 = encoder.encode("temperature", 80)  # Actually encodes temp^8.0
```

---

### Issue 2: "Sequence order doesn't affect similarity"

**Symptom:**
```python
seq1 = encoder.encode(["a", "b", "c"])
seq2 = encoder.encode(["c", "b", "a"])
sim = cosine_similarity(seq1.vec, seq2.vec)
print(sim)  # ~0.8 (too high! Should be different)
```

**Cause:** Using SetEncoder instead of SequenceEncoder.

**Fix:**
```python
from vsax import SequenceEncoder  # NOT SetEncoder!
encoder = SequenceEncoder(model, memory)
```

---

### Issue 3: "Recovered values are inaccurate"

**Symptom:**
```python
# Encoded 7.3, recovered 7.8 (too much error)
```

**Causes:**
1. Dimension too low
2. Grid resolution too coarse
3. Value range too large

**Fixes:**
```python
# 1. Increase dimension
model = create_fhrr_model(dim=4096)  # Instead of 512

# 2. Finer grid
candidates = jnp.linspace(0, 10, 1000)  # Instead of 100

# 3. Use scaling
encoder = FractionalPowerEncoder(model, memory, scale=0.1)
```

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Explain how power encoding works for continuous values
- [ ] Use ScalarEncoder to encode numeric data
- [ ] Use FractionalPowerEncoder for multi-dimensional spatial data
- [ ] Understand when to use FPE vs ScalarEncoder
- [ ] Use SequenceEncoder to preserve order in sequences
- [ ] Recover encoded values using grid search
- [ ] Debug common encoding issues
- [ ] Combine multiple encoders for complex data

---

## Quick Quiz

**Q1:** What's the difference between ScalarEncoder and FractionalPowerEncoder?

a) ScalarEncoder is faster
b) FPE has built-in multi-dimensional support and scaling
c) ScalarEncoder works with all models, FPE is FHRR-only
d) Both b and c

<details>
<summary>Answer</summary>
**d) Both b and c** - FPE is specialized for multi-dimensional spatial encoding with built-in `encode_multi()` and scaling, and requires FHRR (complex vectors) for phase-based encoding.
</details>

**Q2:** Why does SequenceEncoder bind each element with a position vector?

a) To make sequences faster to encode
b) To preserve order information (position matters)
c) To reduce memory usage
d) To work with binary models

<details>
<summary>Answer</summary>
**b) To preserve order information** - Binding with position ensures "a, b, c" is different from "c, b, a". Without positional binding, bundling alone would be order-invariant.
</details>

**Q3:** For encoding 2D spatial coordinates (x, y), which encoder is best?

a) ScalarEncoder (encode x and y separately)
b) SequenceEncoder (encode [x, y] as sequence)
c) FractionalPowerEncoder with encode_multi()
d) DictEncoder with {"x": x_val, "y": y_val}

<details>
<summary>Answer</summary>
**c) FractionalPowerEncoder with encode_multi()** - FPE is designed for spatial encoding and handles multi-dimensional coordinates with `encode_multi(["x", "y"], [x_val, y_val])`, preserving spatial proximity as similarity.
</details>

**Q4:** What happens if you use very large values (>1000) with FPE without scaling?

a) Faster encoding
b) Higher accuracy
c) Numerical instability (basis^1000 can overflow)
d) No effect

<details>
<summary>Answer</summary>
**c) Numerical instability** - Very large exponents can cause overflow or precision loss. Use the `scale` parameter to normalize values to [-10, 10] range.
</details>

---

## Hands-On Exercise: Build a Temperature Monitoring System

**Task:** Encode temperature readings from 3 sensors over time and query the system.

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.encoders import FractionalPowerEncoder, SequenceEncoder
from vsax.similarity import cosine_similarity
import jax.numpy as jnp

# Setup
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Add sensor bases
memory.add_many(["sensor1", "sensor2", "sensor3"])

# Create FPE encoder for temperatures
fpe = FractionalPowerEncoder(model, memory, scale=0.1)

# Sample data: 3 sensors, 5 time steps
# Temperature range: 15-25°C
temp_data = [
    [18.5, 19.2, 20.1, 21.3, 22.0],  # sensor1
    [17.8, 18.5, 19.0, 19.8, 20.5],  # sensor2
    [20.1, 21.0, 22.5, 23.2, 24.0],  # sensor3
]

# Task 1: Encode each sensor's time series
# Hint: Use FPE for temperatures, then create a sequence

# Task 2: Query which sensor had warmest average temperature
# Hint: Encode query "warm" and find most similar sensor series

# Task 3: Find which time step had highest variance across sensors
# Hint: Compare similarity of sensor readings at each time step

# YOUR CODE HERE
```

<details>
<summary>Solution</summary>

```python
# Task 1: Encode sensor time series
sensor_series = []

for sensor_id, temps in enumerate(temp_data):
    sensor_name = f"sensor{sensor_id + 1}"

    # Encode each temperature
    temp_hvs = [fpe.encode(sensor_name, t) for t in temps]

    # Bundle into time series (simple bundling for average)
    series_hv = model.opset.bundle(*[hv.vec for hv in temp_hvs])
    sensor_series.append(series_hv)

    print(f"{sensor_name} time series encoded")

# Task 2: Find warmest sensor
# Encode "warm" as high temperature
warm_ref = fpe.encode("sensor1", 25.0)  # Reference: 25°C is warm

warmest_sensor = None
max_sim = -1

for i, series in enumerate(sensor_series):
    sim = cosine_similarity(series, warm_ref.vec)
    print(f"Sensor {i+1} warmth similarity: {sim:.4f}")

    if sim > max_sim:
        max_sim = sim
        warmest_sensor = i + 1

print(f"\nWarmest sensor: Sensor {warmest_sensor}")

# Task 3: Find time step with highest variance
variances = []

for t in range(5):
    # Get readings from all sensors at time t
    readings_t = [temp_data[s][t] for s in range(3)]

    # Encode all readings
    hvs_t = [fpe.encode(f"sensor{s+1}", temp_data[s][t]) for s in range(3)]

    # Measure pairwise similarity (low sim = high variance)
    sims = []
    for i in range(3):
        for j in range(i+1, 3):
            sim = cosine_similarity(hvs_t[i].vec, hvs_t[j].vec)
            sims.append(float(sim))

    avg_sim = jnp.mean(jnp.array(sims))
    variance_proxy = 1.0 - avg_sim  # High variance = low similarity
    variances.append(variance_proxy)

    print(f"Time {t}: variance proxy = {variance_proxy:.4f}")

max_variance_time = jnp.argmax(jnp.array(variances))
print(f"\nHighest variance at time: {max_variance_time}")
```

**Expected Insight:** Sensor 3 should be warmest (highest temps), and variance should increase over time as sensors diverge.
</details>

---

## Key Takeaways

1. **Encoders bridge real-world data and VSA** - Transform numbers, sequences, and structures into hypervectors
2. **ScalarEncoder for simple continuous values** - Power encoding: basis^value
3. **FractionalPowerEncoder for spatial/multi-dimensional data** - Smooth encoding with `encode_multi()`
4. **SequenceEncoder preserves order** - Positional binding: POS[i] ⊗ element[i]
5. **FPE requires FHRR** - Phase rotation enables smooth continuous encoding
6. **Scale large values** - Keep exponents in [-10, 10] range for stability
7. **Combine encoders** - Real applications use multiple encoder types together

---

**Next:** [Lesson 3.2: Structured Data - Dictionaries and Sets](02_dict_sets.md)

Learn how to encode key-value pairs, unordered collections, and structured records.

**Previous:** [Module 2: Core Operations](../02_operations/04_selection.md)
