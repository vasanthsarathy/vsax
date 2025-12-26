# Lesson 5.1: Vector Function Architecture

**Estimated time:** 50 minutes

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand why encoding functions as hypervectors is valuable
- Explain the mathematical foundation of VFA (RKHS and fractional power kernel)
- Encode functions from samples and evaluate at new points
- Perform function arithmetic (addition, shifting, convolution)
- Apply VFA to regression, signal processing, and control problems
- Understand when VFA is appropriate vs neural networks

## Prerequisites

- Module 3, Lesson 3.1 (Fractional Power Encoding)
- Basic understanding of function approximation
- Familiarity with FHRR model

---

## The Problem: Functions as First-Class Citizens

In traditional programming, **functions are opaque**:

```python
def temperature_profile(time):
    """Temperature over 24 hours."""
    return 20 + 5 * np.sin(2 * np.pi * time / 24)

def humidity_profile(time):
    """Humidity over 24 hours."""
    return 60 + 10 * np.cos(2 * np.pi * time / 24)
```

**What we CANNOT do:**
- ❌ Compare function similarity
- ❌ Store function in VSA memory alongside symbols
- ❌ Bind function to concept ("Paris" ⊗ temperature_profile)
- ❌ Compose functions algebraically
- ❌ Query "what function matches this behavior?"

**Why this matters:**
- **Robotics:** Motor control functions change over time
- **Signal processing:** Filters as transformations
- **Science:** Physical laws as functional relationships
- **Control systems:** Policies as functions of state

---

## The Solution: Vector Function Architecture (VFA)

**Key insight:** Represent functions `f(x)` as hypervectors using Reproducing Kernel Hilbert Space (RKHS).

### What VFA Enables

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.vfa import VectorFunctionEncoder
import jax.numpy as jnp

# Create model
model = create_fhrr_model(dim=512)
memory = VSAMemory(model)
vfa = VectorFunctionEncoder(model, memory)

# Encode temperature function from samples
times = jnp.linspace(0, 24, 50)  # 24 hours
temps = 20 + 5 * jnp.sin(2 * jnp.pi * times / 24)

temp_function = vfa.encode_function_1d(times, temps)

# Now temperature is a hypervector! We can:
# 1. Evaluate at any point
temp_at_3pm = vfa.evaluate_1d(temp_function, 15.0)  # 15:00 hours
print(f"Temperature at 3pm: {temp_at_3pm:.2f}°C")

# 2. Bind to location
memory.add("Paris")
paris_temp = model.opset.bind(memory["Paris"].vec, temp_function)

# 3. Compare functions
humidity_samples = 60 + 10 * jnp.cos(2 * jnp.pi * times / 24)
humidity_function = vfa.encode_function_1d(times, humidity_samples)

from vsax.similarity import cosine_similarity
sim = cosine_similarity(temp_function, humidity_function)
print(f"Temperature vs humidity pattern similarity: {sim:.3f}")

# 4. Add functions
combined = vfa.add_functions(temp_function, humidity_function)
```

**Result:** Functions are now **symbolic** - they can be stored, compared, composed, and reasoned about!

---

## Mathematical Foundation

### Reproducing Kernel Hilbert Space (RKHS)

VFA represents functions in an RKHS with a **fractional power kernel**.

**Function representation:**
```
f(x) ≈ <α, z^x>
```

Where:
- `α` (alpha) = coefficient hypervector (learned from data)
- `z` (zeta) = random basis hypervector
- `z^x` = "raise z to power x" using FPE
- `<·,·>` = inner product (dot product for complex vectors)

**Why this works:**
- Each component `z_i^x` is a basis function
- `α_i` weights how much basis function `i` contributes
- Inner product sums weighted basis functions

### Kernel Function

The kernel is:
```
K(x, x') = <z^x, z^x'> = z^(x-x')
```

**Properties:**
- **Smooth:** Nearby points x, x' → high kernel value
- **Translation invariant:** Depends only on difference x - x'
- **Continuous:** Small Δx → small change in kernel

This enables **generalization** from samples to unseen points.

### Learning from Samples

Given training samples `(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)`:

**Step 1:** Build design matrix `Z`
```
Z[i, j] = (z_j)^(x_i)
```

This is an `n × d` matrix where:
- `n` = number of samples
- `d` = hypervector dimensionality

**Step 2:** Solve for coefficients `α`
```
Z * α = y
```

Using **regularized least squares** (Ridge regression):
```
α = (Z^H Z + λI)^(-1) Z^H y
```

Where:
- `Z^H` = conjugate transpose (for complex vectors)
- `λ` = regularization parameter (prevents overfitting)
- `I` = identity matrix

**Step 3:** Coefficient hypervector `α` now encodes the function!

### Evaluation

To evaluate `f(x_query)` at a new point:

```
f(x_query) = <α, z^x_query>
           = Σᵢ αᵢ * (zᵢ)^x_query
```

**Complexity:** O(d) where d is dimensionality - constant regardless of training set size!

---

## Basic Usage

### Encoding 1D Functions

```python
import jax.numpy as jnp
from vsax import create_fhrr_model, VSAMemory
from vsax.vfa import VectorFunctionEncoder

# Create VFA encoder
model = create_fhrr_model(dim=512, key=jax.random.PRNGKey(42))
memory = VSAMemory(model)
vfa = VectorFunctionEncoder(model, memory)

# Sample a function: sin(x)
x_train = jnp.linspace(0, 2*jnp.pi, 30)
y_train = jnp.sin(x_train)

# Encode
sin_function = vfa.encode_function_1d(x_train, y_train)

# Evaluate at new points
x_test = jnp.linspace(0, 2*jnp.pi, 100)
y_pred = vfa.evaluate_1d(sin_function, x_test)

# Compare to ground truth
y_true = jnp.sin(x_test)
error = jnp.mean(jnp.abs(y_pred - y_true))
print(f"Mean absolute error: {error:.4f}")
# Output: ~0.02 (excellent approximation!)
```

### Function Arithmetic

**Addition:**
```python
# Encode f(x) = sin(x)
f_hv = vfa.encode_function_1d(x_train, jnp.sin(x_train))

# Encode g(x) = cos(x)
g_hv = vfa.encode_function_1d(x_train, jnp.cos(x_train))

# h(x) = f(x) + g(x) = sin(x) + cos(x)
h_hv = vfa.add_functions(f_hv, g_hv)

# Evaluate
y_pred = vfa.evaluate_1d(h_hv, 1.5)
y_true = jnp.sin(1.5) + jnp.cos(1.5)
print(f"h(1.5) = {y_pred:.3f}, true = {y_true:.3f}")
```

**Shifting:**
```python
# Shift f(x) to f(x - 1)
shift_amount = 1.0
shifted_f = vfa.shift_function(f_hv, shift_amount)

# Evaluate
y_shifted = vfa.evaluate_1d(shifted_f, 2.0)  # f(2 - 1) = f(1) = sin(1)
y_expected = jnp.sin(1.0)
print(f"Shifted: {y_shifted:.3f}, expected: {y_expected:.3f}")
```

**Scaling:**
```python
# Scale: 2 * f(x)
scaled_f = vfa.scale_function(f_hv, 2.0)

y_scaled = vfa.evaluate_1d(scaled_f, 1.0)
y_expected = 2 * jnp.sin(1.0)
print(f"Scaled: {y_scaled:.3f}, expected: {y_expected:.3f}")
```

---

## Practical Applications

### Application 1: Time Series Forecasting

Encode historical patterns and query future behavior:

```python
# Historical temperature data (hourly for 7 days)
hours = jnp.arange(0, 168)  # 7 * 24 hours
temperature = 20 + 5*jnp.sin(2*jnp.pi*hours/24) + jnp.random.normal(0, 0.5, size=168)

# Encode pattern
temp_pattern = vfa.encode_function_1d(hours, temperature)

# Forecast next 24 hours
future_hours = jnp.arange(168, 192)
forecast = vfa.evaluate_1d(temp_pattern, future_hours)

print(f"Forecast for hour 169: {forecast[1]:.2f}°C")
```

### Application 2: Signal Processing

Encode signals and perform filtering:

```python
# Noisy signal
t = jnp.linspace(0, 10, 200)
signal = jnp.sin(2*jnp.pi*t) + 0.3*jnp.random.normal(0, 1, size=200)

# Encode
signal_hv = vfa.encode_function_1d(t, signal, lambda_reg=0.1)  # Higher reg = smoother

# Evaluate for smoothing effect
t_smooth = jnp.linspace(0, 10, 500)
smoothed = vfa.evaluate_1d(signal_hv, t_smooth)

# VFA acts as implicit low-pass filter!
```

### Application 3: Control Systems

Encode control policies:

```python
# Control policy: action as function of state
states = jnp.linspace(-1, 1, 50)  # Possible states
actions = jnp.tanh(states * 2)  # Control action (e.g., steering angle)

# Encode policy
policy = vfa.encode_function_1d(states, actions)

# Store in memory
memory.add("steering_policy")
memory["steering_policy"] = model.rep_cls(policy)

# Query: what action for current state?
current_state = 0.3
action = vfa.evaluate_1d(policy, current_state)
print(f"State {current_state:.2f} → Action {action:.2f}")
```

### Application 4: Physics Simulation

Encode physical relationships:

```python
# Hooke's law: F = -kx (spring force)
displacements = jnp.linspace(-1, 1, 30)
forces = -2.0 * displacements  # k = 2.0

# Encode
spring_law = vfa.encode_function_1d(displacements, forces)

# Bind to concept
memory.add("spring_force")
physics_knowledge = model.opset.bind(
    memory["spring_force"].vec,
    spring_law
)

# Query: force at displacement 0.5?
force = vfa.evaluate_1d(spring_law, 0.5)
print(f"Force at x=0.5: {force:.2f} N")  # Should be ≈ -1.0
```

---

## VFA vs Neural Networks

| Feature | VFA | Neural Networks |
|---------|-----|-----------------|
| **Training** | Single least-squares solve | Iterative gradient descent |
| **Speed** | Very fast (closed-form) | Slow (many epochs) |
| **Samples needed** | Few (10-100) | Many (1000s-millions) |
| **Interpolation** | Excellent | Good |
| **Extrapolation** | Poor (kernel-based) | Better (learned features) |
| **Interpretability** | High (linear in RKHS) | Low (black box) |
| **Memory** | Fixed (dim d) | Variable (network size) |
| **Symbolic integration** | Native (it's a hypervector!) | Difficult |

**When to use VFA:**
✅ Few samples available
✅ Fast training needed (real-time)
✅ Symbolic reasoning required
✅ Smooth interpolation needed
✅ Function must integrate with VSA

**When to use Neural Networks:**
✅ Large datasets available
✅ Complex non-smooth functions
✅ Extrapolation critical
✅ Deep hierarchical features needed

---

## Advanced Features

### Multi-Dimensional Functions

VFA supports functions of multiple variables:

```python
# f(x, y) = sin(x) * cos(y)
x = jnp.linspace(0, jnp.pi, 20)
y = jnp.linspace(0, jnp.pi, 20)
X, Y = jnp.meshgrid(x, y)

z = jnp.sin(X) * jnp.cos(Y)

# Encode 2D function
f_2d = vfa.encode_function_2d(X.flatten(), Y.flatten(), z.flatten())

# Evaluate at point (1.0, 1.5)
z_pred = vfa.evaluate_2d(f_2d, 1.0, 1.5)
z_true = jnp.sin(1.0) * jnp.cos(1.5)
print(f"f(1.0, 1.5) = {z_pred:.3f}, true = {z_true:.3f}")
```

### Regularization Control

Tune smoothness vs fit:

```python
# Low regularization: fits noise
f_noisy = vfa.encode_function_1d(x, y, lambda_reg=0.001)

# High regularization: smooth but biased
f_smooth = vfa.encode_function_1d(x, y, lambda_reg=1.0)

# Optimal: cross-validation
# (Try multiple lambda values, pick best on validation set)
```

### Function Similarity

Compare functional behaviors:

```python
# Encode multiple functions
f1 = vfa.encode_function_1d(x, jnp.sin(x))
f2 = vfa.encode_function_1d(x, jnp.sin(x + 0.1))  # Slightly shifted
f3 = vfa.encode_function_1d(x, jnp.cos(x))  # Different

from vsax.similarity import cosine_similarity

sim_12 = cosine_similarity(f1, f2)
sim_13 = cosine_similarity(f1, f3)

print(f"sin(x) vs sin(x+0.1): {sim_12:.3f}")  # High (similar)
print(f"sin(x) vs cos(x): {sim_13:.3f}")  # Lower (different)
```

---

## Performance Considerations

### Dimensionality

| Dimension | Training Samples | Accuracy | Speed |
|-----------|-----------------|----------|-------|
| 128 | 10-20 | Good | Very fast |
| 512 | 20-50 | Better | Fast |
| 2048 | 50-100 | Best | Medium |

**Recommendation:** Start with `dim=512` for most applications.

### Sample Density

More samples → better approximation:

- **Sparse (10-20 samples):** Smooth functions only
- **Medium (50-100 samples):** Most real-world signals
- **Dense (200+ samples):** High-frequency or noisy data

### Regularization

`lambda_reg` controls smoothness:

- **0.001:** Minimal smoothing (fits noise)
- **0.01:** Balanced (default)
- **0.1:** Strong smoothing (may underfit)

**Best practice:** Use cross-validation to select λ.

---

## Common Pitfalls

### Problem 1: Using Non-FHRR Model

```python
# ❌ VFA requires FHRR
model = create_map_model(512)
vfa = VectorFunctionEncoder(model, memory)
# Error: VFA requires ComplexHypervector
```

**Fix:** Always use `create_fhrr_model()`.

### Problem 2: Too Few Samples

```python
# ❌ Only 3 samples for complex function
x = jnp.array([0, 1, 2])
y = jnp.sin(2 * jnp.pi * x) + jnp.cos(3 * jnp.pi * x)
f = vfa.encode_function_1d(x, y)
# Poor approximation!
```

**Fix:** Use at least 20-30 samples for smooth functions, 50+ for complex ones.

### Problem 3: Extrapolation

```python
# ❌ Training on [0, 2π], querying outside
x_train = jnp.linspace(0, 2*jnp.pi, 30)
f = vfa.encode_function_1d(x_train, jnp.sin(x_train))

y_extrap = vfa.evaluate_1d(f, 10.0)  # Outside training range
# Unreliable extrapolation!
```

**Fix:** Only query within training range, or use periodic functions.

---

## Self-Assessment

Before moving on, ensure you can:

- [ ] Explain why VFA encodes functions as hypervectors
- [ ] Describe the RKHS representation and fractional power kernel
- [ ] Encode a function from samples using VFA
- [ ] Evaluate encoded functions at new points
- [ ] Perform function arithmetic (add, shift, scale)
- [ ] Choose appropriate dimensionality and regularization
- [ ] Understand when to use VFA vs neural networks

## Quick Quiz

**Question 1:** What is the key mathematical structure that VFA uses?

a) Fourier series
b) Reproducing Kernel Hilbert Space (RKHS)
c) Convolutional neural networks
d) Decision trees

<details>
<summary>Answer</summary>
**b) Reproducing Kernel Hilbert Space (RKHS)**

VFA represents functions in an RKHS with fractional power kernel `K(x, x') = z^(x-x')`. This allows linear function representation `f(x) = <α, z^x>` with smooth interpolation and efficient learning via least squares.
</details>

**Question 2:** Why does VFA require FHRR (ComplexHypervector)?

a) For better performance
b) Because fractional powers need complex phase representation
c) To save memory
d) It doesn't - VFA works with any model

<details>
<summary>Answer</summary>
**b) Because fractional powers need complex phase representation**

VFA uses `z^x` where x is continuous. This requires complex vectors where exponentiation via phase manipulation is well-defined. MAP and Binary models cannot support continuous fractional powers.
</details>

**Question 3:** How does VFA compare to neural networks for function approximation with 20 training samples?

a) Neural networks are always better
b) VFA is faster to train and better at interpolation
c) They perform identically
d) Neural networks are faster

<details>
<summary>Answer</summary>
**b) VFA is faster to train and better at interpolation**

With few samples (20), VFA trains instantly via closed-form least squares and provides excellent smooth interpolation. Neural networks require many epochs of gradient descent and may overfit with so few samples.
</details>

---

## Hands-On Exercise

**Task:** Encode and compare multiple mathematical functions.

**Requirements:**
1. Encode sin(x), cos(x), and x² as VFA hypervectors
2. Evaluate each at 10 test points
3. Compute pairwise function similarities
4. Create h(x) = sin(x) + cos(x) using function addition
5. Verify h(x) matches ground truth

**Starter code:**

```python
import jax
import jax.numpy as jnp
from vsax import create_fhrr_model, VSAMemory
from vsax.vfa import VectorFunctionEncoder
from vsax.similarity import cosine_similarity

model = create_fhrr_model(dim=512, key=jax.random.PRNGKey(42))
memory = VSAMemory(model)
vfa = VectorFunctionEncoder(model, memory)

# Sample points
x_train = jnp.linspace(0, 2*jnp.pi, 30)

# YOUR CODE HERE:
# 1. Encode sin(x), cos(x), x²
# 2. Evaluate at test points
# 3. Compute similarities
# 4. Add sin + cos
# 5. Verify
```

<details>
<summary>Solution</summary>

```python
# 1. Encode functions
sin_hv = vfa.encode_function_1d(x_train, jnp.sin(x_train))
cos_hv = vfa.encode_function_1d(x_train, jnp.cos(x_train))
quadratic_hv = vfa.encode_function_1d(x_train, x_train**2)

# 2. Evaluate at test points
x_test = jnp.linspace(0, 2*jnp.pi, 10)

sin_pred = vfa.evaluate_1d(sin_hv, x_test)
cos_pred = vfa.evaluate_1d(cos_hv, x_test)
quad_pred = vfa.evaluate_1d(quadratic_hv, x_test)

print("Evaluation at test points:")
for i in range(3):
    print(f"sin({x_test[i]:.2f}) = {sin_pred[i]:.3f}, true = {jnp.sin(x_test[i]):.3f}")

# 3. Compute pairwise similarities
sim_sin_cos = cosine_similarity(sin_hv, cos_hv)
sim_sin_quad = cosine_similarity(sin_hv, quadratic_hv)
sim_cos_quad = cosine_similarity(cos_hv, quadratic_hv)

print(f"\nFunction similarities:")
print(f"sin vs cos: {sim_sin_cos:.3f}")
print(f"sin vs x²: {sim_sin_quad:.3f}")
print(f"cos vs x²: {sim_cos_quad:.3f}")

# 4. Add sin + cos
sum_hv = vfa.add_functions(sin_hv, cos_hv)

# 5. Verify
sum_pred = vfa.evaluate_1d(sum_hv, x_test)
sum_true = jnp.sin(x_test) + jnp.cos(x_test)

error = jnp.mean(jnp.abs(sum_pred - sum_true))
print(f"\nsin(x) + cos(x) mean error: {error:.4f}")
```
</details>

---

## Key Takeaways

✓ **VFA encodes functions as hypervectors** - enables symbolic function manipulation
✓ **RKHS with fractional power kernel** - `f(x) = <α, z^x>`
✓ **Fast training via least squares** - closed-form solution, no gradient descent
✓ **Excellent interpolation** - smooth kernel provides good generalization
✓ **Function arithmetic** - add, shift, scale functions algebraically
✓ **Requires FHRR** - complex phase representation needed for fractional powers
✓ **Ideal for few samples** - works well with 20-100 training points

---

## Next Steps

**Next Lesson:** [Lesson 5.2 - Building Custom Encoders](02_custom_encoders.md)
Learn how to design and implement domain-specific encoders from scratch.

**For Deep Technical Details:** [Vector Function Architecture Guide](../../guide/vfa.md)
Comprehensive reference with advanced features, multi-dimensional functions, and implementation details.

**Related Content:**
- [Module 3, Lesson 3.1 - Fractional Power Encoding](../../course/03_encoders/01_scalar_sequence.md)
- [VFA Examples](../../examples/vfa/)

## References

- Frady, E. P., Kleyko, D., & Sommer, F. T. (2021). "Variable Binding for Sparse Distributed Representations: Theory and Applications." *IEEE Transactions on Neural Networks and Learning Systems.*
- Kleyko, D., et al. (2022). "Vector Symbolic Architectures as a Computing Framework for Nanoscale Hardware." *Proceedings of the IEEE.*
- Plate, T. A. (1995). "Holographic Reduced Representations." *IEEE Transactions on Neural Networks.*
