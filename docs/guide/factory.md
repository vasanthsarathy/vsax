# Factory Functions: Easy Model Creation

Factory functions provide a simple, one-line way to create VSA models with sensible defaults. Instead of manually configuring representations, operation sets, and samplers, use factory functions for quick setup.

## Available Factory Functions

VSAX provides three factory functions, one for each VSA model type:

- `create_fhrr_model()` - Complex hypervectors with FFT-based operations
- `create_map_model()` - Real hypervectors with element-wise operations
- `create_binary_model()` - Binary hypervectors with XOR/majority operations

## create_fhrr_model

Create a FHRR (Fourier Holographic Reduced Representation) model.

```python
from vsax import create_fhrr_model

# Default dimension (512)
model = create_fhrr_model()

# Custom dimension
model = create_fhrr_model(dim=1024)
```

**Properties:**
- Uses `ComplexHypervector` (complex-valued)
- Uses `FHRROperations` (FFT-based circular convolution)
- Uses `sample_complex_random` (unit magnitude, random phase)
- **Default dimension**: 512
- **Unbinding**: Exact (via complex conjugate)

**When to use:**
- Need exact unbinding
- Working with sequential/temporal data
- Frequency-domain representations

## create_map_model

Create a MAP (Multiply-Add-Permute) model.

```python
from vsax import create_map_model

# Default dimension (512)
model = create_map_model()

# Custom dimension
model = create_map_model(dim=2048)
```

**Properties:**
- Uses `RealHypervector` (real-valued)
- Uses `MAPOperations` (element-wise multiplication/mean)
- Uses `sample_random` (Gaussian distribution)
- **Default dimension**: 512
- **Unbinding**: Approximate

**When to use:**
- Continuous feature representations
- Approximate pattern matching
- Lower memory footprint than complex

## create_binary_model

Create a Binary VSA model.

```python
from vsax import create_binary_model

# Default dimension (10000), bipolar mode
model = create_binary_model()

# Custom dimension
model = create_binary_model(dim=5000)

# Binary mode {0, 1} instead of bipolar {-1, +1}
model = create_binary_model(dim=10000, bipolar=False)
```

**Properties:**
- Uses `BinaryHypervector` (discrete binary/bipolar)
- Uses `BinaryOperations` (XOR for bind, majority for bundle)
- Uses `sample_binary_random` (random bipolar or binary)
- **Default dimension**: 10000 (higher than continuous models)
- **Unbinding**: Exact (self-inverse property)
- **Default mode**: Bipolar (`{-1, +1}`)

**When to use:**
- Need exact unbinding with minimal computation
- Boolean/logical operations
- Hardware-friendly representations
- Very large symbol spaces (use higher dimensions)

## Comparison

| Model | Type | Dimension | Unbinding | Memory | Speed |
|-------|------|-----------|-----------|--------|-------|
| FHRR | Complex | 512 (default) | Exact | Medium | Medium (FFT) |
| MAP | Real | 512 (default) | Approximate | Low | Fast |
| Binary | Discrete | 10000 (default) | Exact | Very Low | Very Fast |

## Complete Example

```python
from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory
import jax

# Create all three models
fhrr = create_fhrr_model(dim=512)
map_model = create_map_model(dim=512)
binary = create_binary_model(dim=10000, bipolar=True)

# All models work with VSAMemory
for model in [fhrr, map_model, binary]:
    memory = VSAMemory(model, key=jax.random.PRNGKey(42))
    memory.add_many(["concept1", "concept2"])

    # Same interface across all models
    c1 = memory["concept1"]
    c2 = memory["concept2"]

    # Bind and bundle
    bound = model.opset.bind(c1.vec, c2.vec)
    bundled = model.opset.bundle(c1.vec, c2.vec)
```

## Versus Manual Creation

**Before (v0.2.0):**
```python
from vsax import VSAModel, ComplexHypervector, FHRROperations, sample_complex_random

model = VSAModel(
    dim=512,
    rep_cls=ComplexHypervector,
    opset=FHRROperations(),
    sampler=sample_complex_random
)
```

**After (v0.3.0):**
```python
from vsax import create_fhrr_model

model = create_fhrr_model(dim=512)
```

Much simpler! Factory functions reduce boilerplate while maintaining full flexibility.

## Advanced: Custom Models

If you need custom configurations, you can still use `VSAModel` directly:

```python
from vsax import VSAModel, RealHypervector, MAPOperations

# Custom sampler
def my_sampler(dim, n, key):
    return jax.random.uniform(key, shape=(n, dim)) * 2 - 1

model = VSAModel(
    dim=256,
    rep_cls=RealHypervector,
    opset=MAPOperations(),
    sampler=my_sampler
)
```

But for 95% of use cases, factory functions are sufficient.

## API Reference

```python
def create_fhrr_model(dim: int = 512, key: Optional[jax.Array] = None) -> VSAModel:
    """Create FHRR model with complex hypervectors."""

def create_map_model(dim: int = 512, key: Optional[jax.Array] = None) -> VSAModel:
    """Create MAP model with real hypervectors."""

def create_binary_model(
    dim: int = 10000,
    bipolar: bool = True,
    key: Optional[jax.Array] = None
) -> VSAModel:
    """Create Binary model with discrete hypervectors."""
```

## Next Steps

- [VSAMemory Guide](memory.md) - Symbol table management
- [Operations Guide](operations.md) - Binding and bundling
- [Examples](../examples/) - Complete working examples
