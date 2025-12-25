# Spatial Semantic Pointers

**NEW in v1.2.0** - Continuous spatial representation using Fractional Power Encoding.

## Overview

Spatial Semantic Pointers (SSP) enable encoding and querying **continuous spatial locations** using hypervectors. Based on Komer et al. (2019), SSPs use Fractional Power Encoding to represent coordinates in 1D, 2D, 3D, or higher-dimensional spaces.

**Key insight:** SSP binds objects to spatial locations, enabling "what is at (x, y)?" and "where is object O?" queries.

| Capability | Example Query |
|------------|---------------|
| **Encode location** | Represent point (3.5, 2.1) as hypervector |
| **Bind object-location** | "apple at (3.5, 2.1)" |
| **Query by location** | "What is at (3.5, 2.1)?" → apple |
| **Query by object** | "Where is apple?" → (3.5, 2.1) |
| **Scene shifting** | Move all objects by offset vector |

## Why Spatial Semantic Pointers?

### Problem: Discrete Spatial Encoding

**Without SSP**, encoding spatial locations requires discretization:

```python
# Encode "apple at grid cell (3, 2)"
# Problem: Cannot represent (3.5, 2.1) precisely
# Must round to nearest grid: (3, 2) or (4, 2)
```

### Solution: SSP Encodes Continuous Space

**With SSP**, represent exact coordinates:

```python
from vsax.spatial import SpatialSemanticPointers, SSPConfig

# Create 2D spatial representation
ssp = SpatialSemanticPointers(model, memory, SSPConfig(num_axes=2))

# Encode exact location
location = ssp.encode_location([3.5, 2.1])

# Bind object to location
memory.add("apple")
scene = ssp.bind_object_location("apple", [3.5, 2.1])

# Query: what's at (3.5, 2.1)?
result = ssp.query_location(scene, [3.5, 2.1])
# result is similar to "apple" hypervector
```

### Advantages

✅ **Continuous** - Represent arbitrary real-valued coordinates
✅ **Compositional** - Bind multiple object-location pairs
✅ **Queryable** - Ask "what" or "where" questions
✅ **Shiftable** - Transform entire scenes spatially
✅ **Smooth** - Nearby locations have high similarity
✅ **Scalable** - Works in any number of dimensions

## Mathematical Foundation

### How SSP Works

SSP encodes spatial location `(x, y)` as:

```
S(x, y) = X^x ⊗ Y^y
```

Where:
- `X`, `Y` are basis hypervectors (random FHRR vectors)
- `X^x` means "raise X to power x" (Fractional Power Encoding)
- `⊗` is binding (circular convolution for FHRR)

**For 3D:**
```
S(x, y, z) = X^x ⊗ Y^y ⊗ Z^z
```

### Binding Objects to Locations

To encode "object O at location (x, y)":

```
Object ⊗ S(x, y) = Object ⊗ X^x ⊗ Y^y
```

### Querying

**"What is at (x, y)?"**
```
Scene ⊗ S(x, y)^(-1) ≈ Object
```

**"Where is Object?"**
```
Scene ⊗ Object^(-1) ≈ S(x, y)
```

### Why FHRR Only?

SSP **requires ComplexHypervector (FHRR)** because:
1. Fractional powers (`X^x`) only work with complex phase representation
2. Exact unbinding requires invertible operations
3. Smooth spatial representation needs continuous encoding

## Basic Usage

### Creating SSP System

```python
import jax
from vsax import create_fhrr_model, VSAMemory
from vsax.spatial import SpatialSemanticPointers, SSPConfig

# Create FHRR model
model = create_fhrr_model(dim=512, key=jax.random.PRNGKey(0))
memory = VSAMemory(model)

# Configure spatial dimensions
config = SSPConfig(
    dim=512,
    num_axes=2,  # 2D space
    scale=None,  # Optional coordinate scaling
    axis_names=["x", "y"]  # Optional custom names
)

# Create SSP
ssp = SpatialSemanticPointers(model, memory, config)
```

### Encoding Locations

```python
# Encode 2D point
location = ssp.encode_location([3.5, 2.1])
print(type(location))  # ComplexHypervector

# Encode 3D point (requires num_axes=3)
# location_3d = ssp.encode_location([1.0, 2.0, 3.0])
```

### Binding Objects to Locations

```python
# Add objects to memory
memory.add_many(["apple", "banana", "cherry"])

# Bind each object to a location
apple_here = ssp.bind_object_location("apple", [1.0, 2.0])
banana_there = ssp.bind_object_location("banana", [3.0, 4.0])
cherry_far = ssp.bind_object_location("cherry", [5.0, 1.0])
```

### Creating Scenes

Bundle multiple object-location pairs:

```python
# Create scene with multiple objects
scene = model.opset.bundle(
    apple_here.vec,
    banana_there.vec,
    cherry_far.vec
)
scene_hv = ComplexHypervector(scene)

# Or use utility function
from vsax.spatial.utils import create_spatial_scene

scene = create_spatial_scene(ssp, {
    "apple": [1.0, 2.0],
    "banana": [3.0, 4.0],
    "cherry": [5.0, 1.0]
})
```

### Querying Scenes

#### Query "What is at location?"

```python
# What's at (3.0, 4.0)?
result_hv = ssp.query_location(scene, [3.0, 4.0])

# Check similarity to known objects
from vsax.similarity import cosine_similarity

for obj in ["apple", "banana", "cherry"]:
    sim = cosine_similarity(result_hv.vec, memory[obj].vec)
    print(f"{obj}: {sim:.3f}")

# Output:
# apple: 0.123
# banana: 0.847  ← Highest!
# cherry: 0.201
```

#### Query "Where is object?"

```python
# Where is the apple?
location_hv = ssp.query_object(scene, "apple")

# Decode location (grid search)
coords = ssp.decode_location(
    location_hv,
    search_range=[(0.0, 6.0), (0.0, 5.0)],
    resolution=50
)
print(f"Apple at: {coords}")  # ≈ [1.0, 2.0]
```

## Common Use Cases

### 1. Spatial Navigation

Encode environment with landmarks:

```python
# 2D room layout
memory.add_many(["door", "window", "table", "chair"])

# Create room scene
room = create_spatial_scene(ssp, {
    "door": [0.0, 5.0],
    "window": [5.0, 5.0],
    "table": [2.5, 2.5],
    "chair": [2.0, 2.0]
})

# Query: what's near the center (2.5, 2.5)?
center = ssp.query_location(room, [2.5, 2.5])
# Should be most similar to "table"
```

### 2. Robotics and Localization

Track object positions:

```python
# Robot's world model
world = create_spatial_scene(ssp, {
    "obstacle1": [3.0, 4.0],
    "obstacle2": [5.0, 2.0],
    "goal": [8.0, 8.0]
})

# Where is the goal?
goal_loc = ssp.query_object(world, "goal")
coords = ssp.decode_location(goal_loc, [(0, 10), (0, 10)])
```

### 3. Geographic Information Systems

Encode points of interest:

```python
# City map (latitude, longitude)
memory.add_many(["library", "park", "cafe", "museum"])

city = create_spatial_scene(ssp, {
    "library": [40.7589, -73.9851],
    "park": [40.7829, -73.9654],
    "cafe": [40.7614, -73.9776],
    "museum": [40.7794, -73.9632]
})
```

### 4. Scientific Data

Encode experimental measurements:

```python
# 3D spatial measurements
config_3d = SSPConfig(dim=512, num_axes=3, axis_names=["x", "y", "z"])
ssp_3d = SpatialSemanticPointers(model, memory, config_3d)

# Particle positions
particle_data = create_spatial_scene(ssp_3d, {
    "particle_1": [1.2, 3.4, 5.6],
    "particle_2": [2.1, 4.3, 6.5],
    "particle_3": [3.0, 5.2, 7.4]
})
```

## Advanced Features

### Scene Shifting

Translate entire scene by offset:

```python
# Original scene: apple at (3.5, 2.1)
scene = ssp.bind_object_location("apple", [3.5, 2.1])

# Shift by (+1.0, -0.5)
shifted_scene = ssp.shift_scene(scene, [1.0, -0.5])

# Apple now at (4.5, 1.6)
new_loc = ssp.query_object(shifted_scene, "apple")
coords = ssp.decode_location(new_loc, [(0, 10), (0, 10)])
# coords ≈ [4.5, 1.6]
```

### Similarity Maps

Visualize spatial distributions:

```python
from vsax.spatial.utils import similarity_map_2d

# Create scene
scene = ssp.bind_object_location("apple", [3.5, 2.1])

# Where is apple? (heatmap)
apple_loc = ssp.query_object(scene, "apple")

X, Y, similarities = similarity_map_2d(
    ssp,
    apple_loc,
    x_range=(0.0, 5.0),
    y_range=(0.0, 5.0),
    resolution=50
)

# Plot with matplotlib
import matplotlib.pyplot as plt
plt.contourf(X, Y, similarities, levels=20)
plt.colorbar()
plt.title("Apple location heatmap")
plt.show()
```

### Region Queries

Find objects within spatial region:

```python
from vsax.spatial.utils import region_query

scene = create_spatial_scene(ssp, {
    "apple": [1.0, 1.0],
    "banana": [3.0, 3.0],
    "cherry": [5.0, 5.0]
})

# What's near (3.0, 3.0) within radius 0.5?
results = region_query(
    ssp, scene,
    object_names=["apple", "banana", "cherry"],
    center=[3.0, 3.0],
    radius=0.5
)

# results: {"apple": 0.23, "banana": 0.89, "cherry": 0.18}
# Banana has highest similarity!
```

### 2D Scene Visualization

```python
from vsax.spatial.utils import plot_ssp_2d_scene

scene = create_spatial_scene(ssp, {
    "apple": [1.0, 2.0],
    "banana": [3.0, 4.0]
})

fig = plot_ssp_2d_scene(
    ssp, scene,
    object_names=["apple", "banana"],
    x_range=(0, 5),
    y_range=(0, 5),
    resolution=30
)
plt.show()
```

## SSP vs CliffordOperator

SSP and CliffordOperator serve **different purposes** - use both together!

| Feature | SSP | CliffordOperator |
|---------|-----|------------------|
| **Purpose** | Continuous spatial coordinates | Discrete symbolic relations |
| **Example** | "apple at (3.5, 2.1)" | "cup LEFT_OF plate" |
| **Encoding** | `X^x ⊗ Y^y` | Directional transformation |
| **Inversion** | Approximate (similarity > 0.7) | Exact (similarity > 0.999) |
| **Query type** | "What/where?" | "What relation?" |

### Using Both Together

```python
from vsax.operators import CliffordOperator, OperatorKind

# Combine spatial location + symbolic relation
LEFT_OF = CliffordOperator.random(512, kind=OperatorKind.SPATIAL)

# "cup is at (2.0, 3.0) and LEFT_OF plate"
cup_at_pos = ssp.bind_object_location("cup", [2.0, 3.0])
plate_relation = LEFT_OF.apply(memory["plate"])

# Combined representation
scene = model.opset.bundle(cup_at_pos.vec, plate_relation.vec)
```

## Configuration Options

### SSPConfig

```python
from vsax.spatial import SSPConfig

config = SSPConfig(
    dim=512,              # Hypervector dimensionality
    num_axes=2,          # 1D, 2D, 3D, or higher
    scale=None,          # Optional: scale coordinates (e.g., 0.1)
    axis_names=None      # Optional: ["latitude", "longitude"]
)
```

**Automatic axis naming:**
- 1D-3D: `["x", "y", "z"]`
- 4D+: `["axis_0", "axis_1", ...]`

### Scaling Coordinates

For large coordinate ranges, use scaling:

```python
# Geographic coordinates (latitude: -90 to +90, longitude: -180 to +180)
config = SSPConfig(
    dim=512,
    num_axes=2,
    scale=0.01  # Maps ±90, ±180 to ±0.9, ±1.8
)

ssp = SpatialSemanticPointers(model, memory, config)
location = ssp.encode_location([40.7589, -73.9851])
```

## Performance Considerations

### Encoding Cost

**Single location:** O(num_axes × dim)
**Scene with N objects:** O(N × num_axes × dim)

All operations are GPU-accelerated via JAX.

### Decoding Accuracy

Decoding uses grid search - trade-off between speed and accuracy:

| Resolution | Decode Time | Accuracy |
|------------|-------------|----------|
| 10 | Fast | ±0.5 |
| 20 | Medium | ±0.2 |
| 50 | Slow | ±0.05 |
| 100 | Very slow | ±0.02 |

### Capacity

SSP can store many object-location pairs in a single scene:

| Dimensionality | Objects per Scene |
|----------------|-------------------|
| 512 | ~50 objects |
| 1024 | ~100 objects |
| 2048 | ~200 objects |

Beyond capacity, similarity scores degrade.

## Design Principles

### 1. FPE Foundation

SSP is built on FractionalPowerEncoder:

```python
# SSP creates FPE internally
self.encoder = FractionalPowerEncoder(model, memory, scale=config.scale)

# Uses FPE for coordinate encoding
def encode_location(self, coordinates):
    return self.encoder.encode_multi(self.axis_names, coordinates)
```

### 2. FHRR-Only

SSP requires ComplexHypervector for phase-based fractional powers.

```python
# Type checking enforced
model = create_map_model(512)
ssp = SpatialSemanticPointers(model, memory)
# Raises: TypeError
```

### 3. Immutable

All operations return new hypervectors; original scene unchanged.

## Best Practices

### When to Use SSP

✅ **Use SSP for:**
- Continuous spatial coordinates (2D maps, 3D environments)
- Navigation and localization
- Geographic information systems
- Spatial reasoning tasks
- Object-location binding

❌ **Use other approaches for:**
- Discrete symbolic relations → CliffordOperator
- Grid-based environments → Direct encoding
- Pure object attributes → DictEncoder

### Choosing Dimensionality

Higher dimensions → better accuracy but slower:

- **512**: Good for simple 2D scenes (10-50 objects)
- **1024**: Better for complex 3D environments
- **2048**: High-precision spatial reasoning

### Choosing Resolution

For decoding, balance speed vs accuracy:

```python
# Quick approximation
coords = ssp.decode_location(loc_hv, [(0, 10), (0, 10)], resolution=10)

# Precise recovery
coords = ssp.decode_location(loc_hv, [(0, 10), (0, 10)], resolution=100)
```

### Multi-Object Scenes

Bundle judiciously - too many objects reduces accuracy:

```python
# Good: 10-50 objects
scene = create_spatial_scene(ssp, {f"obj_{i}": [i, i] for i in range(20)})

# Risky: 100+ objects (may degrade)
# scene = create_spatial_scene(ssp, {f"obj_{i}": [i, i] for i in range(200)})
```

## Limitations

### Current Limitations

1. **FHRR-only** - Cannot use Binary or Real hypervectors
2. **Approximate decoding** - Grid search is approximate and slow
3. **Capacity bounds** - Too many objects → similarity decay
4. **No learned representations** - Basis vectors are random

### Workarounds

**For precise decoding:**
```python
# Use fine-grained grid + multi-stage refinement
# Stage 1: coarse grid
coords_coarse = ssp.decode_location(loc, [(0, 10), (0, 10)], resolution=20)

# Stage 2: refine around peak
x_min, x_max = coords_coarse[0] - 0.5, coords_coarse[0] + 0.5
coords_fine = ssp.decode_location(loc, [(x_min, x_max), ...], resolution=50)
```

**For large scenes:**
```python
# Partition space into regions
# Encode region ID + local coordinates
```

## Related Topics

- **Tutorial 11:** [Analogical Reasoning with Conceptual Spaces](../tutorials/11_analogical_reasoning.md)
- **Guide:** [Fractional Power Encoding](fpe.md)
- **Guide:** [Operators](operators.md)
- **API Reference:** [Spatial API](../api/spatial/index.md)
- **Examples:** [SSP 1D](../../examples/spatial/ssp_1d_line.py), [SSP 2D](../../examples/spatial/ssp_2d_navigation.py)

## References

**Theoretical Foundation:**
- Komer et al. (2019) - "A neural representation of continuous space using fractional binding"
- Plate (1995) - "Holographic Reduced Representations"
- Gayler (2003) - "Vector symbolic architectures answer Jackendoff's challenges"

**Applications:**
- Spatial navigation (Komer 2019)
- Path integration (Dumont & Eliasmith 2020)
- Cognitive maps (Tolman 1948)
