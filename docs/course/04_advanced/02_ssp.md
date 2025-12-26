# Lesson 4.2: Spatial Semantic Pointers

**Estimated time:** 50 minutes

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why continuous spatial encoding matters for real-world applications
- Understand how Spatial Semantic Pointers (SSP) encode coordinates using FPE
- Encode objects at specific locations in 2D/3D space
- Query "what is at location (x, y)?" and "where is object O?"
- Apply SSP to navigation, robotics, and geographic applications

## Prerequisites

- Module 3, Lesson 3.1 (Fractional Power Encoding)
- Understanding of binding and bundling operations
- Familiarity with FHRR model

---

## The Problem: Encoding Continuous Space

In the real world, objects exist at **continuous coordinates**, not discrete grid positions:

- A robot at position `(3.47, 2.81)` meters
- A landmark at GPS coordinates `(40.7589, -73.9851)`
- A particle at 3D position `(1.23, 4.56, 7.89)`

### Why Grid-Based Encoding Fails

With standard VSA encoding, you might try:

```python
# Attempt 1: Discretize space into grid cells
memory.add_many([f"cell_{x}_{y}" for x in range(10) for y in range(10)])

# Problem: Must round (3.47, 2.81) to nearest grid cell (3, 3)
# Loss of precision! Cannot distinguish (3.47, 2.81) from (3.51, 2.93)
```

**Limitations:**
- ❌ Loss of precision from rounding
- ❌ Fixed resolution (must choose grid size ahead of time)
- ❌ Memory explosion (100×100 grid = 10,000 basis vectors)
- ❌ No smooth similarity (adjacent cells are orthogonal)

### The Solution: Spatial Semantic Pointers

**Spatial Semantic Pointers (SSP)** encode continuous coordinates using Fractional Power Encoding:

```python
from vsax.spatial import SpatialSemanticPointers, SSPConfig

# Create 2D spatial system
model = create_fhrr_model(dim=512)
memory = VSAMemory(model)

config = SSPConfig(num_axes=2)  # 2D space (x, y)
ssp = SpatialSemanticPointers(model, memory, config)

# Encode EXACT location (3.47, 2.81)
location = ssp.encode_location([3.47, 2.81])

# Bind object to location
memory.add("robot")
scene = ssp.bind_object_location("robot", [3.47, 2.81])
```

**Advantages:**
- ✅ **Continuous** - arbitrary real-valued coordinates
- ✅ **Smooth** - nearby locations have high similarity
- ✅ **Compositional** - bind multiple object-location pairs
- ✅ **Queryable** - ask "what" or "where" questions
- ✅ **Scalable** - works in any number of dimensions

---

## How SSP Works: The Mathematics

### Core Encoding Formula

For 2D location `(x, y)`:

```
S(x, y) = X^x ⊗ Y^y
```

Where:
- `X`, `Y` are random FHRR basis vectors (one per spatial axis)
- `X^x` means "raise X to fractional power x" (using FPE)
- `⊗` is binding (circular convolution)

For 3D location `(x, y, z)`:

```
S(x, y, z) = X^x ⊗ Y^y ⊗ Z^z
```

### Why This Works

**Key insight:** Fractional Power Encoding creates smooth, continuous representations where nearby values have high similarity.

```python
# Locations close together → high similarity
loc1 = ssp.encode_location([3.0, 2.0])
loc2 = ssp.encode_location([3.1, 2.1])  # Nearby

from vsax.similarity import cosine_similarity
sim = cosine_similarity(loc1.vec, loc2.vec)
print(f"Similarity: {sim:.3f}")  # ~0.85 (high!)

# Locations far apart → low similarity
loc3 = ssp.encode_location([8.0, 9.0])  # Far away
sim = cosine_similarity(loc1.vec, loc3.vec)
print(f"Similarity: {sim:.3f}")  # ~0.12 (low)
```

This **smoothness property** is what makes SSP work for spatial reasoning.

### Building on FractionalPowerEncoder

SSP is actually a wrapper around FPE that handles multi-dimensional coordinates:

```python
# Internally, SSP does this:
from vsax.encoders import FractionalPowerEncoder

# Create one basis vector per axis
memory.add_many(["axis_x", "axis_y"])

# Encode each coordinate dimension separately
fpe = FractionalPowerEncoder(model, memory, scale=None)
x_hv = fpe.encode("axis_x", 3.47)  # X^3.47
y_hv = fpe.encode("axis_y", 2.81)  # Y^2.81

# Bind them together
location = model.opset.bind(x_hv.vec, y_hv.vec)  # X^3.47 ⊗ Y^2.81
```

**The SSP API just makes this easier:**

```python
location = ssp.encode_location([3.47, 2.81])  # Same result!
```

---

## Basic Usage: Encoding and Querying

### Step 1: Create SSP System

```python
import jax
from vsax import create_fhrr_model, VSAMemory
from vsax.spatial import SpatialSemanticPointers, SSPConfig

# IMPORTANT: SSP requires FHRR (complex vectors)
model = create_fhrr_model(dim=512, key=jax.random.PRNGKey(0))
memory = VSAMemory(model)

# Configure spatial dimensions
config = SSPConfig(
    dim=512,
    num_axes=2,  # 2D space (x, y)
    scale=None,  # Optional coordinate scaling
    axis_names=["x", "y"]  # Optional custom names
)

ssp = SpatialSemanticPointers(model, memory, config)
```

**Why FHRR only?**
- Fractional powers `X^x` require complex phase representation
- MAP and Binary models cannot support continuous exponentiation

### Step 2: Encode a Scene

Let's create a simple room layout:

```python
# Add objects to memory
memory.add_many(["table", "chair", "lamp", "plant"])

# Create scene with objects at specific locations
from vsax.spatial.utils import create_spatial_scene

room = create_spatial_scene(ssp, {
    "table": [2.5, 2.5],  # Center of room
    "chair": [2.0, 2.0],  # Near table
    "lamp": [0.5, 4.5],   # Corner
    "plant": [4.5, 0.5]   # Opposite corner
})

print(f"Room hypervector shape: {room.shape}")  # (512,)
```

**What just happened?**

```python
# create_spatial_scene does this:
table_here = ssp.bind_object_location("table", [2.5, 2.5])
chair_here = ssp.bind_object_location("chair", [2.0, 2.0])
lamp_here = ssp.bind_object_location("lamp", [0.5, 4.5])
plant_here = ssp.bind_object_location("plant", [4.5, 0.5])

# Bundle all object-location pairs
room = model.opset.bundle(
    table_here.vec,
    chair_here.vec,
    lamp_here.vec,
    plant_here.vec
)
```

**Result:** Single hypervector encoding entire spatial scene!

### Step 3: Query "What is at location?"

```python
from vsax.similarity import cosine_similarity

# What's at the center (2.5, 2.5)?
center_query = ssp.query_location(room, [2.5, 2.5])

# Check similarity to each object
for obj in ["table", "chair", "lamp", "plant"]:
    sim = cosine_similarity(center_query.vec, memory[obj].vec)
    print(f"{obj}: {sim:.3f}")

# Output:
# table: 0.892  ← Highest! Table is at (2.5, 2.5)
# chair: 0.612  (nearby but not exact)
# lamp: 0.203
# plant: 0.187
```

**How it works:**

```
Query = Scene ⊗ S(x, y)^(-1)
      ≈ Object
```

Unbinding the location reveals which object is there.

### Step 4: Query "Where is object?"

```python
# Where is the lamp?
lamp_location = ssp.query_object(room, "lamp")

# Decode the location (grid search)
coords = ssp.decode_location(
    lamp_location,
    search_range=[(0.0, 5.0), (0.0, 5.0)],  # x and y ranges
    resolution=50  # Grid density
)

print(f"Lamp at: ({coords[0]:.2f}, {coords[1]:.2f})")
# Output: Lamp at: (0.52, 4.48)  ← Close to true position (0.5, 4.5)!
```

**How it works:**

```
Location = Scene ⊗ Object^(-1)
         ≈ S(x, y)
```

Unbinding the object reveals its location.

**Note:** Decoding uses grid search, so it's approximate. Higher resolution = more accurate but slower.

---

## Practical Applications

### Application 1: Robot Navigation

Track objects in the environment:

```python
# Robot's world model
memory.add_many(["obstacle1", "obstacle2", "goal", "charging_station"])

world = create_spatial_scene(ssp, {
    "obstacle1": [3.0, 4.0],
    "obstacle2": [5.0, 2.0],
    "goal": [8.0, 8.0],
    "charging_station": [0.5, 0.5]
})

# Where do I need to go?
goal_loc = ssp.query_object(world, "goal")
coords = ssp.decode_location(goal_loc, [(0, 10), (0, 10)], resolution=30)
print(f"Navigate to: {coords}")  # [8.0, 8.0]

# What's at my current position (5.1, 2.1)?
here = ssp.query_location(world, [5.1, 2.1])

# Check what's nearby
for obj in ["obstacle1", "obstacle2", "goal", "charging_station"]:
    sim = cosine_similarity(here.vec, memory[obj].vec)
    if sim > 0.6:
        print(f"WARNING: {obj} nearby (similarity: {sim:.3f})")

# Output: WARNING: obstacle2 nearby (similarity: 0.847)
```

### Application 2: Geographic Information System

Encode points of interest with real coordinates:

```python
# NYC landmarks (latitude, longitude)
memory.add_many(["central_park", "empire_state", "times_square", "statue_of_liberty"])

# Use scaling for large coordinate ranges
config_geo = SSPConfig(
    dim=512,
    num_axes=2,
    scale=0.01,  # Scale lat/lon to reasonable range
    axis_names=["latitude", "longitude"]
)

ssp_geo = SpatialSemanticPointers(model, memory, config_geo)

nyc_map = create_spatial_scene(ssp_geo, {
    "central_park": [40.7829, -73.9654],
    "empire_state": [40.7484, -73.9857],
    "times_square": [40.7580, -73.9855],
    "statue_of_liberty": [40.6892, -74.0445]
})

# What's near Times Square?
times_sq_query = ssp_geo.query_location(nyc_map, [40.7580, -73.9855])

for landmark in ["central_park", "empire_state", "times_square", "statue_of_liberty"]:
    sim = cosine_similarity(times_sq_query.vec, memory[landmark].vec)
    print(f"{landmark}: {sim:.3f}")

# Output shows times_square has highest similarity
```

### Application 3: 3D Scientific Data

Track particles in 3D space:

```python
# Configure for 3D
config_3d = SSPConfig(dim=1024, num_axes=3, axis_names=["x", "y", "z"])
ssp_3d = SpatialSemanticPointers(model, memory, config_3d)

# Add particle IDs
memory.add_many([f"particle_{i}" for i in range(5)])

# Encode particle positions
particle_system = create_spatial_scene(ssp_3d, {
    "particle_0": [1.2, 3.4, 5.6],
    "particle_1": [2.1, 4.3, 6.5],
    "particle_2": [3.0, 5.2, 7.4],
    "particle_3": [1.5, 3.8, 5.9],
    "particle_4": [8.0, 2.0, 1.0]
})

# Which particle is near (1.3, 3.5, 5.7)?
query = ssp_3d.query_location(particle_system, [1.3, 3.5, 5.7])

for i in range(5):
    sim = cosine_similarity(query.vec, memory[f"particle_{i}"].vec)
    print(f"particle_{i}: {sim:.3f}")

# particle_0 has highest similarity (closest position)
```

---

## Advanced Features

### Scene Shifting

Translate entire scene by an offset:

```python
# Original: apple at (3.5, 2.1)
scene = ssp.bind_object_location("apple", [3.5, 2.1])

# Move everything by (+1.0, -0.5)
shifted_scene = ssp.shift_scene(scene, [1.0, -0.5])

# Apple now at (4.5, 1.6)
new_loc = ssp.query_object(shifted_scene, "apple")
coords = ssp.decode_location(new_loc, [(0, 10), (0, 10)], resolution=40)
print(f"Apple moved to: {coords}")  # ≈ [4.5, 1.6]
```

**Use case:** Robot moves coordinate frame, or camera pans across scene.

### Similarity Heatmaps

Visualize spatial distributions:

```python
from vsax.spatial.utils import similarity_map_2d
import matplotlib.pyplot as plt

# Where is the apple?
scene = ssp.bind_object_location("apple", [3.5, 2.1])
apple_loc = ssp.query_object(scene, "apple")

# Generate heatmap
X, Y, similarities = similarity_map_2d(
    ssp,
    apple_loc,
    x_range=(0.0, 5.0),
    y_range=(0.0, 5.0),
    resolution=50
)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, similarities, levels=20, cmap='viridis')
plt.colorbar(label='Similarity')
plt.scatter([3.5], [2.1], color='red', s=100, label='True position')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Apple Location Heatmap')
plt.legend()
plt.show()
```

**Result:** Peak at `(3.5, 2.1)` shows where apple is located!

### Region Queries

Find all objects within a spatial region:

```python
from vsax.spatial.utils import region_query

scene = create_spatial_scene(ssp, {
    "apple": [1.0, 1.0],
    "banana": [3.0, 3.0],
    "cherry": [5.0, 5.0],
    "date": [3.1, 2.9]
})

# What's near (3.0, 3.0) within radius 0.5?
results = region_query(
    ssp, scene,
    object_names=["apple", "banana", "cherry", "date"],
    center=[3.0, 3.0],
    radius=0.5
)

print(results)
# {'banana': 0.89, 'date': 0.82, 'apple': 0.23, 'cherry': 0.18}
# Banana and date are within the region!
```

---

## SSP vs Other Techniques

### SSP vs Clifford Operators

SSP and Clifford Operators solve **different problems** - use both together!

| Feature | SSP | CliffordOperator |
|---------|-----|------------------|
| **Purpose** | Continuous spatial coordinates | Discrete symbolic relations |
| **Example** | "apple at (3.5, 2.1)" | "cup LEFT_OF plate" |
| **Encoding** | `X^x ⊗ Y^y` | Phase transformation |
| **Query** | "What/where?" | "What relation?" |
| **Precision** | Approximate (grid search) | Exact (>0.999 similarity) |

**Combining both:**

```python
from vsax.operators import CliffordOperator, OperatorKind

# Spatial operator for directional relations
LEFT_OF = CliffordOperator.random(512, kind=OperatorKind.SPATIAL)

# "Cup at (2.0, 3.0) is LEFT_OF plate"
cup_at_pos = ssp.bind_object_location("cup", [2.0, 3.0])
plate_relation = LEFT_OF.apply(memory["plate"])

# Combined scene: continuous location + symbolic relation
scene = model.opset.bundle(cup_at_pos.vec, plate_relation.vec)
```

### When to Use SSP

✅ **Use SSP for:**
- Continuous spatial coordinates (not discrete grid)
- Navigation and localization
- Geographic information systems
- Scientific data with spatial structure
- "What is at (x, y)?" queries

❌ **Use other approaches for:**
- Discrete grid positions → Direct encoding
- Symbolic spatial relations → CliffordOperator (Lesson 4.1)
- Pure attribute-value pairs → DictEncoder (Lesson 3.2)

---

## Performance Considerations

### Dimensionality

Higher dimensions → more capacity and accuracy:

| Dimensionality | Objects per Scene | Use Case |
|----------------|-------------------|----------|
| 512 | ~50 objects | Simple 2D scenes |
| 1024 | ~100 objects | Complex 2D or simple 3D |
| 2048 | ~200 objects | High-precision spatial reasoning |

### Decoding Resolution

Grid search trade-off between speed and accuracy:

| Resolution | Decode Time | Position Error |
|------------|-------------|----------------|
| 10 | Fast (~10ms) | ±0.5 |
| 30 | Medium (~50ms) | ±0.2 |
| 50 | Slow (~200ms) | ±0.05 |
| 100 | Very slow (~1s) | ±0.02 |

**Best practice:** Use coarse resolution for real-time queries, fine resolution for final positioning.

### Coordinate Scaling

For large coordinate ranges (e.g., GPS), use scaling:

```python
# Without scaling: lat/lon in [-90, 90] and [-180, 180]
# With scaling: map to [-0.9, 0.9] and [-1.8, 1.8]
config = SSPConfig(dim=512, num_axes=2, scale=0.01)
```

**Why?** FPE works best when values are in range `[-10, 10]` approximately.

---

## Common Pitfalls

### Problem 1: Using MAP or Binary Model

```python
# ❌ This will raise TypeError
model = create_map_model(512)
ssp = SpatialSemanticPointers(model, memory)
# Error: SSP requires ComplexHypervector (FHRR)
```

**Fix:** Always use `create_fhrr_model()` for SSP.

### Problem 2: Too Many Objects

```python
# ❌ Bundling 500 objects with dim=512
scene = create_spatial_scene(ssp, {f"obj_{i}": [i, i] for i in range(500)})
# Similarity scores will be very low (< 0.3)
```

**Fix:** Increase dimensionality or partition the scene:

```python
# ✅ Higher dimension
model = create_fhrr_model(dim=2048)

# ✅ Or partition space into regions
region_A = create_spatial_scene(ssp, {f"obj_{i}": [i, i] for i in range(50)})
region_B = create_spatial_scene(ssp, {f"obj_{i}": [i, i] for i in range(50, 100)})
```

### Problem 3: Large Coordinate Values

```python
# ❌ Very large coordinates
location = ssp.encode_location([10000, 50000])
# Poor similarity structure
```

**Fix:** Use coordinate scaling:

```python
# ✅ Scale down
config = SSPConfig(dim=512, num_axes=2, scale=0.0001)
ssp = SpatialSemanticPointers(model, memory, config)
location = ssp.encode_location([10000, 50000])  # Scaled to [1, 5]
```

---

## Self-Assessment

Before moving on, ensure you can:

- [ ] Explain why SSP is needed for continuous spatial coordinates
- [ ] Describe how SSP uses FPE to encode locations
- [ ] Create a 2D spatial scene with multiple objects
- [ ] Query "what is at (x, y)?" and interpret results
- [ ] Query "where is object O?" and decode coordinates
- [ ] Choose appropriate dimensionality and resolution for your application
- [ ] Understand when to use SSP vs CliffordOperator vs other encoders

## Quick Quiz

**Question 1:** Why does SSP require FHRR (ComplexHypervector)?

a) For better performance on GPU
b) Because fractional powers need complex phase representation
c) To save memory
d) It doesn't - SSP works with any model

<details>
<summary>Answer</summary>
**b) Because fractional powers need complex phase representation**

SSP uses `X^x` where x is a real number. This requires complex vectors where exponentiation is well-defined via phase manipulation. MAP and Binary models cannot support continuous fractional powers.
</details>

**Question 2:** You have a 2D map with 150 landmarks to encode. Which configuration is best?

a) `dim=256`, `resolution=100`
b) `dim=1024`, `resolution=30`
c) `dim=512`, `resolution=10`
d) `dim=2048`, `resolution=50`

<details>
<summary>Answer</summary>
**b) dim=1024, resolution=30**

With 150 objects, you need `dim >= 1024` for sufficient capacity. Resolution=30 provides good accuracy without being too slow. Option (d) would also work but is overkill and slower.
</details>

**Question 3:** What does `ssp.query_location(scene, [x, y])` return?

a) The exact object name at (x, y)
b) A hypervector similar to the object at (x, y)
c) The coordinates of the nearest object
d) A boolean indicating if (x, y) is occupied

<details>
<summary>Answer</summary>
**b) A hypervector similar to the object at (x, y)**

Querying a location unbinds the spatial encoding, returning a hypervector that you must compare to known object vectors using similarity metrics. It does not directly return the object name.
</details>

---

## Hands-On Exercise

**Task:** Build a 2D office navigation system.

**Requirements:**
1. Create SSP system with 2D space
2. Encode office layout with at least 6 objects at different locations
3. Query "what's at the printer location?"
4. Query "where is the coffee machine?"
5. Use region query to find all objects near the center

**Starter code:**

```python
import jax
from vsax import create_fhrr_model, VSAMemory
from vsax.spatial import SpatialSemanticPointers, SSPConfig
from vsax.spatial.utils import create_spatial_scene, region_query
from vsax.similarity import cosine_similarity

# Step 1: Create SSP system
model = create_fhrr_model(dim=1024, key=jax.random.PRNGKey(42))
memory = VSAMemory(model)
config = SSPConfig(dim=1024, num_axes=2)
ssp = SpatialSemanticPointers(model, memory, config)

# Step 2: Define office layout (10m x 10m room)
objects = {
    "desk": [2.0, 2.0],
    "coffee_machine": [0.5, 9.5],
    "printer": [9.5, 0.5],
    "whiteboard": [5.0, 0.5],
    "door": [0.0, 5.0],
    "bookshelf": [9.5, 9.5]
}

# Add objects to memory
memory.add_many(list(objects.keys()))

# Create office scene
office = create_spatial_scene(ssp, objects)

# Step 3: Query - what's at the printer?
# YOUR CODE HERE

# Step 4: Query - where is the coffee machine?
# YOUR CODE HERE

# Step 5: What's near the center (5.0, 5.0)?
# YOUR CODE HERE
```

<details>
<summary>Solution</summary>

```python
# Step 3: What's at the printer location?
printer_query = ssp.query_location(office, [9.5, 0.5])
print("\nWhat's at (9.5, 0.5)?")
for obj in objects.keys():
    sim = cosine_similarity(printer_query.vec, memory[obj].vec)
    print(f"  {obj}: {sim:.3f}")
# printer should have highest similarity

# Step 4: Where is the coffee machine?
coffee_loc = ssp.query_object(office, "coffee_machine")
coords = ssp.decode_location(
    coffee_loc,
    search_range=[(0, 10), (0, 10)],
    resolution=40
)
print(f"\nCoffee machine at: ({coords[0]:.2f}, {coords[1]:.2f})")
# Should be close to [0.5, 9.5]

# Step 5: What's near the center?
center_results = region_query(
    ssp, office,
    object_names=list(objects.keys()),
    center=[5.0, 5.0],
    radius=2.0
)
print("\nObjects near center (5.0, 5.0):")
for obj, sim in sorted(center_results.items(), key=lambda x: x[1], reverse=True):
    if sim > 0.3:  # Threshold
        print(f"  {obj}: {sim:.3f}")
# door and whiteboard should be nearby
```
</details>

---

## Key Takeaways

✓ **SSP enables continuous spatial encoding** - no need for discrete grids
✓ **Built on FractionalPowerEncoder** - `S(x, y) = X^x ⊗ Y^y`
✓ **Supports "what" and "where" queries** - unbind location or object
✓ **Requires FHRR model** - complex vectors needed for fractional powers
✓ **Smooth similarity** - nearby locations have high similarity
✓ **Scalable to any dimensionality** - 1D, 2D, 3D, or higher
✓ **Complementary to Clifford Operators** - continuous coords vs discrete relations

---

## Next Steps

**Next Lesson:** [Lesson 4.3 - Hierarchical Structures & Resonators](03_hierarchical.md)
Learn how to encode tree structures and use resonator networks for convergent factorization.

**For More Details:** [Spatial Semantic Pointers Guide](../../guide/spatial.md)
Comprehensive technical reference with advanced features, utilities, and examples.

**Related Content:**
- [Tutorial 11 - Analogical Reasoning with Conceptual Spaces](../../tutorials/11_analogical_reasoning.md)
- [Fractional Power Encoding Guide](../../guide/fpe.md)
- [SSP Examples](../../examples/spatial/)

## References

- Komer, B., Voelker, A. R., & Eliasmith, C. (2019). "A neural representation of continuous space using fractional binding." *Proceedings of the Annual Meeting of the Cognitive Science Society.*
- Plate, T. A. (1995). "Holographic Reduced Representations." *IEEE Transactions on Neural Networks.*
- Gayler, R. W. (2003). "Vector symbolic architectures answer Jackendoff's challenges for cognitive neuroscience." *Proceedings of ICCS/ASCS.*
