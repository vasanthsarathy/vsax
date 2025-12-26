# Lesson 5.2: Building Custom Encoders

**Estimated time:** 60 minutes

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand the AbstractEncoder interface and design principles
- Design custom encoders for domain-specific data types
- Implement encoders from scratch with proper testing
- Choose appropriate VSA operations (bind, bundle, permute) for your domain
- Validate encoder quality through similarity analysis
- Publish and share custom encoders with the community

## Prerequisites

- Module 3 (all encoder lessons)
- Understanding of binding and bundling operations
- Python object-oriented programming

---

## Why Build Custom Encoders?

VSAX provides built-in encoders for common data types:
- ScalarEncoder → numbers
- SequenceEncoder → ordered data
- DictEncoder → key-value pairs
- SetEncoder → unordered collections
- GraphEncoder → relational triples

**But what if your domain has unique structure?**

### Research Domains Needing Custom Encoders

| Domain | Data Type | Challenge |
|--------|-----------|-----------|
| **Computer Vision** | Images, 3D point clouds | Spatial relationships, local features |
| **Bioinformatics** | DNA sequences, protein structures | Long sequences, functional motifs |
| **Robotics** | Sensor streams, motor commands | Temporal patterns, multi-modal fusion |
| **NLP** | Parse trees, dependency graphs | Hierarchical syntax, semantic roles |
| **Chemistry** | Molecular graphs, reaction pathways | 3D geometry, bond types |
| **Finance** | Time series with events | Irregular sampling, regime changes |

**Goal:** Learn to build encoders that capture YOUR domain's structure.

---

## The AbstractEncoder Interface

All VSAX encoders inherit from `AbstractEncoder`:

```python
from vsax.encoders import AbstractEncoder

class AbstractEncoder:
    """
    Base class for all encoders.

    Encoders transform domain-specific data into hypervectors.
    """

    def __init__(self, model, memory):
        """
        Initialize encoder.

        Args:
            model: VSAModel instance (defines algebra)
            memory: VSAMemory instance (stores basis vectors)
        """
        self.model = model
        self.memory = memory

    def encode(self, data):
        """
        Encode data into hypervector.

        Args:
            data: Domain-specific data

        Returns:
            Hypervector (model.rep_cls instance)
        """
        raise NotImplementedError("Subclasses must implement encode()")
```

**Key methods to implement:**
1. `__init__(model, memory)` - initialization
2. `encode(data)` - main encoding logic

**Optional methods:**
3. `fit(data)` - learn parameters from data (if needed)
4. `decode(hypervector)` - reverse encoding (if possible)

---

## Design Principles

### Principle 1: Preserve Domain Structure

**Bad:** Ignores relationships
```python
# Encoding graph by bundling all nodes (loses edges!)
graph_hv = bundle(node_a, node_b, node_c)
```

**Good:** Encodes structure
```python
# Encoding graph with edge relationships
edge_ab = bind(node_a, bind(relation_"edge", node_b))
edge_bc = bind(node_b, bind(relation_"edge", node_c))
graph_hv = bundle(edge_ab, edge_bc)
```

### Principle 2: Choose Operations by Semantics

| Operation | When to Use | Example |
|-----------|-------------|---------|
| **Bind (⊗)** | Associating roles with fillers | `"color" ⊗ "red"` |
| **Bundle (⊕)** | Aggregating peers | `pixel_1 ⊕ pixel_2 ⊕ ... ⊕ pixel_n` |
| **Permute** | Positional encoding | `permute(item, shift=position)` |

### Principle 3: Normalize Appropriately

```python
# After bundling, normalize to unit length
bundled = bundle(hv1, hv2, hv3)
normalized = bundled / jnp.linalg.norm(bundled)
```

### Principle 4: Make Encoders Composable

```python
# Good: Encoder returns hypervector that can be further composed
class ImageEncoder(AbstractEncoder):
    def encode(self, image):
        # ... encoding logic ...
        return self.model.rep_cls(encoded_vec)  # Returns hypervector object

# Can be used in larger system
image_hv = image_encoder.encode(image)
caption_hv = text_encoder.encode(caption)
multimodal = model.opset.bundle(image_hv.vec, caption_hv.vec)
```

---

## Example 1: Building ImageEncoder

Let's build an encoder for small grayscale images (e.g., MNIST).

### Step 1: Design the Encoding Strategy

**Strategy:** Positional pixel bundling
```
image = ⊕ᵢⱼ (POSᵢⱼ ⊗ VALUEᵢⱼ)
```

Each pixel binds its position to its intensity, then bundle all pixels.

### Step 2: Implement the Class

```python
import jax.numpy as jnp
from vsax.encoders import AbstractEncoder, FractionalPowerEncoder

class ImageEncoder(AbstractEncoder):
    """
    Encode grayscale images using positional pixel bundling.

    Each pixel (i, j) with intensity v is encoded as:
        position_{i,j} ⊗ intensity^v

    All pixels are bundled together.
    """

    def __init__(self, model, memory, image_height, image_width, scale=1.0):
        """
        Initialize ImageEncoder.

        Args:
            model: VSA model
            memory: VSA memory
            image_height: Image height in pixels
            image_width: Image width in pixels
            scale: Scaling factor for intensities (default: 1.0 for [0, 1] range)
        """
        super().__init__(model, memory)

        self.height = image_height
        self.width = image_width

        # Create position basis vectors for each pixel
        self.positions = {}
        for i in range(image_height):
            for j in range(image_width):
                pos_name = f"pixel_{i}_{j}"
                self.memory.add(pos_name)
                self.positions[(i, j)] = self.memory[pos_name].vec

        # Create FPE for intensity encoding
        self.memory.add("intensity")
        self.fpe = FractionalPowerEncoder(model, memory, scale=scale)

    def encode(self, image):
        """
        Encode image.

        Args:
            image: 2D array of shape (height, width) with values in [0, 1]

        Returns:
            ComplexHypervector representing the image
        """
        if image.shape != (self.height, self.width):
            raise ValueError(f"Expected shape {(self.height, self.width)}, got {image.shape}")

        # Encode each pixel
        pixel_hvs = []

        for i in range(self.height):
            for j in range(self.width):
                intensity = image[i, j]

                # Only encode non-zero pixels (sparsity)
                if intensity > 0.01:
                    # Encode intensity using FPE
                    intensity_hv = self.fpe.encode("intensity", float(intensity))

                    # Bind position to intensity
                    pixel_hv = self.model.opset.bind(
                        self.positions[(i, j)],
                        intensity_hv.vec
                    )
                    pixel_hvs.append(pixel_hv)

        # Bundle all pixels
        if len(pixel_hvs) == 0:
            # Empty image → zero vector
            result = jnp.zeros(self.model.dim, dtype=jnp.complex64)
        else:
            result = self.model.opset.bundle(*pixel_hvs)

        # Normalize
        result = result / jnp.linalg.norm(result)

        return self.model.rep_cls(result)
```

### Step 3: Test the Encoder

```python
from vsax import create_fhrr_model, VSAMemory
import jax.numpy as jnp

# Create model
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Create encoder for 8x8 images
encoder = ImageEncoder(model, memory, image_height=8, image_width=8)

# Create test images
image_1 = jnp.zeros((8, 8))
image_1 = image_1.at[3:5, 3:5].set(1.0)  # Small square

image_2 = jnp.zeros((8, 8))
image_2 = image_2.at[3:5, 3:5].set(1.0)  # Same square

image_3 = jnp.zeros((8, 8))
image_3 = image_3.at[1:3, 1:3].set(1.0)  # Different position

# Encode
hv_1 = encoder.encode(image_1)
hv_2 = encoder.encode(image_2)
hv_3 = encoder.encode(image_3)

# Test similarity
from vsax.similarity import cosine_similarity

sim_12 = cosine_similarity(hv_1.vec, hv_2.vec)
sim_13 = cosine_similarity(hv_1.vec, hv_3.vec)

print(f"Same image: {sim_12:.3f}")  # Should be ~1.0
print(f"Different position: {sim_13:.3f}")  # Should be lower

# Expected output:
# Same image: 0.998
# Different position: 0.423
```

---

## Example 2: Building BioinformaticsEncoder

Encode DNA sequences with motif awareness.

### Design Strategy

DNA sequences have:
1. **Position matters:** "ACGT" ≠ "TGCA"
2. **Motifs:** Functional subsequences (e.g., "TATA" box)
3. **Variable length**

**Strategy:**
```
sequence = ⊕ᵢ (POSᵢ ⊗ BASEᵢ) ⊕ ⊕ⱼ (MOTIFⱼ ⊗ POS_startⱼ)
```

### Implementation

```python
class DNASequenceEncoder(AbstractEncoder):
    """
    Encode DNA sequences with position and motif awareness.
    """

    def __init__(self, model, memory, motifs=None):
        """
        Initialize DNA encoder.

        Args:
            model: VSA model
            memory: VSA memory
            motifs: List of important motifs (e.g., ["TATA", "CAAT"])
        """
        super().__init__(model, memory)

        # Add basis vectors for bases
        self.memory.add_many(["A", "C", "G", "T"])

        # Add motifs
        self.motifs = motifs or []
        for motif in self.motifs:
            self.memory.add(f"motif_{motif}")

    def encode(self, sequence):
        """
        Encode DNA sequence.

        Args:
            sequence: String like "ACGTACGT"

        Returns:
            Hypervector representing the sequence
        """
        components = []

        # Component 1: Positional encoding of each base
        for i, base in enumerate(sequence):
            if base in self.memory:
                # Permute by position
                pos_encoding = self.model.opset.permute(
                    self.memory[base].vec,
                    shift=i
                )
                components.append(pos_encoding)

        # Component 2: Motif detection
        for motif in self.motifs:
            # Find motif occurrences
            start_pos = 0
            while True:
                pos = sequence.find(motif, start_pos)
                if pos == -1:
                    break

                # Encode motif at position
                motif_hv = self.model.opset.permute(
                    self.memory[f"motif_{motif}"].vec,
                    shift=pos
                )
                components.append(motif_hv)
                start_pos = pos + 1

        # Bundle all components
        if len(components) == 0:
            result = jnp.zeros(self.model.dim, dtype=jnp.complex64)
        else:
            result = self.model.opset.bundle(*components)

        # Normalize
        result = result / jnp.linalg.norm(result)
        return self.model.rep_cls(result)
```

### Testing

```python
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Create encoder with TATA motif
encoder = DNASequenceEncoder(model, memory, motifs=["TATA"])

# Encode sequences
seq1 = "TATAACGTACGT"  # Contains TATA
seq2 = "TATAACGTACGT"  # Same
seq3 = "ACGTACGTACGT"  # No TATA
seq4 = "ACGTTATAACGT"  # TATA at different position

hv1 = encoder.encode(seq1)
hv2 = encoder.encode(seq2)
hv3 = encoder.encode(seq3)
hv4 = encoder.encode(seq4)

print(f"Same sequence: {cosine_similarity(hv1.vec, hv2.vec):.3f}")  # ~1.0
print(f"Without TATA: {cosine_similarity(hv1.vec, hv3.vec):.3f}")  # Lower
print(f"TATA different pos: {cosine_similarity(hv1.vec, hv4.vec):.3f}")  # Medium
```

---

## Example 3: Building SensorFusionEncoder

Combine heterogeneous sensor readings.

### Design Strategy

Robot sensors:
- **Lidar:** distance measurements (continuous)
- **Camera:** RGB values (continuous)
- **IMU:** acceleration (continuous, 3D)
- **Compass:** heading (periodic)

**Strategy:** Role-filler binding with modality roles

```
sensors = LIDAR ⊗ dist + CAMERA ⊗ rgb + IMU ⊗ accel + COMPASS ⊗ heading
```

### Implementation

```python
from vsax.encoders import ScalarEncoder

class SensorFusionEncoder(AbstractEncoder):
    """
    Fuse multiple sensor modalities into single hypervector.
    """

    def __init__(self, model, memory):
        super().__init__(model, memory)

        # Add modality roles
        self.memory.add_many(["LIDAR", "CAMERA", "IMU", "COMPASS"])

        # Scalar encoder for continuous values
        self.scalar_encoder = ScalarEncoder(model, memory)

    def encode(self, sensor_dict):
        """
        Encode sensor readings.

        Args:
            sensor_dict: Dict like {
                "lidar": 2.5,  # distance in meters
                "camera": [0.3, 0.6, 0.2],  # RGB
                "imu": [0.1, -0.2, 9.8],  # acceleration
                "compass": 45.0  # degrees
            }

        Returns:
            Fused sensor hypervector
        """
        components = []

        # Encode LIDAR
        if "lidar" in sensor_dict:
            self.memory.add("lidar_dist")
            lidar_hv = self.scalar_encoder.encode("lidar_dist", sensor_dict["lidar"])
            lidar_component = self.model.opset.bind(
                self.memory["LIDAR"].vec,
                lidar_hv.vec
            )
            components.append(lidar_component)

        # Encode CAMERA (bundle RGB channels)
        if "camera" in sensor_dict:
            rgb = sensor_dict["camera"]
            for i, channel in enumerate(["R", "G", "B"]):
                self.memory.add(f"camera_{channel}")
                channel_hv = self.scalar_encoder.encode(f"camera_{channel}", rgb[i])
                components.append(
                    self.model.opset.bind(self.memory["CAMERA"].vec, channel_hv.vec)
                )

        # Encode IMU (3D acceleration)
        if "imu" in sensor_dict:
            accel = sensor_dict["imu"]
            for i, axis in enumerate(["X", "Y", "Z"]):
                self.memory.add(f"accel_{axis}")
                axis_hv = self.scalar_encoder.encode(f"accel_{axis}", accel[i])
                components.append(
                    self.model.opset.bind(self.memory["IMU"].vec, axis_hv.vec)
                )

        # Encode COMPASS
        if "compass" in sensor_dict:
            self.memory.add("heading")
            # Use modulo for periodic encoding
            heading_hv = self.scalar_encoder.encode("heading", sensor_dict["compass"] % 360)
            compass_component = self.model.opset.bind(
                self.memory["COMPASS"].vec,
                heading_hv.vec
            )
            components.append(compass_component)

        # Bundle all sensors
        result = self.model.opset.bundle(*components)
        result = result / jnp.linalg.norm(result)

        return self.model.rep_cls(result)
```

---

## Encoder Validation

### Test 1: Self-Similarity

Encoding same data twice should give similarity ≈ 1.0:

```python
data = <some data>
hv1 = encoder.encode(data)
hv2 = encoder.encode(data)

sim = cosine_similarity(hv1.vec, hv2.vec)
assert sim > 0.99, f"Self-similarity too low: {sim}"
```

### Test 2: Different Data → Low Similarity

Different data should produce dissimilar hypervectors:

```python
data1 = <data 1>
data2 = <very different data>

hv1 = encoder.encode(data1)
hv2 = encoder.encode(data2)

sim = cosine_similarity(hv1.vec, hv2.vec)
assert sim < 0.3, f"Dissimilarity too low: {sim}"
```

### Test 3: Gradual Change → Gradual Similarity

Small changes should result in high but not perfect similarity:

```python
data_original = <original>
data_modified = <slightly modified>

hv1 = encoder.encode(data_original)
hv2 = encoder.encode(data_modified)

sim = cosine_similarity(hv1.vec, hv2.vec)
assert 0.6 < sim < 0.95, f"Similarity not in expected range: {sim}"
```

### Test 4: Composition Preserves Information

If encoder is invertible, test reconstruction:

```python
data = <original>
hv = encoder.encode(data)
reconstructed = encoder.decode(hv)  # If decode() is implemented

error = compute_error(data, reconstructed)
assert error < threshold
```

---

## Best Practices

### 1. Document Your Encoder

```python
class MyCustomEncoder(AbstractEncoder):
    """
    One-line description.

    Detailed explanation of:
    - What data types it handles
    - Encoding strategy (binding/bundling pattern)
    - Domain assumptions
    - Expected dimensionality

    Example:
        >>> encoder = MyCustomEncoder(model, memory)
        >>> hv = encoder.encode(my_data)

    References:
        - Paper citation if based on published work
    """
    pass
```

### 2. Provide Usage Examples

Include a `examples/` directory with:
- Minimal working example
- Real-world use case
- Visualization of encoded patterns

### 3. Add Type Hints

```python
from typing import Dict, List
import jax.numpy as jnp

def encode(self, data: Dict[str, float]) -> ComplexHypervector:
    """
    Args:
        data: Sensor readings as dict

    Returns:
        Encoded hypervector
    """
    pass
```

### 4. Handle Edge Cases

```python
def encode(self, data):
    # Check for empty input
    if data is None or len(data) == 0:
        return self.model.rep_cls(jnp.zeros(self.model.dim, dtype=jnp.complex64))

    # Check for invalid dimensions
    if data.shape[0] != self.expected_size:
        raise ValueError(f"Expected size {self.expected_size}, got {data.shape[0]}")

    # ... normal encoding ...
```

---

## Publishing Your Encoder

### Step 1: Package Structure

```
my_custom_encoder/
├── __init__.py
├── encoder.py          # Your encoder class
├── tests/
│   └── test_encoder.py
├── examples/
│   └── example_usage.py
├── README.md
└── requirements.txt
```

### Step 2: Write Tests

```python
import pytest
from vsax import create_fhrr_model, VSAMemory
from my_custom_encoder import MyEncoder

def test_encode_basic():
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    encoder = MyEncoder(model, memory)

    data = <test data>
    hv = encoder.encode(data)

    assert hv.shape == (512,)
    assert jnp.isfinite(hv).all()

def test_self_similarity():
    # ... test self-similarity ...
    pass
```

### Step 3: Document in README

```markdown
# MyCustomEncoder

Encodes [data type] for VSA using [strategy].

## Installation

```bash
pip install my-custom-encoder
```

## Usage

```python
from vsax import create_fhrr_model, VSAMemory
from my_custom_encoder import MyEncoder

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)
encoder = MyEncoder(model, memory)

hv = encoder.encode(my_data)
```

## Citation

If you use this encoder, please cite:
...
```

### Step 4: Contribute to VSAX

Consider submitting your encoder as a pull request to VSAX if it's generally useful!

---

## Self-Assessment

Before moving on, ensure you can:

- [ ] Explain the AbstractEncoder interface
- [ ] Design an encoding strategy for a new data type
- [ ] Implement a custom encoder from scratch
- [ ] Choose bind, bundle, or permute appropriately
- [ ] Test encoder quality through similarity analysis
- [ ] Document and package your encoder for sharing

## Quick Quiz

**Question 1:** When should you use binding (⊗) vs bundling (⊕)?

a) Binding for aggregation, bundling for associations
b) Binding for associations, bundling for aggregation
c) They're interchangeable
d) Always use bundling

<details>
<summary>Answer</summary>
**b) Binding for associations, bundling for aggregation**

Binding (⊗) creates associations between roles and fillers (e.g., `"color" ⊗ "red"`). Bundling (⊕) aggregates peer elements (e.g., `pixel_1 ⊕ pixel_2 ⊕ ... ⊕ pixel_n`). Using them correctly preserves semantic structure.
</details>

**Question 2:** Why normalize after bundling?

a) To make vectors smaller
b) To maintain unit length for consistent similarity comparisons
c) It's optional
d) To speed up computation

<details>
<summary>Answer</summary>
**b) To maintain unit length for consistent similarity comparisons**

Bundling many vectors increases the norm. Normalizing to unit length ensures cosine similarity remains in [-1, 1] and is comparable across different encoded data.
</details>

**Question 3:** What's the minimum your encoder must implement?

a) Only `encode()`
b) Both `encode()` and `decode()`
c) `encode()`, `decode()`, and `fit()`
d) Just inherit from AbstractEncoder

<details>
<summary>Answer</summary>
**a) Only `encode()`**

The minimum requirement is implementing `encode(data)` which returns a hypervector. `decode()` and `fit()` are optional and depend on whether your encoding is invertible or requires learning.
</details>

---

## Hands-On Exercise

**Task:** Build a custom `MusicNoteEncoder` for encoding musical notes.

**Requirements:**
1. Encode note pitch (C, D, E, F, G, A, B)
2. Encode octave (1-8)
3. Encode duration (quarter, half, whole note)
4. Test that C4 (middle C) quarter note has high self-similarity
5. Test that C4 and G4 (perfect fifth) have moderate similarity

**Starter code:**

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.encoders import AbstractEncoder

class MusicNoteEncoder(AbstractEncoder):
    """
    Encode musical notes with pitch, octave, and duration.
    """

    def __init__(self, model, memory):
        # YOUR CODE HERE
        pass

    def encode(self, note_dict):
        """
        Args:
            note_dict: {"pitch": "C", "octave": 4, "duration": "quarter"}
        """
        # YOUR CODE HERE
        pass


# Test your encoder
model = create_fhrr_model(dim=512)
memory = VSAMemory(model)
encoder = MusicNoteEncoder(model, memory)

c4_quarter = {"pitch": "C", "octave": 4, "duration": "quarter"}
g4_quarter = {"pitch": "G", "octave": 4, "duration": "quarter"}

# Encode and test similarities
```

<details>
<summary>Solution</summary>

```python
class MusicNoteEncoder(AbstractEncoder):
    def __init__(self, model, memory):
        super().__init__(model, memory)

        # Add basis vectors
        pitches = ["C", "D", "E", "F", "G", "A", "B"]
        self.memory.add_many(pitches)

        octaves = [f"octave_{i}" for i in range(1, 9)]
        self.memory.add_many(octaves)

        durations = ["quarter", "half", "whole"]
        self.memory.add_many(durations)

        # Roles
        self.memory.add_many(["pitch_role", "octave_role", "duration_role"])

    def encode(self, note_dict):
        pitch = note_dict["pitch"]
        octave = note_dict["octave"]
        duration = note_dict["duration"]

        # Bind each attribute to its role
        pitch_hv = self.model.opset.bind(
            self.memory["pitch_role"].vec,
            self.memory[pitch].vec
        )

        octave_hv = self.model.opset.bind(
            self.memory["octave_role"].vec,
            self.memory[f"octave_{octave}"].vec
        )

        duration_hv = self.model.opset.bind(
            self.memory["duration_role"].vec,
            self.memory[duration].vec
        )

        # Bundle all attributes
        result = self.model.opset.bundle(pitch_hv, octave_hv, duration_hv)
        result = result / jnp.linalg.norm(result)

        return self.model.rep_cls(result)


# Test
model = create_fhrr_model(dim=512)
memory = VSAMemory(model)
encoder = MusicNoteEncoder(model, memory)

c4_quarter = {"pitch": "C", "octave": 4, "duration": "quarter"}
c4_quarter_2 = {"pitch": "C", "octave": 4, "duration": "quarter"}
g4_quarter = {"pitch": "G", "octave": 4, "duration": "quarter"}
c5_quarter = {"pitch": "C", "octave": 5, "duration": "quarter"}

hv_c4_1 = encoder.encode(c4_quarter)
hv_c4_2 = encoder.encode(c4_quarter_2)
hv_g4 = encoder.encode(g4_quarter)
hv_c5 = encoder.encode(c5_quarter)

from vsax.similarity import cosine_similarity

print(f"Self-similarity: {cosine_similarity(hv_c4_1.vec, hv_c4_2.vec):.3f}")  # ~1.0
print(f"C4 vs G4 (different pitch): {cosine_similarity(hv_c4_1.vec, hv_g4.vec):.3f}")  # ~0.6
print(f"C4 vs C5 (different octave): {cosine_similarity(hv_c4_1.vec, hv_c5.vec):.3f}")  # ~0.6
```
</details>

---

## Key Takeaways

✓ **AbstractEncoder provides common interface** - `__init__(model, memory)` and `encode(data)`
✓ **Design encodes domain structure** - choose bind/bundle/permute by semantics
✓ **Normalize after bundling** - maintain unit length for consistent similarity
✓ **Test thoroughly** - self-similarity, dissimilarity, gradual change
✓ **Document and share** - help the VSA community grow
✓ **Composability is key** - encoders should produce hypervectors that can be further combined

---

## Next Steps

**Next Lesson:** [Lesson 5.3 - Research Frontiers & Open Problems](03_frontiers.md)
Explore current research directions, open problems, and how to contribute to VSA research.

**Related Content:**
- [Module 3 - All Encoder Lessons](../../course/03_encoders/index.md)
- [AbstractEncoder API Reference](../../api/encoders/abstract.md)
- [Contributing to VSAX](../../CONTRIBUTING.md)

## References

- Kleyko, D., et al. (2022). "Vector Symbolic Architectures as a Computing Framework for Nanoscale Hardware." *Proceedings of the IEEE.*
- Frady, E. P., Kleyko, D., & Sommer, F. T. (2021). "Variable Binding for Sparse Distributed Representations: Theory and Applications." *IEEE TNNLS.*
- Plate, T. A. (1995). "Holographic Reduced Representations." *IEEE Transactions on Neural Networks.*
