# I/O API Reference

JSON-based persistence for VSA basis vectors.

## Overview

The I/O module provides two functions for saving and loading basis vectors:

- **`save_basis()`** - Save VSAMemory to JSON file
- **`load_basis()`** - Load VSAMemory from JSON file

Both functions work with all three VSA models (FHRR, MAP, Binary).

## Functions

### save_basis

::: vsax.io.save_basis
    options:
      show_root_heading: true
      show_source: true

**Example:**

```python
from vsax import create_fhrr_model, VSAMemory, save_basis

model = create_fhrr_model(dim=512)
memory = VSAMemory(model)
memory.add_many(["dog", "cat", "animal"])

save_basis(memory, "animals.json")
```

### load_basis

::: vsax.io.load_basis
    options:
      show_root_heading: true
      show_source: true

**Example:**

```python
from vsax import create_fhrr_model, VSAMemory, load_basis

model = create_fhrr_model(dim=512)
memory = VSAMemory(model)

load_basis(memory, "animals.json")
print(f"Loaded {len(memory._vectors)} vectors")
```

## JSON Format

### FHRR (Complex Vectors)

Complex hypervectors are stored with separate real and imaginary parts:

```json
{
  "metadata": {
    "dim": 512,
    "rep_type": "complex",
    "num_vectors": 2
  },
  "vectors": {
    "dog": {
      "real": [0.12, -0.34, 0.56, ...],
      "imag": [0.78, -0.23, 0.45, ...]
    },
    "cat": {
      "real": [-0.67, 0.89, -0.12, ...],
      "imag": [0.34, 0.56, -0.78, ...]
    }
  }
}
```

### MAP (Real Vectors)

Real hypervectors are stored as simple float arrays:

```json
{
  "metadata": {
    "dim": 512,
    "rep_type": "real",
    "num_vectors": 2
  },
  "vectors": {
    "red": [0.23, -0.45, 0.67, ...],
    "blue": [-0.12, 0.34, -0.56, ...]
  }
}
```

### Binary (Bipolar Vectors)

Binary hypervectors are stored as integer arrays:

```json
{
  "metadata": {
    "dim": 1000,
    "rep_type": "binary",
    "num_vectors": 2
  },
  "vectors": {
    "x": [-1, 1, -1, 1, -1, ...],
    "y": [1, -1, 1, 1, -1, ...]
  }
}
```

## Error Handling

### Dimension Mismatch

```python
from vsax import create_fhrr_model, VSAMemory, save_basis, load_basis

# Save with dim=128
model_128 = create_fhrr_model(dim=128)
memory_128 = VSAMemory(model_128)
memory_128.add("test")
save_basis(memory_128, "test.json")

# Try to load with dim=256
model_256 = create_fhrr_model(dim=256)
memory_256 = VSAMemory(model_256)

try:
    load_basis(memory_256, "test.json")
except ValueError as e:
    print(e)  # Dimension mismatch: memory has dim=256, but file has dim=128
```

### Representation Type Mismatch

```python
from vsax import create_fhrr_model, create_map_model

# Save FHRR
fhrr_memory = VSAMemory(create_fhrr_model(dim=128))
fhrr_memory.add("test")
save_basis(fhrr_memory, "test.json")

# Try to load as MAP
map_memory = VSAMemory(create_map_model(dim=128))

try:
    load_basis(map_memory, "test.json")
except ValueError as e:
    print(e)  # Representation type mismatch: memory expects real, but file has complex
```

### Non-Empty Memory

```python
memory = VSAMemory(create_fhrr_model(dim=128))
memory.add("existing_vector")

try:
    load_basis(memory, "test.json")
except ValueError as e:
    print(e)  # Memory must be empty to load basis. Current memory contains 1 vectors.
```

## Use Cases

### Persistent Semantic Spaces

```python
# Build once
model = create_fhrr_model(dim=1024)
memory = VSAMemory(model)
memory.add_many(["concept1", "concept2", ...])
save_basis(memory, "knowledge_base.json")

# Reuse across sessions
memory_new = VSAMemory(model)
load_basis(memory_new, "knowledge_base.json")
```

### Sharing Vocabularies

```python
# Project A
save_basis(memory_a, "shared_vocab.json")

# Project B (same dimension and model type!)
load_basis(memory_b, "shared_vocab.json")
```

### Reproducible Research

```bash
# Version control basis vectors
git add experiment_basis.json
git commit -m "Add basis for reproducibility"
```

## See Also

- [Persistence User Guide](../../guide/persistence.md) - Detailed usage guide
- [VSAMemory API](../core/memory.md) - Memory management
- [Examples](../../../examples/persistence.py) - Complete working examples
