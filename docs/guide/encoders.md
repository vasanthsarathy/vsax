# Encoders

VSAX provides 5 core encoders for converting structured data into hypervectors, plus an extensible base class for creating custom encoders.

## Overview

Encoders transform structured data (numbers, sequences, dictionaries, graphs) into hypervector representations that can be manipulated with VSA operations.

All encoders:
- Work with all 3 VSA models (FHRR, MAP, Binary)
- Accept a `VSAModel` and `VSAMemory` in their constructor
- Implement an `encode()` method that returns a hypervector

## Core Encoders

### ScalarEncoder

Encodes numeric values using power encoding (for complex hypervectors) or iterated binding (for real/binary).

```python
from vsax import create_fhrr_model, VSAMemory, ScalarEncoder

model = create_fhrr_model(dim=512)
memory = VSAMemory(model)
memory.add("temperature")

encoder = ScalarEncoder(model, memory, min_val=0, max_val=100)
temp_hv = encoder.encode("temperature", 23.5)
```

**Use cases:** Sensor readings, measurements, ratings, scores

### SequenceEncoder

Encodes ordered sequences (lists, tuples) using positional binding.

```python
from vsax import SequenceEncoder

memory.add_many(["red", "green", "blue"])
encoder = SequenceEncoder(model, memory)

# Order matters!
seq1 = encoder.encode(["red", "green", "blue"])
seq2 = encoder.encode(["blue", "green", "red"])  # Different hypervector
```

**Use cases:** Time series, sentences, ordered lists, paths

### SetEncoder

Encodes unordered collections using bundling (order-invariant).

```python
from vsax import SetEncoder

memory.add_many(["dog", "cat", "bird"])
encoder = SetEncoder(model, memory)

# Order doesn't matter!
set1 = encoder.encode({"dog", "cat", "bird"})
set2 = encoder.encode({"bird", "dog", "cat"})  # Same hypervector
```

**Use cases:** Tags, categories, unordered groups

### DictEncoder

Encodes key-value pairs using role-filler binding.

```python
from vsax import DictEncoder

memory.add_many(["subject", "action", "dog", "run"])
encoder = DictEncoder(model, memory)

sentence = encoder.encode({
    "subject": "dog",
    "action": "run"
})
```

**Use cases:** Structured records, semantic frames, property-value pairs

### GraphEncoder

Encodes graph structures as edge lists.

```python
from vsax import GraphEncoder

memory.add_many(["Alice", "Bob", "knows", "likes"])
encoder = GraphEncoder(model, memory)

social_graph = encoder.encode([
    ("Alice", "knows", "Bob"),
    ("Alice", "likes", "Bob")
])
```

**Use cases:** Knowledge graphs, social networks, dependency graphs

## Custom Encoders

Create custom encoders by subclassing `AbstractEncoder`:

```python
from vsax import AbstractEncoder

class DateEncoder(AbstractEncoder):
    def encode(self, date_obj):
        # Your custom encoding logic
        year_hv = self.encode_component(date_obj.year)
        month_hv = self.encode_component(date_obj.month)
        day_hv = self.encode_component(date_obj.day)

        result = self.model.opset.bundle(year_hv, month_hv, day_hv)
        return self.model.rep_cls(result)
```

See [examples/custom_encoder.py](https://github.com/vasanthsarathy/vsax/blob/main/examples/custom_encoder.py) for complete examples.

## Best Practices

1. **Add symbols first**: Ensure all required symbols are in memory before encoding
2. **Consistent dimensions**: Use the same model for all related encodings
3. **Combine encoders**: Use multiple encoders together for complex data structures
4. **Test with similarity**: Verify encodings make sense by checking similarity between related items

## See Also

- [API Reference - Encoders](../api/encoders/index.md)
- [Examples - FHRR](../examples/fhrr.md)
- [Examples - Custom Encoders](https://github.com/vasanthsarathy/vsax/blob/main/examples/custom_encoder.py)
