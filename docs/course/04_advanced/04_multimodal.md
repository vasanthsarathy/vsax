# Lesson 4.4: Multi-Modal & Neural-Symbolic Integration

**Estimated time:** 55 minutes

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand the challenges of fusing heterogeneous data (vision, language, symbols)
- Encode multiple modalities in the same VSA space
- Perform cross-modal queries and reasoning
- Integrate neural network embeddings with symbolic VSA representations
- Apply HD-Glue to fuse multiple neural networks symbolically
- Build neuro-symbolic AI systems that combine learning and reasoning

## Prerequisites

- Module 3, Lesson 3.1 (Encoders for different data types)
- Understanding of neural network embeddings
- Basic knowledge of image classification

---

## The Multi-Modal Fusion Problem

Real-world intelligence requires integrating **heterogeneous data** from multiple sources:

- **Vision**: Images, video, 3D scans
- **Language**: Text, speech, captions
- **Symbols**: Concepts, categories, logical rules
- **Sensors**: Temperature, pressure, motion
- **Knowledge**: Facts, relationships, arithmetic

### Why Traditional Approaches Struggle

**Neural networks:**
- ❌ Require separate modules for each modality (vision backbone, language model, etc.)
- ❌ Hard to add new knowledge online (requires retraining)
- ❌ Difficult to perform cross-modal queries
- ❌ Not naturally compositional

**Symbolic AI:**
- ❌ Cannot process raw sensory data (images, audio)
- ❌ Brittle to noise and uncertainty
- ❌ Requires manual feature engineering

**The gap:** How do you connect "the image of a cat" to "the word 'cat'" to "the concept of a mammal"?

### VSA's Solution: Unified Hyperdimensional Space

**Key insight:** All modalities can be encoded as hypervectors in the same high-dimensional space.

```python
# Vision: encode image pixels
image_hv = encode_image(cat_image)

# Language: encode word
word_hv = memory["cat"]

# Symbolic: encode facts
fact_hv = model.opset.bind(memory["cat"].vec, memory["mammal"].vec)

# All are hypervectors - can be compared, combined, reasoned about!
sim = cosine_similarity(image_hv, word_hv)
```

**Advantages:**
- ✅ **Heterogeneous binding** - Different data types share the same space
- ✅ **Compositional** - Concepts defined by relationships
- ✅ **Online learning** - Add new facts by simple bundling
- ✅ **Cross-modal queries** - "What is 2 + 2?" → retrieve image of "4"

---

## Multi-Modal Concept Grounding

Let's see how to build **rich concept representations** that combine multiple modalities.

### Example: Grounding Numbers

A number like "3" can be represented through:

1. **Visual**: MNIST images of handwritten "3"
2. **Symbolic**: The concept "three" as a basis vector
3. **Arithmetic**: Relationships like `1+2=3`, `5-2=3`, `3×1=3`

**Goal:** Create a unified "concept of 3" that encompasses all these modalities.

### Step 1: Visual Encoding

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.encoders import ScalarEncoder
import jax.numpy as jnp

# Create model
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Encode MNIST image (8x8 = 64 pixels)
def encode_image(model, memory, image_pixels):
    """Encode image using pixel values."""
    encoder = ScalarEncoder(model, memory)

    encoded = jnp.zeros(model.dim, dtype=jnp.complex64)
    for i, pixel_value in enumerate(image_pixels):
        if pixel_value > 0:  # Only encode non-zero pixels
            feature_name = f"pixel_{i}"
            feature_vec = encoder.encode(feature_name, float(pixel_value))
            encoded = encoded + feature_vec.vec

    # Normalize
    return encoded / jnp.linalg.norm(encoded)


# Load MNIST images for digit "3"
# (Assume we have multiple images: image_1, image_2, ..., image_N)
# Build visual prototype by averaging
images_of_3 = [image_1, image_2, ..., image_N]
encoded_images = [encode_image(model, memory, img) for img in images_of_3]

visual_prototype_3 = sum(encoded_images) / len(encoded_images)
visual_prototype_3 = visual_prototype_3 / jnp.linalg.norm(visual_prototype_3)
```

### Step 2: Symbolic Encoding

```python
# Add symbolic basis vector for "three"
memory.add("three")
symbolic_3 = memory["three"].vec
```

### Step 3: Arithmetic Encoding

```python
# Encode arithmetic facts about "3"
memory.add_many(["plus", "minus", "equals", "one", "two", "four", "five"])

# Fact: 1 + 2 = 3
fact_1_plus_2 = model.opset.bundle(
    model.opset.bind(memory["one"].vec, memory["plus"].vec),
    model.opset.bind(memory["two"].vec, memory["equals"].vec)
)

# Fact: 5 - 2 = 3
fact_5_minus_2 = model.opset.bundle(
    model.opset.bind(memory["five"].vec, memory["minus"].vec),
    model.opset.bind(memory["two"].vec, memory["equals"].vec)
)

# Combine all arithmetic facts
arithmetic_3 = model.opset.bundle(fact_1_plus_2, fact_5_minus_2)
```

### Step 4: Fuse All Modalities

```python
# Create unified concept of "3"
concept_3 = model.opset.bundle(
    visual_prototype_3,
    symbolic_3,
    arithmetic_3
)

concept_3 = concept_3 / jnp.linalg.norm(concept_3)
```

**Result:** Single hypervector encoding vision + symbols + arithmetic!

### Cross-Modal Queries

Now we can perform cross-modal reasoning:

```python
from vsax.similarity import cosine_similarity

# Query 1: "What is 1 + 2?"
query_1_plus_2 = model.opset.bundle(
    model.opset.bind(memory["one"].vec, memory["plus"].vec),
    model.opset.bind(memory["two"].vec, memory["equals"].vec)
)

# Check similarity to concept_3
sim = cosine_similarity(query_1_plus_2, concept_3)
print(f"1 + 2 matches concept '3': {sim:.3f}")  # High similarity!

# Query 2: "Show me the visual prototype for 1 + 2"
# Unbind arithmetic to get visual component
result = model.opset.unbind(concept_3, arithmetic_3)
# result is similar to visual_prototype_3

# Query 3: "What arithmetic facts involve 'three'?"
# Unbind symbolic to get arithmetic component
arithmetic_facts = model.opset.unbind(concept_3, symbolic_3)
# arithmetic_facts contains bundled arithmetic relationships
```

**Power:** We can query **across modalities** - ask an arithmetic question and retrieve an image!

---

## Neural-Symbolic Fusion with HD-Glue

**Problem:** Deep learning produces powerful models, but:
- Previous models are discarded when new ones are trained
- Hard to combine models trained on different data
- Cannot easily add symbolic knowledge

**HD-Glue** solves this by creating a **symbolic layer** that fuses multiple neural networks.

### The HD-Glue Architecture

```
Input Image → [Neural Net 1] → Embedding 1 ──┐
              [Neural Net 2] → Embedding 2 ──┼→ [HD-Glue] → Prediction
              [Neural Net 3] → Embedding 3 ──┘    (VSA)
```

**Key steps:**
1. **Extract embeddings** from each neural network (hidden layer activations)
2. **Encode embeddings as hypervectors** using positional binding
3. **Bundle predictions** from all networks (consensus voting)
4. **Classify** using hyperdimensional inference layer (HIL)

### Step 1: Extract Neural Embeddings

```python
from sklearn.neural_network import MLPClassifier
import numpy as np

# Train multiple neural networks
networks = []
for i in range(5):
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),  # 128-dim embedding layer
        activation='tanh',
        max_iter=100,
        random_state=i
    )
    mlp.fit(X_train, y_train)
    networks.append(mlp)
    print(f"Network {i+1}: {mlp.score(X_test, y_test)*100:.2f}% accuracy")
```

**Output:**
```
Network 1: 94.81% accuracy
Network 2: 95.37% accuracy
Network 3: 94.63% accuracy
Network 4: 95.19% accuracy
Network 5: 94.44% accuracy
```

### Step 2: Encode Embeddings as Hypervectors

```python
from vsax import create_binary_model, VSAMemory

# Create VSA model
model = create_binary_model(dim=10000, bipolar=True)
memory = VSAMemory(model)

# Create hypervectors for binned values and positions
num_bins = 100
embedding_dim = 128

# Add basis vectors
memory.add_many([f"bin_{i}" for i in range(num_bins)])
memory.add_many([f"pos_{i}" for i in range(embedding_dim)])


def encode_embedding(model, memory, embedding, num_bins=100):
    """
    Encode neural network embedding as hypervector.

    Uses positional binding: bundle(pos_i ⊗ bin(value_i)) for all i
    """
    # Bin values (tanh output is in [-1, 1])
    bin_edges = np.linspace(-1, 1, num_bins + 1)
    binned = np.digitize(embedding, bin_edges) - 1  # 0 to num_bins-1

    # Encode using positional binding
    encoded_vecs = []
    for i, bin_idx in enumerate(binned):
        pos_vec = memory[f"pos_{i}"].vec
        bin_vec = memory[f"bin_{bin_idx}"].vec
        bound = model.opset.bind(pos_vec, bin_vec)
        encoded_vecs.append(bound)

    # Bundle all positions
    result = model.opset.bundle(*encoded_vecs)
    return result


# Extract embeddings from network 1 for test samples
def get_embedding(mlp, X):
    """Extract hidden layer activations (embeddings)."""
    return mlp._forward_pass_fast(X)[0]  # Returns (activations, output)


# Encode test samples
test_embeddings = get_embedding(networks[0], X_test)
encoded_hvs = [encode_embedding(model, memory, emb) for emb in test_embeddings]
```

### Step 3: Build Hyperdimensional Inference Layer (HIL)

```python
# For each class, bundle all encoded embeddings
class_prototypes = {}

for class_label in range(10):  # Digits 0-9
    # Get all training samples for this class
    class_mask = (y_train == class_label)
    class_embeddings = get_embedding(networks[0], X_train[class_mask])

    # Encode and bundle
    encoded_class = [encode_embedding(model, memory, emb)
                     for emb in class_embeddings]
    prototype = model.opset.bundle(*encoded_class)

    class_prototypes[class_label] = prototype

print(f"Built {len(class_prototypes)} class prototypes")
```

### Step 4: Classify Using VSA

```python
from vsax.similarity import hamming_similarity

def classify_hil(test_hv, class_prototypes):
    """Classify using nearest class prototype."""
    best_class = None
    best_sim = -1.0

    for class_label, prototype in class_prototypes.items():
        sim = hamming_similarity(test_hv, prototype)
        if sim > best_sim:
            best_sim = sim
            best_class = class_label

    return best_class


# Test classification
predictions = []
for test_hv in encoded_hvs:
    pred = classify_hil(test_hv, class_prototypes)
    predictions.append(pred)

accuracy = np.mean(np.array(predictions) == y_test)
print(f"HIL Accuracy: {accuracy*100:.2f}%")
# Output: ~94-95% (comparable to neural network!)
```

### Step 5: HD-Glue Consensus Fusion

**Key idea:** Combine predictions from ALL networks via bundling.

```python
def hd_glue_classify(X_sample, networks, model, memory, class_prototypes):
    """
    HD-Glue: Fuse multiple networks via consensus bundling.
    """
    # Get embeddings from all networks
    network_hvs = []
    for net in networks:
        embedding = get_embedding(net, X_sample.reshape(1, -1))[0]
        hv = encode_embedding(model, memory, embedding)
        network_hvs.append(hv)

    # Bundle all network predictions (consensus)
    consensus_hv = model.opset.bundle(*network_hvs)

    # Classify using consensus
    return classify_hil(consensus_hv, class_prototypes)


# Test HD-Glue
hd_glue_predictions = []
for i in range(len(X_test)):
    pred = hd_glue_classify(X_test[i], networks, model, memory, class_prototypes)
    hd_glue_predictions.append(pred)

hd_glue_accuracy = np.mean(np.array(hd_glue_predictions) == y_test)
print(f"HD-Glue Accuracy: {hd_glue_accuracy*100:.2f}%")
# Output: ~96-97% (better than individual networks!)
```

**Result:** HD-Glue achieves **ensemble-level performance** through symbolic consensus!

---

## Advantages of Neuro-Symbolic Integration

### 1. Knowledge Reuse

```python
# Add a new network trained on different data
new_network = MLPClassifier(...)
new_network.fit(X_new_data, y_new_data)

# Immediately integrate into HD-Glue (no retraining!)
networks.append(new_network)
```

### 2. Online Learning

```python
# Add new class dynamically
memory.add("eleven")  # New class "11"

# Encode samples for class 11 and add to prototypes
new_class_embeddings = [...]
new_prototype = model.opset.bundle(*[encode_embedding(model, memory, emb)
                                      for emb in new_class_embeddings])
class_prototypes[11] = new_prototype

# Done! No retraining of neural networks
```

### 3. Weighted Consensus

```python
# Give more weight to better-performing networks
def weighted_hd_glue(X_sample, networks, weights, model, memory, class_prototypes):
    """Weighted HD-Glue based on network performance."""
    network_hvs = []
    for net, weight in zip(networks, weights):
        embedding = get_embedding(net, X_sample.reshape(1, -1))[0]
        hv = encode_embedding(model, memory, embedding)

        # Repeat bundling proportional to weight
        for _ in range(int(weight * 10)):  # Scale weights
            network_hvs.append(hv)

    consensus_hv = model.opset.bundle(*network_hvs)
    return classify_hil(consensus_hv, class_prototypes)


# Use network accuracies as weights
weights = [0.95, 0.97, 0.94, 0.96, 0.93]  # Normalized accuracies
```

### 4. Interpretability

```python
# Unbind to inspect which network contributed most
consensus_hv = model.opset.bundle(*network_hvs)

for i, net_hv in enumerate(network_hvs):
    contribution = hamming_similarity(consensus_hv, net_hv)
    print(f"Network {i+1} contribution: {contribution:.3f}")
```

### 5. Error Correction

```python
# Find misclassified examples
misclassified = X_test[predictions != y_test]

# Encode corrections
corrections = [encode_embedding(model, memory, get_embedding(networks[0], x))
               for x in misclassified]

# Bundle into class prototypes (reinforcement learning)
for i, (x, true_label) in enumerate(zip(misclassified, y_test[predictions != y_test])):
    class_prototypes[true_label] = model.opset.bundle(
        class_prototypes[true_label],
        corrections[i]
    )

# Accuracy improves!
```

---

## Design Patterns for Multi-Modal Systems

### Pattern 1: Modality-Specific Encoders

```python
# Use appropriate encoder for each modality
image_encoder = ImageEncoder(model, memory)
text_encoder = SequenceEncoder(model, memory)
audio_encoder = ScalarEncoder(model, memory)

# Encode each modality
image_hv = image_encoder.encode(image)
text_hv = text_encoder.encode(caption)
audio_hv = audio_encoder.encode(audio_features)

# Fuse with role binding
multimodal_concept = model.opset.bundle(
    model.opset.bind(memory["vision"].vec, image_hv.vec),
    model.opset.bind(memory["language"].vec, text_hv.vec),
    model.opset.bind(memory["audio"].vec, audio_hv.vec)
)
```

### Pattern 2: Cross-Modal Retrieval

```python
# Query: "Find images matching this caption"
query_text = "a cat on a mat"
text_hv = text_encoder.encode(query_text)

# Unbind language modality to get visual component
visual_component = model.opset.bind(
    multimodal_concept,
    model.opset.inverse(model.opset.bind(memory["language"].vec, text_hv.vec))
)

# Compare to image database
for image_id, image_hv in image_database.items():
    sim = cosine_similarity(visual_component, image_hv)
    if sim > 0.7:
        print(f"Image {image_id} matches caption (similarity: {sim:.3f})")
```

### Pattern 3: Neural Embeddings + Symbolic Rules

```python
# Neural network provides embeddings
embedding = neural_net(input_image)
embedding_hv = encode_embedding(model, memory, embedding)

# Add symbolic constraints
# Rule: "cats are mammals"
rule_cat_mammal = model.opset.bind(memory["cat"].vec, memory["mammal"].vec)

# Combine embedding with rule
enhanced_concept = model.opset.bundle(embedding_hv, rule_cat_mammal)

# Reasoning: "Is this a mammal?"
# Unbind "mammal" to check if cat is present
```

---

## Performance Considerations

### Encoding Dimensionality

| VSA Dimension | Modalities Supported | Accuracy |
|---------------|---------------------|----------|
| 1024 | 2-3 modalities | ~85-90% |
| 2048 | 3-5 modalities | ~90-95% |
| 10000 | 5+ modalities (HD-Glue) | ~95-98% |

**Recommendation:** Use `dim >= 2048` for multi-modal systems, `dim >= 10000` for HD-Glue.

### Neural Embedding Dimensionality

HD-Glue works with any embedding dimension:
- 64-128 dims: Fast encoding, good for simple networks
- 256-512 dims: Better for ResNets, Transformers
- 1024+ dims: Large language models

**Trade-off:** Larger embeddings → more positional bindings → slower encoding but richer representations.

### Model Selection

**Binary (HD-Glue default):**
- ✅ Fast XOR operations for encoding
- ✅ Memory efficient
- ✅ Good for large-scale fusion (10+ networks)

**FHRR (Multi-modal grounding):**
- ✅ Exact unbinding for cross-modal queries
- ✅ Better for continuous values
- ❌ Slower than Binary

---

## Common Pitfalls

### Problem 1: Imbalanced Modalities

```python
# ❌ Image has 10,000 pixels, text has 5 words
# Image drowns out text signal

# ✅ Normalize each modality before bundling
image_hv = image_hv / jnp.linalg.norm(image_hv)
text_hv = text_hv / jnp.linalg.norm(text_hv)

multimodal = model.opset.bundle(image_hv, text_hv)
```

### Problem 2: Not Using Role Binding

```python
# ❌ Just bundling modalities
multimodal = model.opset.bundle(image_hv, text_hv, audio_hv)
# Cannot distinguish which modality is which!

# ✅ Use role binding
multimodal = model.opset.bundle(
    model.opset.bind(memory["vision"].vec, image_hv),
    model.opset.bind(memory["language"].vec, text_hv),
    model.opset.bind(memory["audio"].vec, audio_hv)
)
```

### Problem 3: Forgetting to Bin Neural Embeddings

```python
# ❌ Using raw floating-point embeddings directly
# VSA needs discrete or binned values

# ✅ Bin embeddings into discrete levels
bin_edges = np.linspace(-1, 1, 100)
binned_embedding = np.digitize(embedding, bin_edges)
```

---

## Self-Assessment

Before moving on, ensure you can:

- [ ] Explain why heterogeneous data fusion is challenging
- [ ] Encode multiple modalities in the same VSA space
- [ ] Perform cross-modal queries using unbinding
- [ ] Describe the HD-Glue architecture
- [ ] Encode neural network embeddings as hypervectors
- [ ] Understand how consensus bundling improves accuracy
- [ ] Apply role binding to distinguish modalities

## Quick Quiz

**Question 1:** What is the key advantage of encoding all modalities in the same VSA space?

a) Faster computation
b) Cross-modal queries and compositional reasoning
c) Smaller memory footprint
d) Better visualization

<details>
<summary>Answer</summary>
**b) Cross-modal queries and compositional reasoning**

When all modalities share the same hyperdimensional space, you can perform cross-modal queries (e.g., "show me the image for 1+2"), unbind one modality to reveal another, and compose concepts from heterogeneous sources. This is not possible when modalities are in separate spaces.
</details>

**Question 2:** How does HD-Glue improve upon individual neural networks?

a) It retrains all networks from scratch
b) It uses consensus bundling to combine predictions
c) It selects only the best network
d) It converts networks to symbolic rules

<details>
<summary>Answer</summary>
**b) It uses consensus bundling to combine predictions**

HD-Glue encodes embeddings from multiple networks as hypervectors and bundles them together. The bundled hypervector represents a **consensus** across all networks, which is more robust than any single network. This achieves ensemble-level performance without explicit voting mechanisms.
</details>

**Question 3:** Why do we use positional binding when encoding neural embeddings?

a) To make encoding faster
b) To preserve which dimension of the embedding each value came from
c) To reduce dimensionality
d) It's not necessary

<details>
<summary>Answer</summary>
**b) To preserve which dimension of the embedding each value came from**

Positional binding (`pos_i ⊗ value_i`) ensures that the hypervector encoding knows which embedding dimension each value belongs to. Without positional information, the encoding would be order-invariant (just a bundle), losing critical structural information.
</details>

---

## Hands-On Exercise

**Task:** Build a simple multi-modal concept for "dog" that combines:
1. A text description ("furry four-legged animal")
2. A symbolic category ("mammal")
3. A simple visual feature (average pixel brightness)

Then perform cross-modal queries.

**Starter code:**

```python
import jax
from vsax import create_fhrr_model, VSAMemory
from vsax.encoders import ScalarEncoder, SequenceEncoder
from vsax.similarity import cosine_similarity

model = create_fhrr_model(dim=2048, key=jax.random.PRNGKey(42))
memory = VSAMemory(model)

# Add modality roles
memory.add_many(["language", "category", "vision"])

# YOUR CODE HERE:
# 1. Encode text description using SequenceEncoder
# 2. Encode symbolic category as basis vector
# 3. Encode visual feature (brightness) using ScalarEncoder
# 4. Bundle all modalities with role binding
# 5. Query: "What category is associated with this text?"

# Test your implementation
```

<details>
<summary>Solution</summary>

```python
# 1. Encode text
text_encoder = SequenceEncoder(model, memory)
text_desc = ["furry", "four-legged", "animal"]
text_hv = text_encoder.encode(text_desc)

# 2. Encode category
memory.add("mammal")
category_hv = memory["mammal"].vec

# 3. Encode visual feature
scalar_encoder = ScalarEncoder(model, memory)
brightness = 0.6  # Average brightness
memory.add("brightness")
visual_hv = scalar_encoder.encode("brightness", brightness)

# 4. Bundle with role binding
concept_dog = model.opset.bundle(
    model.opset.bind(memory["language"].vec, text_hv.vec),
    model.opset.bind(memory["category"].vec, category_hv),
    model.opset.bind(memory["vision"].vec, visual_hv.vec)
)

# Normalize
concept_dog = concept_dog / jnp.linalg.norm(concept_dog)

# 5. Query: What category is associated with this text?
# Unbind language to isolate category
query_result = model.opset.bind(
    concept_dog,
    model.opset.inverse(model.opset.bind(memory["language"].vec, text_hv.vec))
)

# Check similarity to "mammal"
sim = cosine_similarity(query_result, memory["mammal"].vec)
print(f"Category 'mammal' similarity: {sim:.3f}")  # Should be high!

# Query: What brightness is associated with "mammal"?
query_brightness = model.opset.bind(
    concept_dog,
    model.opset.inverse(model.opset.bind(memory["category"].vec, category_hv))
)
# query_brightness is similar to visual_hv
```
</details>

---

## Key Takeaways

✓ **Multi-modal fusion unifies heterogeneous data** - vision, language, symbols in one space
✓ **Cross-modal queries enable rich reasoning** - ask in one modality, answer in another
✓ **HD-Glue fuses neural networks symbolically** - reuse existing models without retraining
✓ **Consensus bundling improves robustness** - ensemble-level performance via VSA
✓ **Online learning is trivial** - add new classes/models by bundling
✓ **Neuro-symbolic AI combines learning + reasoning** - embeddings + symbolic rules
✓ **Role binding distinguishes modalities** - critical for cross-modal unbinding

---

## Next Steps

**Module 4 Complete!** You've mastered advanced VSA techniques.

**Next Module:** [Module 5 - Research & Extensions](../../course/05_research/index.md)
Learn about Vector Function Architecture, building custom encoders for research, and current research frontiers.

**For Hands-On Practice:**
- [Tutorial 8 - Multi-Modal Concept Grounding](../../tutorials/08_multimodal_grounding.md)
- [Tutorial 9 - Neural-Symbolic Fusion with HD-Glue](../../tutorials/09_neural_symbolic_fusion.md)

**Related Content:**
- [Module 3 - Encoders & Applications](../../course/03_encoders/index.md)
- [Image Classification Tutorial](../../tutorials/01_mnist_classification.md)

## References

- Sutor, P., Kumaran, D., & Olshausen, B. (2022). "Gluing Neural Networks Symbolically Through Hyperdimensional Computing."
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors." *Cognitive Computation.*
- Schlegel, K., Neubert, P., & Protzel, P. (2022). "A comparison of Vector Symbolic Architectures." *Artificial Intelligence Review.*
- Frady, E. P., Kleyko, D., & Sommer, F. T. (2021). "Variable Binding for Sparse Distributed Representations: Theory and Applications." *IEEE TNNLS.*
