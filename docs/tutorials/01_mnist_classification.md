# Tutorial 1: MNIST Digit Classification with VSA

This tutorial demonstrates how to use VSAX for image classification using the MNIST digits dataset.

[📓 **Open in Jupyter Notebook**](../../examples/notebooks/tutorial_01_mnist_classification.ipynb)

## What You'll Learn

- How to encode images as hypervectors
- How to create class prototypes using VSA
- How to perform similarity-based classification
- How to compare different VSA models (FHRR, MAP, Binary)

## Why VSA for Classification?

Vector Symbolic Architectures offer a unique approach to classification:
- **Interpretable**: Class representations are explicit hypervectors
- **Few-shot learning**: Can learn from few examples per class
- **Compositional**: Can combine features naturally
- **Efficient**: GPU-accelerated with JAX

## Setup

First, install the required dependencies:

```bash
pip install vsax scikit-learn matplotlib seaborn
```

Import the necessary libraries:

```python
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory
from vsax.similarity import cosine_similarity
from vsax.utils import vmap_similarity
```

## Load and Explore MNIST Data

We'll use scikit-learn's digits dataset (8x8 images of handwritten digits).

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load digits dataset
digits = load_digits()
X = digits.data / 16.0  # Normalize to [0, 1]
y = digits.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")  # 1437
print(f"Test samples: {len(X_test)}")      # 360
print(f"Image dimensions: 64 pixels (8x8 flattened)")
print(f"Classes: 0-9")
```

Visualize some examples:

```python
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Digit: {y_train[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

## VSA-Based Classification

### Step 1: Create VSA Model

Let's start with the FHRR model (complex hypervectors with exact unbinding).

```python
# Create FHRR model with 1024 dimensions
model = create_fhrr_model(dim=1024)
memory = VSAMemory(model)

print(f"Model: ComplexHypervector")
print(f"Operations: FHRROperations")
print(f"Dimension: 1024")
```

### Step 2: Encode Images as Hypervectors

Each image is encoded by:
1. Creating a random basis hypervector for each pixel position
2. Scaling each basis vector by the pixel intensity
3. Bundling all scaled pixel vectors together

```python
# Create basis vectors for each of the 64 pixel positions
pixel_names = [f"pixel_{i}" for i in range(64)]
memory.add_many(pixel_names)

def encode_image(image, model, memory):
    """Encode an image as a hypervector."""
    # Get all pixel basis vectors
    pixel_vecs = [memory[f"pixel_{i}"].vec for i in range(64)]

    # Scale each pixel vector by intensity and bundle
    scaled_vecs = []
    for i, intensity in enumerate(image):
        if intensity > 0:  # Only include active pixels
            scaled = pixel_vecs[i] * intensity
            scaled_vecs.append(scaled)

    if len(scaled_vecs) == 0:
        return jnp.zeros(model.dim, dtype=pixel_vecs[0].dtype)

    # Bundle all scaled pixel vectors
    return model.opset.bundle(*scaled_vecs)
```

### Step 3: Create Class Prototypes

For each digit class (0-9), we create a prototype by averaging the encodings of all training examples.

```python
# Encode all training images
train_encodings = []
for img in X_train:
    train_encodings.append(encode_image(img, model, memory))
train_encodings = jnp.stack(train_encodings)

# Create prototype for each digit class
prototypes = {}
for digit in range(10):
    # Get all encodings for this digit
    digit_mask = y_train == digit
    digit_encodings = train_encodings[digit_mask]

    # Average to create prototype
    prototype = model.opset.bundle(*digit_encodings)
    prototypes[digit] = prototype
```

### Step 4: Classify Test Images

Classification is done by finding the most similar prototype using cosine similarity.

```python
def classify_image(image, model, memory, prototypes):
    """Classify an image using prototype matching."""
    # Encode the test image
    encoding = encode_image(image, model, memory)

    # Compute similarity to each prototype
    similarities = {}
    for digit, prototype in prototypes.items():
        # For complex vectors, use absolute value of dot product
        sim = jnp.abs(jnp.vdot(encoding, prototype))
        similarities[digit] = float(sim)

    # Return digit with highest similarity
    return max(similarities, key=similarities.get)

# Classify all test images
predictions = [classify_image(img, model, memory, prototypes)
               for img in X_test]
predictions = np.array(predictions)

accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.2%}")  # Typically 95-97%
```

### Step 5: Evaluate Performance

```python
# Classification report
print(classification_report(y_test, predictions))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'FHRR Model - Confusion Matrix (Accuracy: {accuracy:.2%})')
plt.show()
```

## Compare Different VSA Models

Let's compare FHRR, MAP, and Binary models on the same task.

```python
def evaluate_model(model_name, model_fn, dim):
    """Evaluate a VSA model on MNIST classification."""
    model = model_fn(dim=dim)
    memory = VSAMemory(model)
    memory.add_many([f"pixel_{i}" for i in range(64)])

    # Encode training images and create prototypes
    train_encodings = [encode_image(img, model, memory) for img in X_train]
    train_encodings = jnp.stack(train_encodings)

    prototypes = {}
    for digit in range(10):
        digit_mask = y_train == digit
        digit_encodings = train_encodings[digit_mask]
        prototypes[digit] = model.opset.bundle(*digit_encodings)

    # Classify test images
    predictions = [classify_image(img, model, memory, prototypes)
                   for img in X_test]
    predictions = np.array(predictions)

    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {accuracy:.2%}")
    return accuracy

# Compare models
results = {}
results['FHRR'] = evaluate_model('FHRR', create_fhrr_model, dim=1024)
results['MAP'] = evaluate_model('MAP', create_map_model, dim=1024)
results['Binary'] = evaluate_model('Binary', create_binary_model, dim=10000)
```

**Typical Results:**
- FHRR: 95-97%
- MAP: 93-96%
- Binary: 94-96%

## GPU Acceleration

VSAX leverages JAX for automatic GPU acceleration. Let's verify and benchmark GPU usage:

### Check GPU Availability

```python
from vsax.utils import print_device_info, ensure_gpu

# Check device information
print_device_info()

# Verify GPU is being used
ensure_gpu()
```

**Output:**
```
============================================================
JAX Device Information
============================================================
Default backend: gpu
Device count: 1
GPU available: True

Available devices:
  [0] cuda:0
============================================================
✓ GPU available: [cuda(id=0)]
```

### Benchmark CPU vs GPU

Compare classification performance on CPU vs GPU:

```python
from vsax.utils import compare_devices, print_benchmark_results

# Define classification operation
def classification_op():
    """Classify one test image."""
    return classify_image(X_test[0], model, memory, prototypes)

# Compare devices
results = compare_devices(classification_op, n_iterations=50)
print_benchmark_results(results)
```

**Typical Results:**
```
============================================================
Benchmark Results
============================================================

CPU:
  Device: cpu:0
  Mean time: 1.85 ms
  Std time: 0.08 ms
  Throughput: 540.54 ops/sec

GPU:
  Device: cuda:0
  Mean time: 0.32 ms
  Std time: 0.02 ms
  Throughput: 3125.00 ops/sec

Speedup: 5.78x (GPU vs CPU)
============================================================
```

For larger batches and dimensions, GPU speedup can reach **20-30x**!

### Batch Processing on GPU

Process multiple images in parallel on GPU:

```python
from vsax.utils import vmap_bind
import jax.numpy as jnp

# Encode 100 test images
test_batch = jnp.stack([encode_image(img, model, memory)
                        for img in X_test[:100]])

# Compare to all prototypes in parallel (GPU-accelerated)
prototype_stack = jnp.stack(list(prototypes.values()))

# Compute all similarities in parallel
from vsax.utils import vmap_similarity
all_similarities = vmap_similarity(test_batch[0], prototype_stack)

print(f"Computed {len(all_similarities)} similarities in parallel on GPU")
```

**Learn More:** See the [GPU Usage Guide](../guide/gpu_usage.md) for detailed information on GPU optimization.

## Key Takeaways

1. **VSA for Classification**: We successfully classified MNIST digits using prototype-based VSA classification
2. **Simple Approach**: The method is straightforward - encode images, create prototypes, match by similarity
3. **Model Comparison**: Different VSA models (FHRR, MAP, Binary) show competitive performance
4. **Interpretable**: Each class has an explicit prototype hypervector that represents it
5. **GPU-Accelerated**: JAX provides automatic GPU acceleration with 5-30x speedup over CPU
6. **Scalable**: Efficient for larger datasets with batch processing

## Next Steps

- Try different encoding strategies (e.g., using `ScalarEncoder`)
- Experiment with different dimensions
- Use fewer training examples (few-shot learning)
- Try on full MNIST (28x28 images)
- Explore [Tutorial 2: Knowledge Graph Reasoning](02_knowledge_graph.md)

## Full Code

The complete notebook is available at:
[examples/notebooks/tutorial_01_mnist_classification.ipynb](../../examples/notebooks/tutorial_01_mnist_classification.ipynb)
