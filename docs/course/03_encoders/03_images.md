# Lesson 3.3: Application - Image Classification

**Duration:** ~60 minutes (30 min theory + 30 min tutorial)

**Learning Objectives:**

- Understand how to encode images as hypervectors
- Learn the prototype-based classification approach
- Apply bundling to create class prototypes
- Use similarity search for classification
- Complete a full MNIST classification project
- Compare FHRR, MAP, and Binary models on real data

---

## Introduction

You've learned how to encode structured data (scalars, sequences, dictionaries). Now let's apply these concepts to **real-world image classification**!

In this lesson, we'll classify handwritten digits (MNIST) using VSA. This demonstrates how VSA's compositional operations naturally handle high-dimensional data like images.

**What makes VSA good for classification?**
- ✅ **Interpretable:** Class prototypes are explicit hypervectors
- ✅ **Few-shot learning:** Learn from few examples per class
- ✅ **Compositional:** Combine pixel features naturally
- ✅ **Efficient:** GPU-accelerated, no gradient descent needed
- ✅ **Robust:** Gracefully handles noise and occlusion

---

## The VSA Classification Pipeline

VSA classification follows a simple 3-step process:

```
1. ENCODE
   Images → Hypervectors
   [Use bundling to combine pixel features]

2. LEARN
   Create class prototypes by bundling training examples
   Prototype[i] = Bundle(all images of class i)

3. CLASSIFY
   Find prototype with highest similarity to test image
   Prediction = argmax(similarity(test, prototype))
```

**Key insight:** No gradient descent, no backpropagation—just encoding and similarity!

---

## Step 1: Encoding Images

### The Challenge: High-Dimensional Pixel Data

An 8×8 grayscale image has 64 pixels, each with intensity in [0, 1].

**How do we encode this as a hypervector?**

### Approach: Spatial Bundling

Each pixel position gets a **random basis vector**, and we bundle pixel values weighted by intensity:

$$\text{image} = \sum_{i=1}^{64} \text{intensity}_i \cdot \text{pixel\_basis}_i$$

**Example:**
```python
from vsax import create_fhrr_model, VSAMemory

model = create_fhrr_model(dim=1024)
memory = VSAMemory(model)

# Create basis vectors for each pixel position
for i in range(64):  # 8x8 = 64 pixels
    memory.add(f"pixel_{i}")

# Encode image: bundle weighted by intensity
def encode_image(image, model, memory):
    """
    Encode 64-pixel image as hypervector.

    Args:
        image: 1D array of 64 pixel intensities [0, 1]
        model: VSA model
        memory: VSA memory with pixel basis vectors

    Returns:
        Encoded hypervector
    """
    encoded = None

    for i, intensity in enumerate(image):
        pixel_vec = memory[f"pixel_{i}"].vec

        # Weight pixel vector by intensity
        weighted = intensity * pixel_vec

        # Bundle (accumulate)
        if encoded is None:
            encoded = weighted
        else:
            encoded = encoded + weighted

    # Normalize
    import jax.numpy as jnp
    encoded = encoded / jnp.linalg.norm(encoded)

    return encoded
```

**Why it works:**
- Pixels with high intensity contribute more to the final hypervector
- Pixel position information is preserved (different basis for each position)
- Similar images → similar hypervectors

---

### Efficient Batch Encoding with vmap

For real applications, use JAX's `vmap` to encode batches efficiently:

```python
import jax.numpy as jnp
from jax import vmap

def encode_batch(images, basis_vectors):
    """
    Encode batch of images efficiently.

    Args:
        images: (batch_size, 64) array of images
        basis_vectors: (64, dim) array of pixel basis vectors

    Returns:
        (batch_size, dim) encoded hypervectors
    """
    # Matrix multiplication: (batch, 64) @ (64, dim) → (batch, dim)
    encoded = images @ basis_vectors

    # Normalize each row
    norms = jnp.linalg.norm(encoded, axis=1, keepdims=True)
    encoded = encoded / norms

    return encoded

# Usage
basis = jnp.stack([memory[f"pixel_{i}"].vec for i in range(64)])
encoded_batch = encode_batch(X_train, basis)
```

**Performance:** ~100× faster than looping!

---

## Step 2: Creating Class Prototypes

Once we have encoded images, we create **prototypes** for each class by **bundling** all training examples:

$$\text{Prototype}_c = \text{normalize}\left(\sum_{i \in \text{class } c} \text{encoded}(x_i)\right)$$

**Code:**
```python
import jax.numpy as jnp

def create_prototypes(encoded_images, labels, num_classes=10):
    """
    Create class prototypes by bundling training examples.

    Args:
        encoded_images: (num_samples, dim) encoded training images
        labels: (num_samples,) class labels
        num_classes: Number of classes

    Returns:
        (num_classes, dim) prototype vectors
    """
    dim = encoded_images.shape[1]
    prototypes = jnp.zeros((num_classes, dim))

    for class_id in range(num_classes):
        # Get all images of this class
        class_mask = labels == class_id
        class_images = encoded_images[class_mask]

        # Bundle (sum and normalize)
        prototype = jnp.sum(class_images, axis=0)
        prototype = prototype / jnp.linalg.norm(prototype)

        prototypes = prototypes.at[class_id].set(prototype)

    return prototypes

# Create prototypes
prototypes = create_prototypes(encoded_train, y_train, num_classes=10)
print(f"Prototype shape: {prototypes.shape}")  # (10, 1024)
```

**Interpretation:**
- Each prototype is the **average** encoded representation of a class
- Captures common features across all training examples
- New examples are classified by similarity to prototypes

---

## Step 3: Classification by Similarity

To classify a test image:

1. Encode test image as hypervector
2. Compute similarity to all class prototypes
3. Predict class with highest similarity

```python
from vsax.similarity import cosine_similarity
from vsax.utils import vmap_similarity

def classify(test_image_encoded, prototypes):
    """
    Classify test image by finding nearest prototype.

    Args:
        test_image_encoded: (dim,) encoded test image
        prototypes: (num_classes, dim) class prototypes

    Returns:
        Predicted class (0-9)
    """
    # Compute similarities to all prototypes
    similarities = vmap_similarity(None, test_image_encoded, prototypes)

    # Return class with highest similarity
    predicted_class = jnp.argmax(similarities)

    return int(predicted_class)

# Classify single test image
prediction = classify(encoded_test[0], prototypes)
print(f"Predicted: {prediction}, True: {y_test[0]}")
```

---

### Batch Classification

For efficient classification of many test images:

```python
def classify_batch(test_images_encoded, prototypes):
    """
    Classify batch of test images.

    Args:
        test_images_encoded: (num_test, dim) encoded test images
        prototypes: (num_classes, dim) class prototypes

    Returns:
        (num_test,) predicted classes
    """
    # Compute similarity matrix: (num_test, dim) @ (dim, num_classes)
    # → (num_test, num_classes)
    similarities = test_images_encoded @ prototypes.T

    # Argmax over classes
    predictions = jnp.argmax(similarities, axis=1)

    return predictions

# Classify all test images
predictions = classify_batch(encoded_test, prototypes)
accuracy = jnp.mean(predictions == y_test)
print(f"Test accuracy: {accuracy:.4f}")
```

---

## Why This Approach Works

### 1. Distributed Representations

Each pixel contributes to the overall hypervector representation. Similar images have similar patterns of pixel activations → similar hypervectors.

### 2. Noise Robustness

**Graceful degradation:** Even if some pixels are corrupted, the overall representation remains similar.

```python
# Original image
clean_image = X_test[0]

# Add noise
noisy_image = clean_image + 0.1 * np.random.randn(64)
noisy_image = np.clip(noisy_image, 0, 1)

# Encode both
clean_encoded = encode_image(clean_image, model, memory)
noisy_encoded = encode_image(noisy_image, model, memory)

# Still similar!
sim = cosine_similarity(clean_encoded, noisy_encoded)
print(f"Similarity: {sim:.4f}")  # ~0.95
```

### 3. Compositional Features

Bundling naturally combines pixel features without explicit feature engineering.

### 4. Few-Shot Learning

Prototypes can be learned from **very few examples**:

```python
# Use only 5 examples per class
few_shot_mask = []
for class_id in range(10):
    class_indices = np.where(y_train == class_id)[0][:5]
    few_shot_mask.extend(class_indices)

few_shot_train = encoded_train[few_shot_mask]
few_shot_labels = y_train[few_shot_mask]

# Create prototypes from 50 total examples
prototypes_few = create_prototypes(few_shot_train, few_shot_labels)

# Still reasonable accuracy!
predictions = classify_batch(encoded_test, prototypes_few)
accuracy = np.mean(predictions == y_test)
print(f"Few-shot accuracy (5 per class): {accuracy:.4f}")
```

---

## Comparing VSA Models: FHRR vs MAP vs Binary

Different VSA models offer different trade-offs for image classification:

| Model | Accuracy | Speed | Memory | Best For |
|-------|----------|-------|---------|----------|
| **FHRR** | Highest (~95%) | Moderate | High (8 bytes/elem) | Maximum accuracy |
| **MAP** | High (~93%) | Fast | Medium (4 bytes/elem) | Balanced |
| **Binary** | Good (~88%) | Fastest | Low (1 bit/elem) | Edge devices, embedded |

**Recommendation:** Start with FHRR for best accuracy, switch to Binary for deployment on constrained hardware.

---

## Hands-On: Complete MNIST Classification Tutorial

Now that you understand the foundations, **complete the full tutorial** to implement and experiment with VSA-based image classification.

**📓 [Tutorial 1: MNIST Digit Classification](../../tutorials/01_mnist_classification.md)**

**What you'll do in the tutorial:**

1. **Setup:** Load MNIST digits dataset (8×8 images)
2. **Encoding:** Implement spatial bundling for images
3. **Training:** Create class prototypes by bundling examples
4. **Evaluation:** Classify test images and measure accuracy
5. **Comparison:** Benchmark FHRR, MAP, and Binary models
6. **Visualization:** Plot confusion matrices and analyze errors
7. **Experimentation:** Try different dimensions, few-shot learning, noise robustness

**Time estimate:** 30-45 minutes

**Prerequisites:**
- Completed Lessons 3.1 and 3.2 (encoders)
- Basic Python and NumPy knowledge
- scikit-learn installed (`pip install scikit-learn`)

---

## Key Concepts from the Tutorial

### 1. Pixel Basis Encoding

Each pixel position gets its own random basis vector. Images are encoded as weighted bundles:

```python
# Basis for 64 pixels
pixel_basis = [memory.add(f"pixel_{i}") for i in range(64)]

# Encode image
image_hv = sum(intensity[i] * pixel_basis[i] for i in range(64))
```

### 2. Prototype Learning

Class prototypes are learned by averaging (bundling) encoded training examples:

```python
# Prototype for digit "3"
digit_3_examples = encoded_train[y_train == 3]
prototype_3 = np.mean(digit_3_examples, axis=0)
prototype_3 = prototype_3 / np.linalg.norm(prototype_3)
```

### 3. Nearest Prototype Classification

Classification is **similarity search** to prototypes:

```python
similarities = [cosine_similarity(test_hv, proto) for proto in prototypes]
predicted_digit = np.argmax(similarities)
```

### 4. Confusion Matrix Analysis

Analyze which digits are confused:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
```

**Common confusions:** 3 ↔ 8, 4 ↔ 9, 5 ↔ 6 (similar shapes)

---

## Extensions and Experiments

After completing the tutorial, try these experiments:

### 1. Dimension Scaling

How does accuracy change with dimension?

```python
dimensions = [256, 512, 1024, 2048, 4096]
accuracies = []

for dim in dimensions:
    model = create_fhrr_model(dim=dim)
    # ... encode, train, test
    accuracies.append(test_accuracy)

plt.plot(dimensions, accuracies)
plt.xlabel('Dimension')
plt.ylabel('Accuracy')
```

**Expected:** Accuracy increases with dimension, plateaus around d=2048.

---

### 2. Few-Shot Learning Curve

How many examples per class are needed?

```python
examples_per_class = [1, 3, 5, 10, 20, 50, 100]
accuracies = []

for n in examples_per_class:
    # Sample n examples per class
    # Create prototypes
    # Test
    accuracies.append(test_accuracy)

plt.plot(examples_per_class, accuracies)
plt.xlabel('Examples per Class')
plt.ylabel('Accuracy')
```

---

### 3. Noise Robustness

Add Gaussian noise to test images:

```python
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]
accuracies = []

for noise in noise_levels:
    noisy_test = X_test + noise * np.random.randn(*X_test.shape)
    noisy_test = np.clip(noisy_test, 0, 1)
    # Encode and classify
    accuracies.append(test_accuracy)

plt.plot(noise_levels, accuracies)
plt.xlabel('Noise Level (σ)')
plt.ylabel('Accuracy')
```

**Expected:** VSA should degrade gracefully (accuracy drops slowly).

---

### 4. Occlusion Robustness

Mask out random pixels:

```python
def occlude_image(image, occlusion_fraction=0.3):
    """Randomly set occlusion_fraction of pixels to 0."""
    mask = np.random.rand(64) > occlusion_fraction
    return image * mask

# Test with occlusion
occluded_test = [occlude_image(img, 0.3) for img in X_test]
# Encode and classify
```

---

## Comparison with Traditional ML

| Approach | Accuracy | Training Time | Interpretability | Few-Shot |
|----------|----------|---------------|------------------|----------|
| **VSA (FHRR)** | ~95% | <1 sec | High (prototypes) | Excellent |
| **SVM** | ~97% | ~5 sec | Low (hyperplane) | Poor |
| **Random Forest** | ~96% | ~10 sec | Medium (trees) | Poor |
| **Neural Network** | ~98% | ~60 sec | Low (weights) | Poor |

**VSA advantages:**
- ✅ Extremely fast training (no gradient descent)
- ✅ Excellent few-shot learning
- ✅ Interpretable prototypes
- ✅ Robust to noise

**VSA disadvantages:**
- ❌ Slightly lower peak accuracy than deep learning
- ❌ Requires careful dimension selection

---

## Real-World Applications

VSA-based classification extends beyond MNIST:

**1. Medical Imaging**
- X-ray classification (pneumonia detection)
- MRI scan analysis
- Skin lesion classification
- **Advantage:** Few-shot learning (limited medical data)

**2. Manufacturing Quality Control**
- Defect detection in products
- Surface inspection
- **Advantage:** Fast inference on edge devices (Binary model)

**3. Biometric Recognition**
- Face recognition
- Fingerprint matching
- **Advantage:** Template protection (hypervector representations)

**4. Remote Sensing**
- Satellite image classification
- Land cover mapping
- **Advantage:** Handles noise and missing data

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Explain how images are encoded using spatial bundling
- [ ] Create class prototypes by bundling training examples
- [ ] Classify new images using similarity to prototypes
- [ ] Implement efficient batch encoding with vmap
- [ ] Complete the MNIST classification tutorial
- [ ] Compare FHRR, MAP, and Binary models on real data
- [ ] Analyze confusion matrices and error patterns
- [ ] Extend the approach to custom datasets

---

## Quick Quiz

**Q1:** Why do we create separate basis vectors for each pixel position?

a) To reduce memory usage
b) To preserve spatial information (pixel location matters)
c) To make encoding faster
d) Required by VSAX API

<details>
<summary>Answer</summary>
**b) To preserve spatial information** - Each pixel position needs its own basis vector so the encoding knows "this is pixel 0" vs "this is pixel 63". Without position-specific bases, we'd lose where each intensity value came from.
</details>

**Q2:** How are class prototypes created in VSA classification?

a) Gradient descent optimization
b) K-means clustering
c) Bundling (averaging) all training examples of that class
d) Random sampling

<details>
<summary>Answer</summary>
**c) Bundling all training examples** - Prototypes are created by summing (bundling) the encoded hypervectors of all training images for a class, then normalizing. This creates an "average" representation.
</details>

**Q3:** Why is VSA good for few-shot learning?

a) Uses less memory than neural networks
b) Prototypes can be learned from very few examples via bundling
c) Doesn't require labeled data
d) Runs faster on GPUs

<details>
<summary>Answer</summary>
**b) Prototypes from few examples** - Bundling allows us to create meaningful prototypes even with 1-5 examples per class. Each new example refines the prototype through averaging.
</details>

**Q4:** Which VSA model should you use for embedded edge device deployment?

a) FHRR (highest accuracy)
b) MAP (balanced)
c) Binary (minimal memory, fast XOR operations)
d) All models work equally well

<details>
<summary>Answer</summary>
**c) Binary** - Binary model uses only 1 bit per element (8× less memory than FHRR), and XOR operations are extremely fast on embedded hardware. Accuracy is slightly lower (~88% vs 95%) but acceptable for many applications.
</details>

---

## Key Takeaways

1. **VSA enables interpretable classification** - Class prototypes are explicit hypervectors
2. **Spatial bundling encodes images** - Weight pixel basis vectors by intensity
3. **Prototype learning is simple** - Bundle training examples, no gradient descent
4. **Classification is similarity search** - argmax(similarity to prototypes)
5. **Few-shot learning works naturally** - Prototypes from as few as 1-5 examples
6. **Robust to noise** - Distributed representations degrade gracefully
7. **Model trade-offs** - FHRR (accuracy), MAP (balanced), Binary (efficiency)

---

**Next:** [Lesson 3.4: Application - Knowledge Graph Reasoning](04_knowledge_graphs.md)

Apply DictEncoder and GraphEncoder to build queryable knowledge bases.

**Previous:** [Lesson 3.2: Structured Data - Dictionaries and Sets](02_dict_sets.md)
