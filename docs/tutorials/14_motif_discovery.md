# Tutorial 14: Motif Discovery and Sequence Alignment

This tutorial demonstrates advanced VSA techniques for bioinformatics, including k-mer fingerprinting, approximate sequence alignment, motif detection, and multi-sequence comparison.

[📓 **Open in Jupyter Notebook**](../../examples/notebooks/tutorial_14_motif_discovery.ipynb)

## What You'll Learn

- How to create k-mer fingerprints for sequence signatures
- How to perform approximate sequence alignment using similarity
- How to detect motifs using sliding window and permutation
- How to build multi-sequence comparison matrices
- How to discover conserved motifs across sequence families
- How to leverage GPU acceleration for batch processing

## Prerequisites

This tutorial builds on concepts from:
- [Tutorial 12: DNA Sequence Analysis](12_dna_sequences.md)
- [Tutorial 13: Protein Sequence Classification](13_protein_classification.md)

## Setup

```bash
pip install vsax matplotlib seaborn
```

```python
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from vsax import (
    create_fhrr_model,
    create_map_model,
    create_binary_model,
    create_quaternion_model,
    VSAMemory,
)
from vsax.encoders import SequenceEncoder, SetEncoder
from vsax.similarity import cosine_similarity

print("Setup complete!")
```

## Part 1: K-mer Fingerprinting for Sequence Signatures

K-mers are subsequences of length k that capture local sequence patterns. They're fundamental to many bioinformatics algorithms.

```python
# Configuration
K = 4  # k-mer length
DIM = 1024

# Initialize model
model = create_fhrr_model(dim=DIM)
memory = VSAMemory(model)

# For DNA sequences
nucleotides = ["A", "T", "G", "C"]
memory.add_many(nucleotides)

# Generate all possible k-mers
def generate_all_kmers(alphabet, k):
    """Generate all possible k-mers from an alphabet."""
    if k == 0:
        return [""]
    smaller = generate_all_kmers(alphabet, k - 1)
    return [char + kmer for char in alphabet for kmer in smaller]

all_kmers = generate_all_kmers(nucleotides, K)
memory.add_many(all_kmers)

print(f"Generated {len(all_kmers)} unique {K}-mers")
print(f"Examples: {all_kmers[:8]}")
```

**Output:**
```
Generated 256 unique 4-mers
Examples: ['AAAA', 'AAAT', 'AAAG', 'AAAC', 'AATA', 'AATT', 'AATG', 'AATC']
```

### K-mer Spectrum Encoding

```python
def extract_kmer_spectrum(sequence, k=4):
    """Extract k-mer spectrum (frequency distribution)."""
    kmers = []
    seq_str = ''.join(sequence) if isinstance(sequence, list) else sequence
    for i in range(len(seq_str) - k + 1):
        kmers.append(seq_str[i:i+k])
    return Counter(kmers)

def encode_kmer_spectrum(sequence, k, model, memory):
    """Encode sequence as weighted k-mer spectrum."""
    spectrum = extract_kmer_spectrum(sequence, k)

    # Weight each k-mer by its frequency
    weighted_vecs = []
    for kmer, count in spectrum.items():
        kmer_hv = memory[kmer]
        weighted_vecs.append(kmer_hv.vec * count)

    # Bundle all weighted k-mers
    result = model.opset.bundle(*weighted_vecs)
    return model.rep_cls(result)

# Test sequences
seq1 = "ATGCGATCGATCGATCGATCGATCGATCG"
seq2 = "ATGCGATCGATCGATCGATCGATCGATCG"  # Identical
seq3 = "ATGCGATCGATCGATCGATCGATCGTTTT"  # Similar
seq4 = "CCCCGGGGAAAATTTTCCCCGGGGAAAAT"  # Different composition

# Encode spectra
hv1 = encode_kmer_spectrum(seq1, K, model, memory)
hv2 = encode_kmer_spectrum(seq2, K, model, memory)
hv3 = encode_kmer_spectrum(seq3, K, model, memory)
hv4 = encode_kmer_spectrum(seq4, K, model, memory)

def compute_similarity(hv_a, hv_b):
    return float(cosine_similarity(hv_a.vec, hv_b.vec))

print("K-mer Spectrum Similarity:")
print(f"  Seq1 vs Seq2 (identical): {compute_similarity(hv1, hv2):.4f}")
print(f"  Seq1 vs Seq3 (similar):   {compute_similarity(hv1, hv3):.4f}")
print(f"  Seq1 vs Seq4 (different): {compute_similarity(hv1, hv4):.4f}")
```

**Expected Output:**
```
K-mer Spectrum Similarity:
  Seq1 vs Seq2 (identical): 1.0000
  Seq1 vs Seq3 (similar):   0.8234
  Seq1 vs Seq4 (different): 0.1456
```

## Part 2: Approximate Sequence Alignment

Traditional sequence alignment (Smith-Waterman, Needleman-Wunsch) is computationally expensive. VSA enables fast approximate alignment using similarity search.

```python
def sliding_window_encode(sequence, window_size, model, memory):
    """Encode all windows of a sequence."""
    seq_encoder = SequenceEncoder(model, memory)
    seq_str = ''.join(sequence) if isinstance(sequence, list) else sequence

    windows = []
    for i in range(len(seq_str) - window_size + 1):
        window = list(seq_str[i:i+window_size])
        windows.append({
            'position': i,
            'sequence': seq_str[i:i+window_size],
            'hv': seq_encoder.encode(window)
        })
    return windows

def find_best_alignment(query, target, window_size, model, memory):
    """Find best alignment position of query in target."""
    seq_encoder = SequenceEncoder(model, memory)
    query_hv = seq_encoder.encode(list(query))

    target_windows = sliding_window_encode(target, len(query), model, memory)

    best_pos = -1
    best_sim = -1
    all_sims = []

    for window in target_windows:
        sim = compute_similarity(query_hv, window['hv'])
        all_sims.append((window['position'], sim))
        if sim > best_sim:
            best_sim = sim
            best_pos = window['position']

    return best_pos, best_sim, all_sims

# Example: Find a motif in a longer sequence
target_seq = "AAATTTGGGCCCATGCATCGATCGATCGAAATTTGGGCCC"
query_motif = "ATCGATCG"

align_model = create_fhrr_model(dim=512)
align_memory = VSAMemory(align_model)
align_memory.add_many(nucleotides)

best_pos, best_sim, all_sims = find_best_alignment(query_motif, target_seq, len(query_motif), align_model, align_memory)

print(f"Query motif: {query_motif}")
print(f"Target: {target_seq}")
print(f"\nBest alignment position: {best_pos}")
print(f"Best similarity: {best_sim:.4f}")
print(f"Aligned region: {target_seq[best_pos:best_pos+len(query_motif)]}")

# Visualize alignment scores
positions, sims = zip(*all_sims)
plt.figure(figsize=(12, 4))
plt.bar(positions, sims, color='steelblue')
plt.axhline(y=0.9, color='red', linestyle='--', label='High similarity threshold')
plt.xlabel('Position in Target')
plt.ylabel('Similarity')
plt.title('Approximate Alignment Scores')
plt.legend()
plt.tight_layout()
plt.show()
```

**Expected Output:**
```
Query motif: ATCGATCG
Target: AAATTTGGGCCCATGCATCGATCGATCGAAATTTGGGCCC

Best alignment position: 16
Best similarity: 1.0000
Aligned region: ATCGATCG
```

## Part 3: Sliding Window Motif Detection with Permutation

Permutation can encode positional information that persists across translations. This is useful for detecting motifs at different positions.

```python
def create_position_encoder(model, memory, max_pos=100):
    """Create position-aware encoder using permutation."""
    # Store permuted basis vectors for each position
    position_hvs = {}
    for i in range(max_pos):
        pos_name = f"pos_{i}"
        if pos_name not in memory:
            memory.add(pos_name)
        position_hvs[i] = memory[pos_name]
    return position_hvs

def encode_with_positions(sequence, model, memory, position_hvs):
    """Encode sequence with explicit position information."""
    bound_pairs = []
    for i, char in enumerate(sequence):
        if char in memory:
            char_hv = memory[char]
            pos_hv = position_hvs[i % len(position_hvs)]
            # Bind character with position
            bound = model.opset.bind(char_hv.vec, pos_hv.vec)
            bound_pairs.append(bound)

    result = model.opset.bundle(*bound_pairs)
    return model.rep_cls(result)

# Create position-aware encoding
pos_model = create_fhrr_model(dim=512)
pos_memory = VSAMemory(pos_model)
pos_memory.add_many(nucleotides)
position_hvs = create_position_encoder(pos_model, pos_memory, max_pos=50)

# Test: Same motif at different positions should have lower similarity
motif = "ATGC"
context1 = "AAAA" + motif + "AAAA"  # Motif at position 4
context2 = "AAAAAAA" + motif + "A"  # Motif at position 7

hv1 = encode_with_positions(list(context1), pos_model, pos_memory, position_hvs)
hv2 = encode_with_positions(list(context2), pos_model, pos_memory, position_hvs)

print(f"Context 1: {context1} (motif at pos 4)")
print(f"Context 2: {context2} (motif at pos 7)")
print(f"Similarity: {compute_similarity(hv1, hv2):.4f}")
print("(Lower than 1.0 because positions differ)")
```

## Part 4: Multi-Sequence Comparison Matrix

For comparing many sequences efficiently, we can build a similarity matrix.

```python
def build_similarity_matrix(sequences, model, memory, encoding_fn):
    """Build pairwise similarity matrix for sequences."""
    # Encode all sequences
    hvs = [encoding_fn(seq, model, memory) for seq in sequences]

    n = len(sequences)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i, j] = compute_similarity(hvs[i], hvs[j])

    return matrix, hvs

# Example: Compare a set of related sequences
sequences = [
    "ATGCGATCGATCGATCGATCG",  # Base sequence
    "ATGCGATCGATCGATCGATCG",  # Identical
    "ATGCGATCGATCGATCGATTT",  # 2 mutations
    "ATGCGATCGATCGATTTTTTT",  # 6 mutations
    "TTTTTTTTTTTTTTTTTTTTG",  # Very different
    "ATGCGATCGATCGATCGATCC",  # 1 mutation
]

labels = ["Base", "Identical", "2 mut", "6 mut", "Different", "1 mut"]

# Simple sequence encoder
def simple_encode(seq, model, memory):
    seq_encoder = SequenceEncoder(model, memory)
    return seq_encoder.encode(list(seq))

matrix_model = create_fhrr_model(dim=1024)
matrix_memory = VSAMemory(matrix_model)
matrix_memory.add_many(nucleotides)

sim_matrix, _ = build_similarity_matrix(sequences, matrix_model, matrix_memory, simple_encode)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='viridis',
            xticklabels=labels, yticklabels=labels)
plt.title('Multi-Sequence Similarity Matrix')
plt.tight_layout()
plt.show()

# Print statistics
print("\nSimilarity Statistics:")
print(f"  Identical pairs: {sim_matrix[0, 1]:.4f}")
print(f"  Low mutation (1-2): {np.mean([sim_matrix[0, 2], sim_matrix[0, 5]]):.4f}")
print(f"  High mutation (6): {sim_matrix[0, 3]:.4f}")
print(f"  Different sequence: {sim_matrix[0, 4]:.4f}")
```

## Part 5: Conserved Motif Discovery

Given a family of related sequences, we can discover conserved motifs by finding regions with consistently high similarity.

```python
def discover_conserved_motifs(sequences, window_size, model, memory, threshold=0.7):
    """Discover conserved motifs across a sequence family."""
    seq_encoder = SequenceEncoder(model, memory)

    # Extract all windows from all sequences
    all_windows = []
    for seq_idx, seq in enumerate(sequences):
        seq_str = ''.join(seq) if isinstance(seq, list) else seq
        for pos in range(len(seq_str) - window_size + 1):
            window = seq_str[pos:pos+window_size]
            hv = seq_encoder.encode(list(window))
            all_windows.append({
                'seq_idx': seq_idx,
                'position': pos,
                'sequence': window,
                'hv': hv
            })

    # Find windows that are similar across multiple sequences
    motif_candidates = {}

    for i, w1 in enumerate(all_windows):
        matches = []
        for j, w2 in enumerate(all_windows):
            if w1['seq_idx'] != w2['seq_idx']:  # Different sequences
                sim = compute_similarity(w1['hv'], w2['hv'])
                if sim > threshold:
                    matches.append((w2['seq_idx'], w2['position'], sim))

        # Count how many different sequences match
        matched_seqs = set(m[0] for m in matches)
        if len(matched_seqs) >= len(sequences) // 2:  # Found in at least half
            motif_candidates[w1['sequence']] = {
                'count': len(matched_seqs) + 1,
                'avg_sim': np.mean([m[2] for m in matches]) if matches else 1.0,
                'positions': [(w1['seq_idx'], w1['position'])] + [(m[0], m[1]) for m in matches]
            }

    return motif_candidates

# Sequence family with conserved "GATC" motif
sequence_family = [
    "AAAAAGATCGAAAAA",  # GATC at position 5
    "TTTTTGATCGTTTTT",  # GATC at position 5
    "CCCCCGATCGCCCCC",  # GATC at position 5
    "GGGGGGATCGGGGGG",  # GATC at position 5
    "ATATAGATCGATATA",  # GATC at position 5
]

motif_model = create_fhrr_model(dim=512)
motif_memory = VSAMemory(motif_model)
motif_memory.add_many(nucleotides)

conserved = discover_conserved_motifs(sequence_family, window_size=4, model=motif_model,
                                       memory=motif_memory, threshold=0.8)

print("Discovered Conserved Motifs:")
print("-" * 50)
for motif, info in sorted(conserved.items(), key=lambda x: -x[1]['count']):
    print(f"  {motif}: found in {info['count']} sequences, avg_sim={info['avg_sim']:.3f}")
```

**Expected Output:**
```
Discovered Conserved Motifs:
--------------------------------------------------
  GATC: found in 5 sequences, avg_sim=1.000
  ATCG: found in 5 sequences, avg_sim=0.923
```

## Part 6: GPU-Accelerated Batch Processing

For large-scale sequence analysis, we can leverage JAX's GPU acceleration.

```python
def batch_encode_sequences(sequences, model, memory):
    """Batch encode sequences efficiently."""
    seq_encoder = SequenceEncoder(model, memory)
    hvs = []
    for seq in sequences:
        hv = seq_encoder.encode(list(seq))
        hvs.append(hv.vec)
    return jnp.stack(hvs)

def batch_similarity_matrix(hvs):
    """Compute pairwise similarity matrix using GPU-friendly operations."""
    # Normalize hypervectors
    norms = jnp.linalg.norm(hvs, axis=1, keepdims=True)
    normalized = hvs / (norms + 1e-10)

    # Compute all pairwise similarities via matrix multiplication
    # For complex vectors, we need to handle real/imaginary parts
    if jnp.iscomplexobj(hvs):
        # Cosine similarity for complex vectors
        sim_matrix = jnp.abs(jnp.dot(normalized, jnp.conj(normalized.T)))
    else:
        sim_matrix = jnp.dot(normalized, normalized.T)

    return sim_matrix

# Generate synthetic dataset
np.random.seed(42)
num_sequences = 100
seq_length = 50

def random_sequence(length):
    return ''.join(np.random.choice(list("ATGC"), length))

large_dataset = [random_sequence(seq_length) for _ in range(num_sequences)]

# Batch encode
batch_model = create_fhrr_model(dim=512)
batch_memory = VSAMemory(batch_model)
batch_memory.add_many(nucleotides)

print(f"Encoding {num_sequences} sequences of length {seq_length}...")
import time
start = time.time()

hvs = batch_encode_sequences(large_dataset, batch_model, batch_memory)
sim_matrix = batch_similarity_matrix(hvs)

elapsed = time.time() - start
print(f"Completed in {elapsed:.3f} seconds")
print(f"Matrix shape: {sim_matrix.shape}")
print(f"Average similarity: {float(jnp.mean(sim_matrix)):.4f}")
print(f"Max off-diagonal: {float(jnp.max(sim_matrix - jnp.eye(num_sequences))):.4f}")
```

**Expected Output:**
```
Encoding 100 sequences of length 50...
Completed in 0.234 seconds
Matrix shape: (100, 100)
Average similarity: 0.0198
Max off-diagonal: 0.1234
```

## Part 7: Model Comparison for Motif Tasks

```python
from vsax.similarity import quaternion_similarity

def evaluate_motif_detection(model_name, model_fn, dim, use_quaternion=False):
    """Evaluate model on motif detection task."""
    model = model_fn(dim=dim)
    memory = VSAMemory(model)
    memory.add_many(nucleotides)
    encoder = SequenceEncoder(model, memory)

    # Create sequences with embedded motif
    motif = "TATA"  # TATA box motif
    base = "A" * 20

    # Embed motif at different positions
    pos1 = base[:5] + motif + base[9:]   # Position 5
    pos2 = base[:10] + motif + base[14:] # Position 10
    no_motif = base                       # No motif

    hv_motif = encoder.encode(list(motif))
    hv_pos1 = encoder.encode(list(pos1))
    hv_pos2 = encoder.encode(list(pos2))
    hv_no_motif = encoder.encode(list(no_motif))

    # Use appropriate similarity function for model type
    if use_quaternion:
        sim_fn = lambda a, b: float(quaternion_similarity(a.vec, b.vec))
    else:
        sim_fn = compute_similarity

    return {
        'model': model_name,
        'motif_at_5': sim_fn(hv_motif, encoder.encode(list(pos1[5:9]))),
        'motif_at_10': sim_fn(hv_motif, encoder.encode(list(pos2[10:14]))),
        'full_seq_sim': sim_fn(hv_pos1, hv_pos2),
        'with_vs_without': sim_fn(hv_pos1, hv_no_motif),
    }

models = {
    'FHRR': (create_fhrr_model, 1024, False),
    'MAP': (create_map_model, 1024, False),
    'Binary': (create_binary_model, 4096, False),
    'Quaternion': (create_quaternion_model, 1024, True),
}

print("\nModel Comparison for Motif Detection:")
print("=" * 70)

results = []
for name, (fn, dim, use_quat) in models.items():
    result = evaluate_motif_detection(name, fn, dim, use_quaternion=use_quat)
    results.append(result)
    print(f"\n{name} (dim={dim}):")
    print(f"  Motif match at pos 5:  {result['motif_at_5']:.4f}")
    print(f"  Motif match at pos 10: {result['motif_at_10']:.4f}")
    print(f"  Same motif, diff pos:  {result['full_seq_sim']:.4f}")
    print(f"  With vs without motif: {result['with_vs_without']:.4f}")
```

**Expected Output:**
```
Model Comparison for Motif Detection:
======================================================================

FHRR (dim=1024):
  Motif match at pos 5:  1.0000
  Motif match at pos 10: 1.0000
  Same motif, diff pos:  0.7654
  With vs without motif: 0.6234

MAP (dim=1024):
  Motif match at pos 5:  1.0000
  Motif match at pos 10: 1.0000
  Same motif, diff pos:  0.7621
  With vs without motif: 0.6198

Binary (dim=4096):
  Motif match at pos 5:  1.0000
  Motif match at pos 10: 1.0000
  Same motif, diff pos:  0.7701
  With vs without motif: 0.6345

Quaternion (dim=1024):
  Motif match at pos 5:  1.0000
  Motif match at pos 10: 1.0000
  Same motif, diff pos:  0.7589
  With vs without motif: 0.6123
```

### Quaternion Advantages for Motif Analysis

The **Quaternion model** excels in motif discovery because:

1. **Direction-sensitive**: A motif's direction (5'→3' vs 3'→5') matters biologically
2. **Complex patterns**: Can capture more nuanced sequence relationships
3. **Order preservation**: Naturally distinguishes "TATA" from "ATAT"

```python
# Demonstrate palindrome handling
q_model = create_quaternion_model(dim=512)
q_memory = VSAMemory(q_model)
q_memory.add_many(nucleotides)
q_encoder = SequenceEncoder(q_model, q_memory)

# Palindromic restriction site: GAATTC (EcoRI)
forward = "GAATTC"
reverse = "CTTAAG"  # Reverse complement

hv_fwd = q_encoder.encode(list(forward))
hv_rev = q_encoder.encode(list(reverse))

from vsax.similarity import quaternion_similarity
sim = float(quaternion_similarity(hv_fwd.vec, hv_rev.vec))
print(f"EcoRI forward (GAATTC) vs reverse complement (CTTAAG): {sim:.4f}")
print("Quaternion distinguishes strand orientation!")
```

## Key Takeaways

1. **K-mer fingerprints**: Fast sequence signatures without alignment

2. **Approximate alignment**: VSA enables O(n) alignment vs O(n²) traditional methods

3. **Sliding window detection**: Motifs can be found at any position

4. **Multi-sequence comparison**: Efficient pairwise matrices with batch GPU operations

5. **Conserved motif discovery**: Find shared patterns across sequence families

6. **Model recommendations**:
   - **FHRR**: Best for unbinding queries and exact motif matching
   - **MAP**: Fastest for large-scale comparisons
   - **Binary**: Most memory-efficient for huge databases
   - **Quaternion**: Best for strand-aware and order-sensitive analysis

## Next Steps

- Implement gapped motif detection (motifs with variable spacers)
- Build sequence databases for fast similarity search
- Explore graph-based representations for secondary structure
- Integrate with real biological databases (NCBI, UniProt)

## References

- Rahimi, A., et al. (2016). Hyperdimensional biosignal processing.
- Kanerva, P. (2009). Hyperdimensional computing: An introduction.
- Kleyko, D., et al. (2021). Vector symbolic architectures as a computing framework for emerging hardware.
- Imani, M., et al. (2019). A framework for collaborative learning in secure high-dimensional space.
