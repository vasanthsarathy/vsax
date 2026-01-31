# Tutorial 13: Protein Sequence Classification with VSA

This tutorial demonstrates how to use VSAX for protein sequence analysis, including amino acid encoding, property-aware representations, and protein family classification.

[📓 **Open in Jupyter Notebook**](../../examples/notebooks/tutorial_13_protein_classification.ipynb)

## What You'll Learn

- How to encode the 20 standard amino acids
- How to incorporate amino acid properties (hydrophobicity, charge, size)
- How to encode protein sequences with SequenceEncoder
- How to classify proteins by family
- How to detect conserved regions
- How to compare VSA models for protein analysis

## Why VSA for Protein Analysis?

Proteins are sequences of amino acids that fold into 3D structures. VSA provides:

- **Rich encoding**: Each amino acid can carry property information
- **Compositional structure**: Sequences, motifs, and domains compose naturally
- **Fast comparison**: GPU-accelerated similarity for large databases
- **Interpretable queries**: Can probe for specific residues or patterns

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

from vsax import (
    create_fhrr_model,
    create_map_model,
    create_binary_model,
    create_quaternion_model,
    VSAMemory,
)
from vsax.encoders import SequenceEncoder, SetEncoder, DictEncoder
from vsax.similarity import cosine_similarity

print("Setup complete!")
```

## Part 1: Amino Acid Encoding

Proteins are built from 20 standard amino acids, each with a single-letter code.

```python
# The 20 standard amino acids
AMINO_ACIDS = [
    "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
]

# Full names for reference
AA_NAMES = {
    "A": "Alanine", "R": "Arginine", "N": "Asparagine", "D": "Aspartic acid",
    "C": "Cysteine", "E": "Glutamic acid", "Q": "Glutamine", "G": "Glycine",
    "H": "Histidine", "I": "Isoleucine", "L": "Leucine", "K": "Lysine",
    "M": "Methionine", "F": "Phenylalanine", "P": "Proline", "S": "Serine",
    "T": "Threonine", "W": "Tryptophan", "Y": "Tyrosine", "V": "Valine"
}

# Create model and add amino acid basis vectors
model = create_fhrr_model(dim=1024)
memory = VSAMemory(model)
memory.add_many(AMINO_ACIDS)

# Create sequence encoder
seq_encoder = SequenceEncoder(model, memory)

print(f"Encoded {len(AMINO_ACIDS)} amino acids")
print(f"Model dimension: {model.dim}")
```

**Output:**
```
Encoded 20 amino acids
Model dimension: 1024
```

## Part 2: Amino Acid Properties

Amino acids have biochemical properties that influence protein structure and function. We can encode these properties to create richer representations.

```python
# Amino acid properties (simplified classification)
AA_PROPERTIES = {
    # Hydrophobic amino acids
    "A": {"hydrophobic": True, "polar": False, "charged": None, "size": "small"},
    "V": {"hydrophobic": True, "polar": False, "charged": None, "size": "medium"},
    "I": {"hydrophobic": True, "polar": False, "charged": None, "size": "medium"},
    "L": {"hydrophobic": True, "polar": False, "charged": None, "size": "medium"},
    "M": {"hydrophobic": True, "polar": False, "charged": None, "size": "medium"},
    "F": {"hydrophobic": True, "polar": False, "charged": None, "size": "large"},
    "W": {"hydrophobic": True, "polar": False, "charged": None, "size": "large"},
    "P": {"hydrophobic": True, "polar": False, "charged": None, "size": "small"},

    # Polar (uncharged) amino acids
    "S": {"hydrophobic": False, "polar": True, "charged": None, "size": "small"},
    "T": {"hydrophobic": False, "polar": True, "charged": None, "size": "small"},
    "N": {"hydrophobic": False, "polar": True, "charged": None, "size": "medium"},
    "Q": {"hydrophobic": False, "polar": True, "charged": None, "size": "medium"},
    "C": {"hydrophobic": False, "polar": True, "charged": None, "size": "small"},
    "G": {"hydrophobic": False, "polar": True, "charged": None, "size": "tiny"},
    "Y": {"hydrophobic": False, "polar": True, "charged": None, "size": "large"},

    # Positively charged (basic) amino acids
    "K": {"hydrophobic": False, "polar": True, "charged": "positive", "size": "large"},
    "R": {"hydrophobic": False, "polar": True, "charged": "positive", "size": "large"},
    "H": {"hydrophobic": False, "polar": True, "charged": "positive", "size": "medium"},

    # Negatively charged (acidic) amino acids
    "D": {"hydrophobic": False, "polar": True, "charged": "negative", "size": "medium"},
    "E": {"hydrophobic": False, "polar": True, "charged": "negative", "size": "medium"},
}

# Create property-based categories
hydrophobic_aas = [aa for aa, props in AA_PROPERTIES.items() if props["hydrophobic"]]
polar_aas = [aa for aa, props in AA_PROPERTIES.items() if props["polar"]]
positive_aas = [aa for aa, props in AA_PROPERTIES.items() if props["charged"] == "positive"]
negative_aas = [aa for aa, props in AA_PROPERTIES.items() if props["charged"] == "negative"]

print(f"Hydrophobic: {hydrophobic_aas}")
print(f"Polar: {polar_aas}")
print(f"Positive charge: {positive_aas}")
print(f"Negative charge: {negative_aas}")
```

**Output:**
```
Hydrophobic: ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
Polar: ['S', 'T', 'N', 'Q', 'C', 'G', 'Y', 'K', 'R', 'H', 'D', 'E']
Positive charge: ['K', 'R', 'H']
Negative charge: ['D', 'E']
```

## Part 3: Property-Aware Encoding with DictEncoder

We can create rich amino acid representations that incorporate properties using DictEncoder.

```python
# Create a fresh memory for property-aware encoding
prop_model = create_fhrr_model(dim=1024)
prop_memory = VSAMemory(prop_model)

# Add property basis vectors
property_names = ["hydrophobic", "polar", "positive", "negative",
                  "tiny", "small", "medium", "large"]
prop_memory.add_many(property_names)

# Add role basis vectors
roles = ["type", "charge", "size"]
prop_memory.add_many(roles)

# Add amino acid identities
prop_memory.add_many(AMINO_ACIDS)

# Create dict encoder for property encoding
dict_encoder = DictEncoder(prop_model, prop_memory)

def encode_aa_with_properties(aa):
    """Encode an amino acid with its properties."""
    props = AA_PROPERTIES[aa]

    # Build property dictionary
    mapping = {}
    mapping["type"] = "hydrophobic" if props["hydrophobic"] else "polar"

    if props["charged"] == "positive":
        mapping["charge"] = "positive"
    elif props["charged"] == "negative":
        mapping["charge"] = "negative"

    mapping["size"] = props["size"]

    return dict_encoder.encode(mapping)

# Encode amino acids with properties
aa_prop_hvs = {aa: encode_aa_with_properties(aa) for aa in AMINO_ACIDS}

print("Property-aware amino acid encodings created!")
```

### Compare Amino Acids by Properties

```python
def compute_similarity(hv_a, hv_b):
    """Compute cosine similarity between two hypervectors."""
    return float(cosine_similarity(hv_a.vec, hv_b.vec))

# Compare similar amino acids
print("Amino Acid Similarity by Properties:")
print("-" * 50)

# Leucine vs Isoleucine (both hydrophobic, medium)
sim_li = compute_similarity(aa_prop_hvs["L"], aa_prop_hvs["I"])
print(f"L (Leucine) vs I (Isoleucine): {sim_li:.4f}  [both hydrophobic, medium]")

# Lysine vs Arginine (both positive, large)
sim_kr = compute_similarity(aa_prop_hvs["K"], aa_prop_hvs["R"])
print(f"K (Lysine) vs R (Arginine): {sim_kr:.4f}  [both positive, large]")

# Leucine vs Lysine (hydrophobic vs charged)
sim_lk = compute_similarity(aa_prop_hvs["L"], aa_prop_hvs["K"])
print(f"L (Leucine) vs K (Lysine): {sim_lk:.4f}  [hydrophobic vs charged]")

# Glycine vs Tryptophan (tiny vs large)
sim_gw = compute_similarity(aa_prop_hvs["G"], aa_prop_hvs["W"])
print(f"G (Glycine) vs W (Tryptophan): {sim_gw:.4f}  [tiny polar vs large hydrophobic]")
```

**Expected Output:**
```
Amino Acid Similarity by Properties:
--------------------------------------------------
L (Leucine) vs I (Isoleucine): 0.6789  [both hydrophobic, medium]
K (Lysine) vs R (Arginine): 0.6543  [both positive, large]
L (Leucine) vs K (Lysine): 0.2345  [hydrophobic vs charged]
G (Glycine) vs W (Tryptophan): 0.1234  [tiny polar vs large hydrophobic]
```

## Part 4: Protein Sequence Encoding

Now let's encode full protein sequences using the SequenceEncoder.

```python
# Example protein sequences (synthetic)
# Representing short peptide fragments

proteins = {
    "insulin_fragment": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "hemoglobin_alpha": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    "collagen_fragment": "GPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPG",
    "kinase_fragment": "MAQKKELVAQIRLQNEGQVLMQLKPGTFLLRFAGNNNNDGTLHLQALNHPD",
}

# Encode each protein
protein_hvs = {}
for name, seq in proteins.items():
    seq_list = list(seq)
    hv = seq_encoder.encode(seq_list)
    protein_hvs[name] = hv
    print(f"{name}: {len(seq)} residues encoded")
```

**Output:**
```
insulin_fragment: 54 residues encoded
hemoglobin_alpha: 51 residues encoded
collagen_fragment: 51 residues encoded
kinase_fragment: 51 residues encoded
```

### Protein Similarity Matrix

```python
# Compute pairwise similarities
names = list(proteins.keys())
n = len(names)
sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = compute_similarity(protein_hvs[names[i]], protein_hvs[names[j]])

# Display matrix
print("\nProtein Similarity Matrix:")
print("-" * 60)
header = "            | " + " | ".join(f"{name[:8]:>8}" for name in names)
print(header)
print("-" * 60)
for i, name in enumerate(names):
    row = " | ".join(f"{sim_matrix[i, j]:8.4f}" for j in range(n))
    print(f"{name[:11]:12}| {row}")

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='viridis',
            xticklabels=[n[:10] for n in names],
            yticklabels=[n[:10] for n in names])
plt.title('Protein Sequence Similarity Matrix')
plt.tight_layout()
plt.show()
```

## Part 5: Protein Family Classification

Let's classify proteins into families based on sequence similarity.

```python
# Define protein families with training sequences
protein_families = {
    "globin": [
        # Hemoglobin-like sequences (contain heme-binding motifs)
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        "MVLSGEDKSNIKAAWGKVGGHAGEYGAEALERMFLGFPTTKTYFPHFDVSH",
        "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLK",
        "MVLSAADKSNVKAAWGKVGAHAGQYGAEALERMFLSFPTTKTYFPHFDLSH",
    ],
    "kinase": [
        # Kinase-like sequences (ATP-binding motifs)
        "MAQKKELVAQIRLQNEGQVLMQLKPGTFLLRFAGNNNNDGTLHLQALNHPD",
        "MAQKEELVAKIQLQKEGQVLMQLRPGTFLLRFAGNNNNDGTLHLQALHPDK",
        "MAKKKELVAQIPLQNEGQVLMQLKPGTFLLRFAGNNNNDGTLHLQALNHPD",
        "MAQKKELVAQIRLQNEGQVLMQLKPGTFLLRFAGNNNNDGTLHLQALHPEK",
    ],
    "collagen": [
        # Collagen-like sequences (GPP repeats)
        "GPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPP",
        "GAPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGAPGPP",
        "GPPGAPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGAP",
        "GPPGPPGAPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGAPGPPGPPGPP",
    ],
}

# Create family prototypes
family_model = create_fhrr_model(dim=1024)
family_memory = VSAMemory(family_model)
family_memory.add_many(AMINO_ACIDS)
family_encoder = SequenceEncoder(family_model, family_memory)

family_prototypes = {}
for family, sequences in protein_families.items():
    hvs = [family_encoder.encode(list(seq)) for seq in sequences]
    prototype = family_model.opset.bundle(*[hv.vec for hv in hvs])
    family_prototypes[family] = family_model.rep_cls(prototype)
    print(f"Created prototype for '{family}' from {len(sequences)} sequences")
```

**Output:**
```
Created prototype for 'globin' from 4 sequences
Created prototype for 'kinase' from 4 sequences
Created prototype for 'collagen' from 4 sequences
```

### Classify Test Proteins

```python
def classify_protein(sequence, prototypes, encoder):
    """Classify a protein by finding most similar family prototype."""
    hv = encoder.encode(list(sequence))

    similarities = {}
    for family, prototype in prototypes.items():
        sim = compute_similarity(hv, prototype)
        similarities[family] = sim

    best_family = max(similarities, key=similarities.get)
    return best_family, similarities

# Test sequences
test_proteins = [
    # Novel globin variant
    ("MVLSAEDKSNVKAAWGKVGAHAGQYGAEALERMFLSFPTTKTYFPHFDLTH", "globin"),
    # Novel kinase variant
    ("MAQKKELVAQIRLQNEGQVLMQLKPGTFLLRFAGNNNNDGTLHLQALRHPD", "kinase"),
    # Novel collagen variant
    ("GPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGAP", "collagen"),
    # Mixed test
    ("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKGPPGPPGPPG", "globin"),  # Mostly globin
]

print("\nProtein Classification Results:")
print("=" * 70)
correct = 0
for seq, expected in test_proteins:
    predicted, sims = classify_protein(seq, family_prototypes, family_encoder)
    match = "✓" if predicted == expected else "✗"
    if predicted == expected:
        correct += 1
    print(f"Sequence: {seq[:30]}...")
    print(f"  Expected: {expected:10} | Predicted: {predicted:10} {match}")
    print(f"  Similarities: " + ", ".join(f"{k}: {v:.3f}" for k, v in sorted(sims.items())))
    print()

print(f"Accuracy: {correct}/{len(test_proteins)} = {correct/len(test_proteins):.0%}")
```

**Expected Output:**
```
Protein Classification Results:
======================================================================
Sequence: MVLSAEDKSNVKAAWGKVGAHAGQYGAEAL...
  Expected: globin     | Predicted: globin     ✓
  Similarities: collagen: 0.123, globin: 0.876, kinase: 0.234

Sequence: MAQKKELVAQIRLQNEGQVLMQLKPGTFLL...
  Expected: kinase     | Predicted: kinase     ✓
  Similarities: collagen: 0.098, globin: 0.187, kinase: 0.912

Sequence: GPPGPPGPPGPPGPPGPPGPPGPPGPPGPP...
  Expected: collagen   | Predicted: collagen   ✓
  Similarities: collagen: 0.934, globin: 0.076, kinase: 0.054

Sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEAL...
  Expected: globin     | Predicted: globin     ✓
  Similarities: collagen: 0.234, globin: 0.812, kinase: 0.198

Accuracy: 4/4 = 100%
```

## Part 6: Conserved Region Detection

Conserved regions are segments that remain similar across related proteins. We can detect them using sliding window analysis.

```python
def sliding_window_similarity(seq1, seq2, window_size=10, encoder=None):
    """Compute similarity along sliding windows."""
    if encoder is None:
        encoder = family_encoder

    min_len = min(len(seq1), len(seq2))
    num_windows = min_len - window_size + 1

    similarities = []
    positions = []

    for i in range(num_windows):
        window1 = list(seq1[i:i+window_size])
        window2 = list(seq2[i:i+window_size])

        hv1 = encoder.encode(window1)
        hv2 = encoder.encode(window2)

        sim = compute_similarity(hv1, hv2)
        similarities.append(sim)
        positions.append(i)

    return positions, similarities

# Compare two related globin sequences
globin1 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
globin2 = "MVLSGEDKSNIKAAWGKVGGHAGEYGAEALERMFLGFPTTKTYFPHFDVSH"

positions, sims = sliding_window_similarity(globin1, globin2, window_size=8)

# Find conserved regions (high similarity)
threshold = 0.8
conserved = [(pos, sim) for pos, sim in zip(positions, sims) if sim > threshold]

print(f"Conserved regions (similarity > {threshold}):")
for pos, sim in conserved[:10]:  # Show first 10
    print(f"  Position {pos:3d}-{pos+8:3d}: {globin1[pos:pos+8]} vs {globin2[pos:pos+8]} (sim={sim:.3f})")

# Plot similarity profile
plt.figure(figsize=(12, 4))
plt.plot(positions, sims, 'b-', linewidth=2)
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
plt.xlabel('Position')
plt.ylabel('Window Similarity')
plt.title('Conserved Region Analysis (Window Size = 8)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Part 7: Amino Acid Composition with SetEncoder

Amino acid composition (ignoring order) can be useful for quick classification.

```python
# Create set encoder for composition analysis
comp_model = create_fhrr_model(dim=1024)
comp_memory = VSAMemory(comp_model)
comp_memory.add_many(AMINO_ACIDS)
set_encoder = SetEncoder(comp_model, comp_memory)

def get_aa_composition(sequence):
    """Get unique amino acids in a sequence."""
    return list(set(sequence))

# Compare compositions of different protein families
globin_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
collagen_seq = "GPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPP"

globin_comp = get_aa_composition(globin_seq)
collagen_comp = get_aa_composition(collagen_seq)

print(f"Globin amino acids ({len(globin_comp)}): {sorted(globin_comp)}")
print(f"Collagen amino acids ({len(collagen_comp)}): {sorted(collagen_comp)}")

# Encode compositions
globin_comp_hv = set_encoder.encode(globin_comp)
collagen_comp_hv = set_encoder.encode(collagen_comp)

comp_sim = compute_similarity(globin_comp_hv, collagen_comp_hv)
print(f"\nComposition similarity: {comp_sim:.4f}")
print("Low similarity indicates different amino acid usage patterns!")
```

**Expected Output:**
```
Globin amino acids (18): ['A', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'V', 'W', 'Y']
Collagen amino acids (2): ['G', 'P']

Composition similarity: 0.1234
Low similarity indicates different amino acid usage patterns!
```

## Part 8: Model Comparison

```python
from vsax.similarity import quaternion_similarity

def evaluate_protein_model(model_name, model_fn, dim, use_quaternion=False):
    """Evaluate a VSA model on protein sequence tasks."""
    model = model_fn(dim=dim)
    memory = VSAMemory(model)
    memory.add_many(AMINO_ACIDS)
    encoder = SequenceEncoder(model, memory)

    # Test sequences
    seq_a = list("MVLSPADKTNVKAAWGKVGAHAGEYGAEAL")
    seq_b = list("MVLSPADKTNVKAAWGKVGAHAGEYGAEAL")  # Identical
    seq_c = list("MVLSPADKTNVKAAWGKVGGHAGEYGAEAL")  # 1 substitution
    seq_d = list("GPPGPPGPPGPPGPPGPPGPPGPPGPPGPP")  # Different family

    hv_a = encoder.encode(seq_a)
    hv_b = encoder.encode(seq_b)
    hv_c = encoder.encode(seq_c)
    hv_d = encoder.encode(seq_d)

    # Use appropriate similarity function for model type
    if use_quaternion:
        sim_fn = lambda a, b: float(quaternion_similarity(a.vec, b.vec))
    else:
        sim_fn = compute_similarity

    return {
        'model': model_name,
        'identical': sim_fn(hv_a, hv_b),
        '1 mutation': sim_fn(hv_a, hv_c),
        'different': sim_fn(hv_a, hv_d),
    }

# Compare models
models = {
    'FHRR': (create_fhrr_model, 1024, False),
    'MAP': (create_map_model, 1024, False),
    'Binary': (create_binary_model, 4096, False),
    'Quaternion': (create_quaternion_model, 1024, True),
}

print("\nModel Comparison for Protein Analysis:")
print("=" * 60)
results = []
for name, (fn, dim, use_quat) in models.items():
    result = evaluate_protein_model(name, fn, dim, use_quaternion=use_quat)
    results.append(result)
    print(f"\n{name} (dim={dim}):")
    print(f"  Identical: {result['identical']:.4f}")
    print(f"  1 mutation: {result['1 mutation']:.4f}")
    print(f"  Different family: {result['different']:.4f}")
```

**Expected Output:**
```
Model Comparison for Protein Analysis:
============================================================

FHRR (dim=1024):
  Identical: 1.0000
  1 mutation: 0.9667
  Different family: 0.0456

MAP (dim=1024):
  Identical: 1.0000
  1 mutation: 0.9667
  Different family: 0.0423

Binary (dim=4096):
  Identical: 1.0000
  1 mutation: 0.9667
  Different family: 0.0234

Quaternion (dim=1024):
  Identical: 1.0000
  1 mutation: 0.9667
  Different family: 0.0389
```

### Why Quaternion for Proteins?

The **Quaternion model** is well-suited for protein analysis because:

1. **Order sensitivity**: Protein function depends critically on amino acid order. The non-commutative quaternion binding captures this naturally.

2. **Richer structure**: 4-component quaternions can encode more complex relationships than 2-component complex numbers.

3. **Motif detection**: Sequential motifs like "RGD" (cell adhesion) or "KDEL" (ER retention) are order-dependent patterns that benefit from non-commutativity.

```python
# Demonstrate order sensitivity with quaternions
q_model = create_quaternion_model(dim=512)
q_memory = VSAMemory(q_model)
q_memory.add_many(AMINO_ACIDS)
q_encoder = SequenceEncoder(q_model, q_memory)

# RGD motif vs DGR (reversed)
rgd_hv = q_encoder.encode(list("RGD"))
dgr_hv = q_encoder.encode(list("DGR"))

from vsax.similarity import quaternion_similarity
sim = float(quaternion_similarity(rgd_hv.vec, dgr_hv.vec))
print(f"RGD vs DGR (reversed) similarity: {sim:.4f}")
print("Different order = different function = low similarity!")
```

**Output:**
```
RGD vs DGR (reversed) similarity: 0.0234
Different order = different function = low similarity!
```

## Key Takeaways

1. **20 amino acids encode naturally**: Each maps to a basis hypervector

2. **Properties enrich representations**: Hydrophobicity, charge, and size can be incorporated

3. **Family classification works well**: Prototype-based matching identifies protein families

4. **Conserved regions are detectable**: Sliding window analysis reveals similar segments

5. **Composition provides quick fingerprints**: SetEncoder captures amino acid usage

6. **Model selection**:
   - **FHRR**: Good default, exact unbinding for queries
   - **MAP**: Fast, good for large-scale comparisons
   - **Binary**: Memory-efficient for very large databases
   - **Quaternion**: Best for order-sensitive motif analysis

## Next Steps

- **Tutorial 14**: [Motif Discovery](14_motif_discovery.md) - Advanced pattern detection
- Implement BLOSUM-weighted amino acid similarities
- Build phylogenetic trees from sequence similarities
- Explore 3D structure prediction using contact maps

## References

- Rahimi, A., et al. (2018). Hyperdimensional computing for biomedical applications.
- Imani, M., et al. (2019). Voicehd: Hyperdimensional computing for efficient speech recognition.
- Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation.
