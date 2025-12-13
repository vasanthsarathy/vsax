# VSAX Development Plan - Agile Iterative Approach

## Overview
Build a JAX-native VSA library implementing all three VSA models (FHRR, MAP, Binary) in parallel. Each iteration delivers a fully tested, "complete thing" with occasional PyPI releases.

**Total Timeline:** ~3-4 weeks to v1.0.0
**Testing Standard:** ≥80% coverage per iteration
**Approach:** Horizontal development (features across all models) not vertical (one model at a time)

---

## Iteration 1: Foundation & Infrastructure (1-2 days)
**Version:** 0.1.0-dev
**PyPI:** No public release
**Goal:** Complete development infrastructure

### Files to Create
```
pyproject.toml                     # Package config with dependencies
.gitignore                         # Python, JAX, IDE patterns
pytest.ini                         # Test configuration
mkdocs.yml                         # Documentation site
LICENSE, CONTRIBUTING.md, CHANGELOG.md

vsax/
├── __init__.py
├── py.typed
├── core/
│   ├── __init__.py
│   ├── base.py                   # AbstractHypervector, AbstractOpSet
│   └── model.py                  # VSAModel dataclass
├── representations/__init__.py
├── ops/__init__.py
├── sampling/__init__.py
├── encoders/__init__.py
├── similarity/__init__.py
├── io/__init__.py
└── utils/__init__.py

tests/
├── __init__.py
├── conftest.py
├── test_infrastructure.py
└── core/__init__.py

.github/workflows/
├── ci.yml                         # Test on push/PR
└── publish.yml                    # PyPI publishing

docs/
├── index.md
├── getting-started.md
└── api/index.md
```

### Key Implementations

**vsax/core/base.py** - Abstract base classes:
- `AbstractHypervector(ABC)` - wraps `jnp.ndarray` with `.vec`, `.normalize()`, `.to_numpy()`
- `AbstractOpSet(ABC)` - defines `bind()`, `bundle()`, `inverse()`, `permute()`

**vsax/core/model.py** - VSAModel:
- `@dataclass(frozen=True)` with `dim`, `rep_cls`, `opset`, `sampler`
- Immutable container for algebra definition

**pyproject.toml** dependencies:
- Core: `jax>=0.4.20`, `jaxlib>=0.4.20`, `numpy>=1.24.0`
- Dev: `pytest`, `pytest-cov`, `ruff`, `mypy`
- Docs: `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`

### Completion Criteria
- [x] Package installs with `uv pip install -e .`
- [x] CI pipeline configured
- [x] Abstract base classes importable
- [x] Test infrastructure complete with 80%+ coverage
- [x] All tests pass (36 tests)
- [x] Type checking passes (mypy)
- [x] Linting passes (ruff)
- [x] Documentation builds successfully

---

## Iteration 2: All 3 Representations + All 3 OpSets (3-4 days)
**Version:** 0.2.0
**PyPI:** Yes - "Core Algebras"
**Goal:** Complete parallel implementation of all VSA components

### Files to Create
```
vsax/
├── representations/
│   ├── __init__.py
│   ├── complex_hv.py             # ComplexHypervector (FHRR)
│   ├── real_hv.py                # RealHypervector (MAP)
│   └── binary_hv.py              # BinaryHypervector
├── ops/
│   ├── __init__.py
│   ├── fhrr.py                   # FFT-based circular convolution
│   ├── map.py                    # Element-wise multiply/mean
│   └── binary.py                 # XOR/majority
└── sampling/
    ├── __init__.py
    ├── random.py                 # sample_random, sample_complex_random, sample_binary_random
    └── circular.py               # sample_circular

tests/
├── representations/
│   ├── test_complex_hv.py
│   ├── test_real_hv.py
│   └── test_binary_hv.py
├── ops/
│   ├── test_fhrr.py
│   ├── test_map.py
│   └── test_binary.py
├── sampling/test_samplers.py
└── test_model_combinations.py    # Cross-test all combinations
```

### Key Implementations

**ComplexHypervector:**
- Phase-based encoding with complex arrays
- `.normalize()` → unit magnitude (phase-only)
- `.phase` property

**FHRROperations:**
- `bind()` → `jnp.fft.ifft(jnp.fft.fft(a) * jnp.fft.fft(b))`
- `bundle()` → sum and normalize
- `inverse()` → complex conjugate

**RealHypervector:**
- Continuous real-valued vectors
- L2 normalization

**MAPOperations:**
- `bind()` → element-wise multiplication
- `bundle()` → element-wise mean
- `inverse()` → approximate via normalization

**BinaryHypervector:**
- Bipolar {-1, +1} or {0, 1}
- Validation of binary values

**BinaryOperations:**
- `bind()` → XOR (multiplication in bipolar)
- `bundle()` → majority vote
- `inverse()` → self-inverse

### Completion Criteria
- [ ] All 3 representations working
- [ ] All 3 operation sets working
- [ ] Samplers for all types
- [ ] Can create VSAModel for each algebra
- [ ] ≥80% coverage

---

## Iteration 3: VSAModel + VSAMemory + Integration (2-3 days)
**Version:** 0.3.0
**PyPI:** Yes - "Models and Memory"
**Goal:** Working model instances and symbol storage

### Files to Create
```
vsax/
├── core/
│   ├── memory.py                 # VSAMemory class
│   └── factory.py                # create_fhrr_model, create_map_model, create_binary_model
└── utils/
    ├── coerce.py                 # coerce_to_array, coerce_many
    └── validation.py

tests/
├── core/
│   ├── test_memory.py
│   └── test_factory.py
└── test_integration.py           # End-to-end for all 3 models
```

### Key Implementations

**VSAMemory:**
- `__init__(model: VSAModel)`
- `add(name: str)` → create and store hypervector
- `add_many(names: List[str])`
- `get(name: str)`, `__getitem__`, `__contains__`
- `keys()`, `clear()`
- Dictionary-style access

**Factory functions:**
- `create_fhrr_model(dim)` → VSAModel with ComplexHypervector + FHRROperations
- `create_map_model(dim)` → VSAModel with RealHypervector + MAPOperations
- `create_binary_model(dim, bipolar=True)` → VSAModel with BinaryHypervector + BinaryOperations

### Completion Criteria
- [ ] VSAMemory works with all 3 models
- [ ] Factory functions provide easy model creation
- [ ] End-to-end workflows tested
- [ ] ≥80% coverage

---

## Iteration 4: Encoders + Working Examples (3-4 days) 🎯 FIRST USABLE
**Version:** 0.4.0
**PyPI:** Yes - "First Usable Release"
**Goal:** Users can encode structured data types (scalars, sequences, sets, dicts, graphs) + custom encoders

### Files to Create
```
vsax/encoders/
├── __init__.py
├── base.py                       # AbstractEncoder
├── scalar.py                     # ScalarEncoder (power encoding)
├── sequence.py                   # SequenceEncoder (lists, tuples)
├── set.py                        # SetEncoder (unordered collections)
├── dict.py                       # DictEncoder (role-filler binding)
└── graph.py                      # GraphEncoder (nodes + edges)

examples/
├── basic_binding.py              # Simple bind operation
├── scalar_encoding.py            # Encode temperature=23.5
├── sequence_encoding.py          # Encode ["red", "green", "blue"]
├── set_encoding.py               # Encode {"dog", "cat", "bird"}
├── dict_encoding.py              # Encode {"subject": "dog", "action": "run"}
├── graph_encoding.py             # Encode graph structures
├── custom_encoder.py             # Example custom encoder
├── fhrr_example.py               # Complete FHRR workflow
├── map_example.py                # Complete MAP workflow
└── binary_example.py             # Complete Binary workflow

tests/encoders/
├── test_scalar.py
├── test_sequence.py
├── test_set.py
├── test_dict.py
├── test_graph.py
└── test_custom.py

docs/guide/encoders/
├── overview.md                   # Encoders overview
├── scalar.md
├── sequence.md
├── set.md
├── dict.md
├── graph.md
└── custom.md                     # How to write custom encoders
```

### Key Implementations

**AbstractEncoder (base class):**
- `__init__(model, memory)` - accepts VSAModel and VSAMemory
- `encode(data)` - abstract method for encoding
- Optional: `fit(data)` for learned encodings
- Optional: `decode(hv, candidates)` for decoding

**ScalarEncoder:**
- Encode numeric values (int, float)
- Power encoding: `basis ** value` (for complex)
- Iterated binding for real/binary
- Example: `encode("temperature", 23.5)`

**SequenceEncoder:**
- Encode ordered sequences (list, tuple)
- Positional binding: `bundle(bind(pos0, item0), bind(pos1, item1), ...)`
- Preserves order information
- Example: `encode(["red", "green", "blue"])`

**SetEncoder:**
- Encode unordered collections (set)
- Simple bundling (no position info)
- Order-invariant
- Example: `encode({"dog", "cat", "bird"})`

**DictEncoder:**
- Encode key-value mappings (dict)
- Role-filler binding: `bundle(bind(key1, val1), bind(key2, val2), ...)`
- Example: `encode({"subject": "dog", "action": "run"})`

**GraphEncoder:**
- Encode graph structures
- Nodes as bundled hypervectors
- Edges as `bind(source, bind(relation, target))`
- Supports directed/undirected graphs
- Example: Encode social networks, knowledge graphs

**Custom Encoder Support:**
- Users can subclass AbstractEncoder
- Template and examples provided
- Documented best practices

### Implementation Plan

#### Phase 1: Base Infrastructure ✅
- [ ] Create `vsax/encoders/__init__.py`
- [ ] Implement `AbstractEncoder` base class
- [ ] Write tests for abstract interface
- [ ] Update main `__init__.py` exports

#### Phase 2: ScalarEncoder ✅
- [ ] Implement ScalarEncoder
- [ ] Support all 3 models (FHRR, MAP, Binary)
- [ ] Write comprehensive tests
- [ ] Create example + documentation

#### Phase 3: SequenceEncoder ✅
- [ ] Implement SequenceEncoder
- [ ] Support lists and tuples
- [ ] Positional binding strategy
- [ ] Tests + example + docs

#### Phase 4: SetEncoder ✅
- [ ] Implement SetEncoder
- [ ] Verify order-invariance
- [ ] Tests + example + docs

#### Phase 5: DictEncoder ✅
- [ ] Implement DictEncoder
- [ ] Role-filler binding
- [ ] Tests + example + docs

#### Phase 6: GraphEncoder ✅
- [ ] Implement GraphEncoder
- [ ] Support directed/undirected
- [ ] Tests + example + docs

#### Phase 7: Custom Encoder Support ✅
- [ ] Create custom encoder template
- [ ] Write guide + examples
- [ ] Document best practices

#### Phase 8: Integration Examples ✅
- [ ] basic_binding.py
- [ ] fhrr_example.py
- [ ] map_example.py
- [ ] binary_example.py

#### Phase 9: Documentation ✅
- [ ] Update main documentation
- [ ] API reference for all encoders
- [ ] Update mkdocs navigation

#### Phase 10: Release ✅
- [ ] ≥80% test coverage
- [ ] Update CHANGELOG.md
- [ ] Publish v0.4.0 to PyPI

### Design Decisions to Finalize

1. **Scalar encoding range**: What range? Normalized 0-1? User-specified?
2. **Sequence length**: Fixed or dynamic max length?
3. **Graph format**: NetworkX, edge lists, or both?
4. **Decoding**: Implement decode() or defer to similarity search?
5. **Batch encoding**: Include in v0.4.0 or defer?

### Completion Criteria
- [ ] All 5 core encoders implemented (Scalar, Sequence, Set, Dict, Graph)
- [ ] Custom encoder support with examples
- [ ] Encoders work with all 3 models
- [ ] Integration examples for each model
- [ ] ≥80% coverage
- [ ] Full documentation
- [ ] **FIRST USABLE MILESTONE** 🚀

---

## Iteration 5: Similarity Metrics & Utilities (2 days) ✅ COMPLETE
**Version:** 0.5.0
**PyPI:** Yes - "Similarity and Utilities"
**Goal:** Query and compare hypervectors

### Files to Create
```
vsax/
├── similarity/
│   ├── __init__.py
│   ├── cosine.py
│   ├── dot.py
│   └── hamming.py
└── utils/
    ├── batch.py                  # vmap_bind, vmap_bundle
    └── repr.py                   # Pretty printing

examples/
├── similarity_search.py
└── batch_operations.py

tests/similarity/
└── test_cosine.py, test_dot.py, test_hamming.py
```

### Key Implementations

**Similarity metrics:**
- `cosine_similarity(a, b)` → handles complex vectors
- `dot_similarity(a, b)`
- `hamming_similarity(a, b)` → for binary vectors

**Batch operations:**
- `vmap_bind(opset, X, Y)` → JAX vectorized binding
- `vmap_bundle(opset, X)`

### Completion Criteria
- ✅ All 3 similarity metrics working
- ✅ Batch ops support JAX vmap (vmap_bind, vmap_bundle, vmap_similarity)
- ✅ Accepts both arrays and hypervectors
- ✅ 95%+ coverage (319 tests)
- ✅ Visualization utilities (pretty_repr, format_similarity_results)
- ✅ Complete examples and documentation

---

## Iteration 6: I/O & Persistence (2 days)
**Version:** 0.6.0
**PyPI:** Yes - "I/O and Persistence"
**Goal:** Save and load basis vectors

### Files to Create
```
vsax/io/
├── __init__.py
├── save.py                       # save_basis(memory, path)
└── load.py                       # load_basis(memory, path)

examples/persistence.py

tests/io/
├── test_save_load.py
└── fixtures/sample_basis.json
```

### Key Implementations

**save_basis:**
- JSON serialization of memory
- Handle complex (split real/imag) and real vectors
- Store dim, rep_type metadata

**load_basis:**
- Validate dimension and rep_type match
- Reconstruct hypervectors
- Populate memory

### Completion Criteria
- [ ] Round-trip save/load for all 3 models
- [ ] Human-readable JSON format
- [ ] Validation prevents mismatched loads
- [ ] ≥80% coverage

---

## Iteration 7: Full Documentation & v1.0.0 (3-4 days) 🚀 STABLE RELEASE
**Version:** 1.0.0
**PyPI:** Yes - "Production Release"
**Goal:** Production-ready with comprehensive docs

### Files to Create/Update
```
docs/
├── index.md                      # Updated landing page
├── getting-started.md
├── user-guide/
│   ├── models.md
│   ├── representations.md
│   ├── operations.md
│   ├── encoders.md
│   ├── similarity.md
│   └── persistence.md
├── api/                          # Auto-generated API docs
│   ├── core.md
│   ├── representations.md
│   ├── ops.md
│   ├── encoders.md
│   ├── similarity.md
│   └── io.md
├── examples/
│   ├── fhrr-tutorial.md
│   ├── map-tutorial.md
│   ├── binary-tutorial.md
│   └── advanced.md
└── development/
    ├── contributing.md
    ├── architecture.md
    └── testing.md

README.md                         # Updated with badges, examples
CITATION.cff                      # Citation metadata
```

### Documentation Tasks
1. Complete API docs with mkdocstrings
2. User guides for all features
3. Tutorial notebooks for each model
4. Update README with badges, examples
5. Contributing guidelines
6. Citation metadata

### Final Validation
- All tests pass (≥80% coverage)
- Documentation builds without errors
- All examples execute
- mypy passes
- ruff passes
- Performance benchmarks documented

### Completion Criteria
- [ ] Complete API documentation
- [ ] User guides + tutorials
- [ ] Example notebooks
- [ ] Professional README
- [ ] All tests passing
- [ ] **PRODUCTION READY**

---

## Critical Files (Implementation Order)

1. **vsax/core/base.py** - Abstract base classes (foundation)
2. **vsax/core/model.py** - VSAModel dataclass
3. **vsax/representations/complex_hv.py** - Most sophisticated representation
4. **vsax/ops/fhrr.py** - Most complex operations (validates JAX integration)
5. **vsax/core/memory.py** - Symbol table (critical for usability)

---

## PyPI Release Schedule

| Version | Iteration | Description | Milestone |
|---------|-----------|-------------|-----------|
| 0.1.0-dev | 1 | Infrastructure only | - |
| 0.2.0 | 2 | Core Algebras | All 3 VSA models |
| 0.3.0 | 3 | Models and Memory | Symbol storage |
| **0.4.0** | 4 | **First Usable** | **Encoders + examples** |
| 0.5.0 | 6 | Similarity + I/O | Persistence |
| **1.0.0** | 7 | **Production** | **Stable release** |

---

## Development Workflow (Per Iteration)

1. **Planning** - Review goals, create issues, feature branch
2. **Implementation** - TDD (tests first), implement, iterate
3. **Review** - Coverage check, documentation review, CI validation
4. **Release** - Merge to main, tag, PyPI (if scheduled), update CHANGELOG

---

## Review Section
*This section will be updated with summaries after each iteration is complete.*

### Iteration 1 Review ✅ COMPLETE

**Completion Date:** 2025-12-12

**Summary:**
Successfully established complete development infrastructure for VSAX library. All foundation components are in place for rapid iteration on VSA implementations.

**Files Created (30 total):**
- Core Python modules: `vsax/core/base.py`, `vsax/core/model.py`
- Package structure: 9 `__init__.py` files across all modules
- Configuration: `pyproject.toml`, `.gitignore`, `mkdocs.yml`
- Testing: `tests/test_infrastructure.py`, `tests/conftest.py`
- CI/CD: `.github/workflows/ci.yml`, `.github/workflows/publish.yml`
- Documentation: `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`
- Docs site: `docs/index.md`, `docs/getting-started.md`, `docs/api/index.md`
- Setup scripts: `setup.sh`, `setup.ps1` (for uv)
- Type marker: `vsax/py.typed`

**Key Achievements:**
1. ✅ Abstract base classes (AbstractHypervector, AbstractOpSet) with full type annotations
2. ✅ VSAModel dataclass with validation
3. ✅ Complete package structure with placeholder modules for iterations 2-6
4. ✅ pytest infrastructure with fixtures and mock implementations
5. ✅ GitHub Actions CI pipeline (test, lint, type-check, publish)
6. ✅ MkDocs documentation site with Material theme
7. ✅ Development tooling (ruff, mypy, pytest-cov)
8. ✅ **uv package manager integration** for fast, reliable environment management

**Test Results:**
- Infrastructure tests: Ready to run (requires uv venv setup)
- Expected coverage: 100% (minimal code at this stage)
- All abstract classes properly defined and importable

**Changes from Original Plan:**
- Added uv package manager support (Windows + Unix)
- Created setup.sh and setup.ps1 for one-command environment setup
- Enhanced .gitignore with uv-specific entries

**Lessons Learned:**
- Using `uv` significantly simplifies environment management
- Abstract base classes with comprehensive docstrings improve developer experience
- Setting up full CI/CD from iteration 1 enables continuous quality assurance

**Next Steps (Iteration 2):**
- Implement all three representations (Complex, Real, Binary)
- Implement all three operation sets (FHRR, MAP, Binary)
- Add sampling utilities
- Publish v0.2.0 to PyPI

### Iteration 2 Review
- TBD

### Iteration 3 Review
- TBD

### Iteration 4 Review
- TBD

### Iteration 5 Review
- TBD

### Iteration 6 Review
- TBD

### Iteration 7 Review
- TBD
