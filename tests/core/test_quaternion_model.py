"""Integration tests for Quaternion VSA model."""

import jax
import jax.numpy as jnp
import pytest

from vsax import (
    VSAMemory,
    create_quaternion_model,
    quaternion_similarity,
)
from vsax.ops.quaternion import QuaternionOperations
from vsax.representations.quaternion_hv import QuaternionHypervector


class TestCreateQuaternionModel:
    """Test suite for create_quaternion_model factory function."""

    def test_creates_valid_model(self):
        """Test that factory creates a valid model."""
        model = create_quaternion_model(dim=256)

        assert model.dim == 256
        assert model.rep_cls == QuaternionHypervector
        assert isinstance(model.opset, QuaternionOperations)

    def test_default_dimension(self):
        """Test default dimension is 512."""
        model = create_quaternion_model()

        assert model.dim == 512

    def test_custom_dimension(self):
        """Test custom dimension."""
        model = create_quaternion_model(dim=1024)

        assert model.dim == 1024

    def test_sampler_produces_correct_shape(self):
        """Test that sampler produces correct shape."""
        model = create_quaternion_model(dim=256)
        key = jax.random.PRNGKey(42)

        vectors = model.sampler(256, 5, key)

        assert vectors.shape == (5, 256, 4)

    def test_sampler_produces_unit_quaternions(self):
        """Test that sampler produces unit quaternions."""
        model = create_quaternion_model(dim=256)
        key = jax.random.PRNGKey(42)

        vectors = model.sampler(256, 5, key)
        norms = jnp.linalg.norm(vectors, axis=-1)

        assert jnp.allclose(norms, 1.0, atol=1e-6)


class TestQuaternionModelWithMemory:
    """Test Quaternion model integration with VSAMemory."""

    @pytest.fixture
    def model(self):
        """Create a quaternion model."""
        return create_quaternion_model(dim=256)

    @pytest.fixture
    def memory(self, model):
        """Create a VSAMemory with quaternion model."""
        return VSAMemory(model, key=jax.random.PRNGKey(42))

    def test_add_symbol(self, memory):
        """Test adding a symbol to memory."""
        memory.add("apple")

        assert "apple" in memory
        assert isinstance(memory["apple"], QuaternionHypervector)

    def test_add_many_symbols(self, memory):
        """Test adding multiple symbols."""
        memory.add_many(["dog", "cat", "bird"])

        assert "dog" in memory
        assert "cat" in memory
        assert "bird" in memory

    def test_symbol_shape(self, memory):
        """Test that symbols have correct shape."""
        memory.add("test")

        hv = memory["test"]

        assert hv.shape == (256, 4)

    def test_symbols_are_unit_quaternions(self, memory):
        """Test that symbols are unit quaternions."""
        memory.add("test")

        hv = memory["test"]

        assert hv.is_unit()


class TestQuaternionBindUnbindRoundTrip:
    """Test bind/unbind round-trip with quaternion model."""

    @pytest.fixture
    def model(self):
        """Create a quaternion model."""
        return create_quaternion_model(dim=512)

    @pytest.fixture
    def memory(self, model):
        """Create memory with symbols."""
        mem = VSAMemory(model, key=jax.random.PRNGKey(42))
        mem.add_many(["role", "filler", "query"])
        return mem

    def test_right_unbind_round_trip(self, model, memory):
        """Test right-unbind: z = bind(x, y), unbind(z, y) ≈ x."""
        x = memory["role"].vec
        y = memory["filler"].vec

        # Bind
        z = model.opset.bind(x, y)

        # Right-unbind
        recovered_x = model.opset.unbind(z, y)

        # Should recover x
        similarity = quaternion_similarity(x, recovered_x)
        assert similarity > 0.99, f"Right-unbind similarity {similarity} too low"

    def test_left_unbind_round_trip(self, model, memory):
        """Test left-unbind: z = bind(x, y), unbind_left(x, z) ≈ y."""
        x = memory["role"].vec
        y = memory["filler"].vec

        # Bind
        z = model.opset.bind(x, y)

        # Left-unbind
        recovered_y = model.opset.unbind_left(x, z)

        # Should recover y
        similarity = quaternion_similarity(y, recovered_y)
        assert similarity > 0.99, f"Left-unbind similarity {similarity} too low"

    def test_non_commutative_binding(self, model, memory):
        """Test that binding is non-commutative."""
        x = memory["role"].vec
        y = memory["filler"].vec

        xy = model.opset.bind(x, y)
        yx = model.opset.bind(y, x)

        similarity = quaternion_similarity(xy, yx)
        assert similarity < 0.5, f"Binding should be non-commutative, sim={similarity}"

    def test_order_sensitive_role_filler(self, model, memory):
        """Test order-sensitive role/filler binding and recovery."""
        role = memory["role"].vec
        filler = memory["filler"].vec

        # Create role-filler binding
        role_filler = model.opset.bind(role, filler)

        # Right-unbind with filler recovers role
        recovered_role = model.opset.unbind(role_filler, filler)
        role_sim = quaternion_similarity(role, recovered_role)
        assert role_sim > 0.99

        # Left-unbind with role recovers filler
        recovered_filler = model.opset.unbind_left(role, role_filler)
        filler_sim = quaternion_similarity(filler, recovered_filler)
        assert filler_sim > 0.99

        # Cross-recovery should fail (low similarity)
        wrong_unbind = model.opset.unbind(role_filler, role)  # Wrong key
        wrong_sim = quaternion_similarity(filler, wrong_unbind)
        assert wrong_sim < 0.5, "Cross-recovery should have low similarity"


class TestQuaternionBundling:
    """Test bundling with quaternion model."""

    @pytest.fixture
    def model(self):
        """Create a quaternion model."""
        return create_quaternion_model(dim=512)

    @pytest.fixture
    def memory(self, model):
        """Create memory with symbols."""
        mem = VSAMemory(model, key=jax.random.PRNGKey(42))
        mem.add_many(["a", "b", "c", "d", "e"])
        return mem

    def test_bundle_similarity(self, model, memory):
        """Test that bundled vector is similar to constituents."""
        a = memory["a"].vec
        b = memory["b"].vec
        c = memory["c"].vec

        bundled = model.opset.bundle(a, b, c)

        # Should be similar to all inputs
        sim_a = quaternion_similarity(bundled, a)
        sim_b = quaternion_similarity(bundled, b)
        sim_c = quaternion_similarity(bundled, c)

        assert sim_a > 0.3, f"Bundle-a similarity {sim_a} too low"
        assert sim_b > 0.3, f"Bundle-b similarity {sim_b} too low"
        assert sim_c > 0.3, f"Bundle-c similarity {sim_c} too low"

    def test_bundle_decreasing_similarity(self, model, memory):
        """Test that bundling more items decreases individual similarity."""
        items = [memory[k].vec for k in ["a", "b", "c", "d", "e"]]

        # Bundle 2 items
        bundle_2 = model.opset.bundle(items[0], items[1])
        sim_2 = quaternion_similarity(bundle_2, items[0])

        # Bundle 5 items
        bundle_5 = model.opset.bundle(*items)
        sim_5 = quaternion_similarity(bundle_5, items[0])

        # Similarity should decrease with more items
        assert sim_2 > sim_5, "Bundling more items should decrease similarity"

    def test_bundle_is_normalized(self, model, memory):
        """Test that bundled vector is normalized."""
        a = memory["a"].vec
        b = memory["b"].vec

        bundled = model.opset.bundle(a, b)

        norms = jnp.linalg.norm(bundled, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)


class TestQuaternionComplexScenarios:
    """Test complex scenarios with quaternion model."""

    @pytest.fixture
    def model(self):
        """Create a quaternion model."""
        return create_quaternion_model(dim=1024)

    @pytest.fixture
    def memory(self, model):
        """Create memory with symbols."""
        mem = VSAMemory(model, key=jax.random.PRNGKey(42))
        mem.add_many(["subject", "verb", "object", "john", "eats", "apple"])
        return mem

    def test_triple_encoding(self, model, memory):
        """Test encoding and recovery of (subject, verb, object) triple."""
        # Roles
        subject = memory["subject"].vec
        verb = memory["verb"].vec
        obj = memory["object"].vec

        # Fillers
        john = memory["john"].vec
        eats = memory["eats"].vec
        apple = memory["apple"].vec

        # Encode triple: bundle of role-filler bindings
        # "John eats apple"
        subj_bind = model.opset.bind(subject, john)
        verb_bind = model.opset.bind(verb, eats)
        obj_bind = model.opset.bind(obj, apple)

        triple = model.opset.bundle(subj_bind, verb_bind, obj_bind)

        # Query: What is the subject?
        query_subj = model.opset.unbind_left(subject, triple)
        sim_john = quaternion_similarity(query_subj, john)
        sim_eats = quaternion_similarity(query_subj, eats)
        sim_apple = quaternion_similarity(query_subj, apple)

        # John should have highest similarity
        assert sim_john > sim_eats, "Subject query should match john best"
        assert sim_john > sim_apple, "Subject query should match john best"

    def test_nested_structure(self, model, memory):
        """Test nested binding structure."""
        a = memory["subject"].vec
        b = memory["verb"].vec
        c = memory["object"].vec

        # Create nested structure: bind(bind(a, b), c)
        ab = model.opset.bind(a, b)
        abc = model.opset.bind(ab, c)

        # Recover ab by right-unbind with c
        recovered_ab = model.opset.unbind(abc, c)
        sim = quaternion_similarity(ab, recovered_ab)
        assert sim > 0.99, f"Nested unbind similarity {sim} too low"

        # Recover a by right-unbind with b
        recovered_a = model.opset.unbind(recovered_ab, b)
        sim_a = quaternion_similarity(a, recovered_a)
        assert sim_a > 0.98, f"Double nested unbind similarity {sim_a} too low"
