"""Tests for semantic role operators."""

import jax.numpy as jnp

from vsax import VSAMemory, create_fhrr_model
from vsax.operators import (
    OperatorKind,
    create_agent,
    create_experiencer,
    create_goal,
    create_instrument,
    create_location,
    create_patient,
    create_semantic_operators,
    create_source,
    create_theme,
)
from vsax.similarity import cosine_similarity


def test_create_agent() -> None:
    """Test creating AGENT operator."""
    op = create_agent(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "AGENT"
    assert op.metadata.kind == OperatorKind.SEMANTIC


def test_create_patient() -> None:
    """Test creating PATIENT operator."""
    op = create_patient(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "PATIENT"
    assert op.metadata.kind == OperatorKind.SEMANTIC


def test_create_theme() -> None:
    """Test creating THEME operator."""
    op = create_theme(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "THEME"
    assert op.metadata.kind == OperatorKind.SEMANTIC


def test_create_experiencer() -> None:
    """Test creating EXPERIENCER operator."""
    op = create_experiencer(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "EXPERIENCER"
    assert op.metadata.kind == OperatorKind.SEMANTIC


def test_create_instrument() -> None:
    """Test creating INSTRUMENT operator."""
    op = create_instrument(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "INSTRUMENT"
    assert op.metadata.kind == OperatorKind.SEMANTIC


def test_create_location() -> None:
    """Test creating LOCATION operator."""
    op = create_location(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "LOCATION"
    assert op.metadata.kind == OperatorKind.SEMANTIC


def test_create_goal() -> None:
    """Test creating GOAL operator."""
    op = create_goal(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "GOAL"
    assert op.metadata.kind == OperatorKind.SEMANTIC


def test_create_source() -> None:
    """Test creating SOURCE operator."""
    op = create_source(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "SOURCE"
    assert op.metadata.kind == OperatorKind.SEMANTIC


def test_semantic_operators_reproducible() -> None:
    """Test that semantic operators are reproducible."""
    # Create operators twice
    AGENT_1 = create_agent(512)
    AGENT_2 = create_agent(512)

    PATIENT_1 = create_patient(512)
    PATIENT_2 = create_patient(512)

    # Should have identical parameters
    assert jnp.allclose(AGENT_1.params, AGENT_2.params)
    assert jnp.allclose(PATIENT_1.params, PATIENT_2.params)


def test_semantic_operators_different() -> None:
    """Test that different semantic operators have different parameters."""
    AGENT = create_agent(512)
    PATIENT = create_patient(512)
    THEME = create_theme(512)

    # Different operators should have different parameters
    assert not jnp.allclose(AGENT.params, PATIENT.params)
    assert not jnp.allclose(AGENT.params, THEME.params)
    assert not jnp.allclose(PATIENT.params, THEME.params)


def test_create_semantic_operators() -> None:
    """Test creating all semantic operators at once."""
    operators = create_semantic_operators(512)

    # Should have all 8 operators
    assert len(operators) == 8
    assert "AGENT" in operators
    assert "PATIENT" in operators
    assert "THEME" in operators
    assert "EXPERIENCER" in operators
    assert "INSTRUMENT" in operators
    assert "LOCATION" in operators
    assert "GOAL" in operators
    assert "SOURCE" in operators

    # All should have correct dimension
    for op in operators.values():
        assert op.dim == 512
        assert op.metadata.kind == OperatorKind.SEMANTIC


def test_semantic_operators_basic_sentence() -> None:
    """Test semantic operators with basic sentence encoding."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["dog", "cat", "chase"])

    AGENT = create_agent(512)
    PATIENT = create_patient(512)

    # Encode: "dog chases cat"
    sentence = model.opset.bundle(
        AGENT.apply(memory["dog"]).vec, memory["chase"].vec, PATIENT.apply(memory["cat"]).vec
    )

    # Query: Who is the AGENT?
    who_agent = AGENT.inverse().apply(model.rep_cls(sentence))
    sim_dog = cosine_similarity(who_agent.vec, memory["dog"].vec)
    sim_cat = cosine_similarity(who_agent.vec, memory["cat"].vec)

    # Query: Who is the PATIENT?
    who_patient = PATIENT.inverse().apply(model.rep_cls(sentence))
    sim_cat_patient = cosine_similarity(who_patient.vec, memory["cat"].vec)
    sim_dog_patient = cosine_similarity(who_patient.vec, memory["dog"].vec)

    # AGENT query should retrieve dog
    assert sim_dog > sim_cat
    assert sim_dog > 0.4

    # PATIENT query should retrieve cat
    assert sim_cat_patient > sim_dog_patient
    assert sim_cat_patient > 0.4


def test_semantic_operators_complex_event() -> None:
    """Test semantic operators with complex event encoding."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["John", "bread", "knife", "cut", "kitchen"])

    AGENT = create_agent(512)
    PATIENT = create_patient(512)
    INSTRUMENT = create_instrument(512)
    LOCATION = create_location(512)

    # Encode: "John cut the bread with a knife in the kitchen"
    event = model.opset.bundle(
        AGENT.apply(memory["John"]).vec,
        memory["cut"].vec,
        PATIENT.apply(memory["bread"]).vec,
        INSTRUMENT.apply(memory["knife"]).vec,
        LOCATION.apply(memory["kitchen"]).vec,
    )

    # Query 1: Who is the AGENT?
    who = AGENT.inverse().apply(model.rep_cls(event))
    sim_john = cosine_similarity(who.vec, memory["John"].vec)

    # Query 2: What is the PATIENT?
    what = PATIENT.inverse().apply(model.rep_cls(event))
    sim_bread = cosine_similarity(what.vec, memory["bread"].vec)

    # Query 3: What is the INSTRUMENT?
    with_what = INSTRUMENT.inverse().apply(model.rep_cls(event))
    sim_knife = cosine_similarity(with_what.vec, memory["knife"].vec)

    # Query 4: Where is the LOCATION?
    where = LOCATION.inverse().apply(model.rep_cls(event))
    sim_kitchen = cosine_similarity(where.vec, memory["kitchen"].vec)

    # All queries should work
    assert sim_john > 0.3
    assert sim_bread > 0.3
    assert sim_knife > 0.3
    assert sim_kitchen > 0.3


def test_semantic_operators_transfer_event() -> None:
    """Test semantic operators with transfer/motion event."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["John", "book", "Mary", "give"])

    AGENT = create_agent(512)
    THEME = create_theme(512)
    GOAL = create_goal(512)

    # Encode: "John gave the book to Mary"
    event = model.opset.bundle(
        AGENT.apply(memory["John"]).vec,
        memory["give"].vec,
        THEME.apply(memory["book"]).vec,
        GOAL.apply(memory["Mary"]).vec,
    )

    # Query: What was given? (THEME)
    what = THEME.inverse().apply(model.rep_cls(event))
    sim_book = cosine_similarity(what.vec, memory["book"].vec)

    # Query: To whom? (GOAL)
    to_whom = GOAL.inverse().apply(model.rep_cls(event))
    sim_mary = cosine_similarity(to_whom.vec, memory["Mary"].vec)

    assert sim_book > 0.35
    assert sim_mary > 0.35


def test_semantic_operators_motion_event() -> None:
    """Test semantic operators with motion event (SOURCE and GOAL)."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["John", "Boston", "Paris", "travel"])

    AGENT = create_agent(512)
    SOURCE = create_source(512)
    GOAL = create_goal(512)

    # Encode: "John traveled from Boston to Paris"
    event = model.opset.bundle(
        AGENT.apply(memory["John"]).vec,
        memory["travel"].vec,
        SOURCE.apply(memory["Boston"]).vec,
        GOAL.apply(memory["Paris"]).vec,
    )

    # Query: From where? (SOURCE)
    from_where = SOURCE.inverse().apply(model.rep_cls(event))
    sim_boston = cosine_similarity(from_where.vec, memory["Boston"].vec)

    # Query: To where? (GOAL)
    to_where = GOAL.inverse().apply(model.rep_cls(event))
    sim_paris = cosine_similarity(to_where.vec, memory["Paris"].vec)

    assert sim_boston > 0.4
    assert sim_paris > 0.4


def test_semantic_operators_experiencer_event() -> None:
    """Test semantic operators with mental/emotional state."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["Mary", "music", "love"])

    EXPERIENCER = create_experiencer(512)
    THEME = create_theme(512)

    # Encode: "Mary loves music"
    event = model.opset.bundle(
        EXPERIENCER.apply(memory["Mary"]).vec, memory["love"].vec, THEME.apply(memory["music"]).vec
    )

    # Query: Who experiences the feeling?
    who = EXPERIENCER.inverse().apply(model.rep_cls(event))
    sim_mary = cosine_similarity(who.vec, memory["Mary"].vec)

    # Query: What is experienced?
    what = THEME.inverse().apply(model.rep_cls(event))
    sim_music = cosine_similarity(what.vec, memory["music"].vec)

    assert sim_mary > 0.4
    assert sim_music > 0.4


def test_semantic_operators_exact_inversion() -> None:
    """Test exact inversion with semantic operators."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add("test")

    AGENT = create_agent(512)

    hv = memory["test"]
    transformed = AGENT.apply(hv)
    recovered = AGENT.inverse().apply(transformed)

    # Should recover original with high similarity
    similarity = cosine_similarity(recovered.vec, hv.vec)
    assert similarity > 0.999


def test_semantic_operators_composition() -> None:
    """Test composing semantic operators."""
    AGENT = create_agent(512)
    PATIENT = create_patient(512)

    # Compose operators
    composed = AGENT.compose(PATIENT)

    assert composed.dim == 512
    # Composed params should be sum of individual params
    assert jnp.allclose(composed.params, AGENT.params + PATIENT.params)


def test_semantic_operators_multiple_sentences() -> None:
    """Test distinguishing multiple sentences with different roles."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["dog", "cat", "chase", "fish", "eat"])

    AGENT = create_agent(512)
    PATIENT = create_patient(512)

    # Sentence 1: "dog chases cat"
    s1 = model.opset.bundle(
        AGENT.apply(memory["dog"]).vec, memory["chase"].vec, PATIENT.apply(memory["cat"]).vec
    )

    # Sentence 2: "cat eats fish"
    s2 = model.opset.bundle(
        AGENT.apply(memory["cat"]).vec, memory["eat"].vec, PATIENT.apply(memory["fish"]).vec
    )

    # Query S1 PATIENT
    patient_s1 = PATIENT.inverse().apply(model.rep_cls(s1))
    sim_cat_s1 = cosine_similarity(patient_s1.vec, memory["cat"].vec)
    sim_fish_s1 = cosine_similarity(patient_s1.vec, memory["fish"].vec)

    # Query S2 PATIENT
    patient_s2 = PATIENT.inverse().apply(model.rep_cls(s2))
    sim_cat_s2 = cosine_similarity(patient_s2.vec, memory["cat"].vec)
    sim_fish_s2 = cosine_similarity(patient_s2.vec, memory["fish"].vec)

    # S1 should retrieve cat, S2 should retrieve fish
    assert sim_cat_s1 > sim_fish_s1
    assert sim_fish_s2 > sim_cat_s2


def test_semantic_operators_different_dimensions() -> None:
    """Test creating semantic operators with different dimensions."""
    op_512 = create_agent(512)
    op_1024 = create_agent(1024)

    assert op_512.dim == 512
    assert op_1024.dim == 1024

    # Different dimensions should have different parameter vectors
    assert op_512.params.shape[0] == 512
    assert op_1024.params.shape[0] == 1024
