"""Example: Semantic Role Labeling with Clifford Operators.

Demonstrates how to use semantic operators to encode and query "who did what to whom".

This example shows:
- Creating semantic role operators (AGENT, PATIENT, etc.)
- Encoding sentences and events
- Querying semantic roles
- Handling complex multi-role events
"""

from vsax import VSAMemory, create_fhrr_model
from vsax.operators import (
    create_agent,
    create_experiencer,
    create_goal,
    create_instrument,
    create_location,
    create_patient,
    create_source,
    create_theme,
)
from vsax.similarity import cosine_similarity


def main() -> None:
    """Run semantic role labeling example."""
    print("=" * 80)
    print("Semantic Role Labeling with Clifford Operators")
    print("=" * 80)
    print()

    # Setup
    print("Setting up FHRR model with dimension 1024...")
    model = create_fhrr_model(dim=1024)
    memory = VSAMemory(model)

    # Add vocabulary
    vocab = [
        "dog",
        "cat",
        "mouse",
        "chase",
        "eat",
        "John",
        "Mary",
        "bread",
        "knife",
        "cut",
        "kitchen",
        "give",
        "book",
        "Paris",
        "Boston",
        "travel",
        "music",
        "love",
    ]
    memory.add_many(vocab)
    print(f"Added {len(vocab)} words to vocabulary")
    print()

    # Create semantic operators
    print("Creating semantic role operators...")
    AGENT = create_agent(1024)
    PATIENT = create_patient(1024)
    THEME = create_theme(1024)
    EXPERIENCER = create_experiencer(1024)
    INSTRUMENT = create_instrument(1024)
    LOCATION = create_location(1024)
    GOAL = create_goal(1024)
    SOURCE = create_source(1024)

    print(f"  {AGENT}")
    print(f"  {PATIENT}")
    print(f"  {THEME}")
    print(f"  {EXPERIENCER}")
    print(f"  {INSTRUMENT}")
    print(f"  {LOCATION}")
    print(f"  {GOAL}")
    print(f"  {SOURCE}")
    print()

    # Example 1: Simple transitive action
    print("-" * 80)
    print("Example 1: Simple Transitive Action")
    print("-" * 80)
    print()

    print("Encoding: 'dog chases cat'")
    print("  AGENT: dog")
    print("  ACTION: chase")
    print("  PATIENT: cat")

    sentence1 = model.opset.bundle(
        AGENT.apply(memory["dog"]).vec,
        memory["chase"].vec,
        PATIENT.apply(memory["cat"]).vec,
    )

    print("\nQuery: Who is the AGENT?")
    who_agent = AGENT.inverse().apply(model.rep_cls(sentence1))
    for word in ["dog", "cat", "chase"]:
        sim = cosine_similarity(who_agent.vec, memory[word].vec)
        print(f"  Similarity to '{word}': {sim:.3f}")
    print("  -> 'dog' has highest similarity")

    print("\nQuery: Who is the PATIENT?")
    who_patient = PATIENT.inverse().apply(model.rep_cls(sentence1))
    for word in ["dog", "cat", "chase"]:
        sim = cosine_similarity(who_patient.vec, memory[word].vec)
        print(f"  Similarity to '{word}': {sim:.3f}")
    print("  -> 'cat' has highest similarity")
    print()

    # Example 2: Complex event with multiple roles
    print("-" * 80)
    print("Example 2: Complex Event with Multiple Roles")
    print("-" * 80)
    print()

    print("Encoding: 'John cut the bread with a knife in the kitchen'")
    print("  AGENT: John")
    print("  ACTION: cut")
    print("  PATIENT: bread")
    print("  INSTRUMENT: knife")
    print("  LOCATION: kitchen")

    event1 = model.opset.bundle(
        AGENT.apply(memory["John"]).vec,
        memory["cut"].vec,
        PATIENT.apply(memory["bread"]).vec,
        INSTRUMENT.apply(memory["knife"]).vec,
        LOCATION.apply(memory["kitchen"]).vec,
    )

    print("\nQuerying all roles:")

    print("\n  Who is the AGENT?")
    who = AGENT.inverse().apply(model.rep_cls(event1))
    sim_john = cosine_similarity(who.vec, memory["John"].vec)
    print(f"    'John': {sim_john:.3f} *")

    print("\n  What is the PATIENT?")
    what = PATIENT.inverse().apply(model.rep_cls(event1))
    sim_bread = cosine_similarity(what.vec, memory["bread"].vec)
    print(f"    'bread': {sim_bread:.3f} *")

    print("\n  What is the INSTRUMENT?")
    with_what = INSTRUMENT.inverse().apply(model.rep_cls(event1))
    sim_knife = cosine_similarity(with_what.vec, memory["knife"].vec)
    print(f"    'knife': {sim_knife:.3f} *")

    print("\n  Where is the LOCATION?")
    where = LOCATION.inverse().apply(model.rep_cls(event1))
    sim_kitchen = cosine_similarity(where.vec, memory["kitchen"].vec)
    print(f"    'kitchen': {sim_kitchen:.3f} *")

    print("\n-> All roles successfully retrieved!")
    print()

    # Example 3: Transfer event (THEME, GOAL)
    print("-" * 80)
    print("Example 3: Transfer Event (THEME, GOAL)")
    print("-" * 80)
    print()

    print("Encoding: 'John gave the book to Mary'")
    print("  AGENT: John")
    print("  ACTION: give")
    print("  THEME: book (what was given)")
    print("  GOAL: Mary (recipient)")

    event2 = model.opset.bundle(
        AGENT.apply(memory["John"]).vec,
        memory["give"].vec,
        THEME.apply(memory["book"]).vec,
        GOAL.apply(memory["Mary"]).vec,
    )

    print("\nQuery: What was given? (THEME)")
    what_theme = THEME.inverse().apply(model.rep_cls(event2))
    sim_book = cosine_similarity(what_theme.vec, memory["book"].vec)
    print(f"  'book': {sim_book:.3f} *")

    print("\nQuery: To whom? (GOAL)")
    to_whom = GOAL.inverse().apply(model.rep_cls(event2))
    sim_mary = cosine_similarity(to_whom.vec, memory["Mary"].vec)
    print(f"  'Mary': {sim_mary:.3f} *")
    print()

    # Example 4: Motion event (SOURCE, GOAL)
    print("-" * 80)
    print("Example 4: Motion Event (SOURCE, GOAL)")
    print("-" * 80)
    print()

    print("Encoding: 'John traveled from Boston to Paris'")
    print("  AGENT: John")
    print("  ACTION: travel")
    print("  SOURCE: Boston (starting point)")
    print("  GOAL: Paris (destination)")

    event3 = model.opset.bundle(
        AGENT.apply(memory["John"]).vec,
        memory["travel"].vec,
        SOURCE.apply(memory["Boston"]).vec,
        GOAL.apply(memory["Paris"]).vec,
    )

    print("\nQuery: From where? (SOURCE)")
    from_where = SOURCE.inverse().apply(model.rep_cls(event3))
    sim_boston = cosine_similarity(from_where.vec, memory["Boston"].vec)
    print(f"  'Boston': {sim_boston:.3f} *")

    print("\nQuery: To where? (GOAL)")
    to_where = GOAL.inverse().apply(model.rep_cls(event3))
    sim_paris = cosine_similarity(to_where.vec, memory["Paris"].vec)
    print(f"  'Paris': {sim_paris:.3f} *")
    print()

    # Example 5: Experiencer event
    print("-" * 80)
    print("Example 5: Mental/Emotional State (EXPERIENCER)")
    print("-" * 80)
    print()

    print("Encoding: 'Mary loves music'")
    print("  EXPERIENCER: Mary (who experiences the feeling)")
    print("  ACTION: love")
    print("  THEME: music (what is loved)")

    event4 = model.opset.bundle(
        EXPERIENCER.apply(memory["Mary"]).vec,
        memory["love"].vec,
        THEME.apply(memory["music"]).vec,
    )

    print("\nQuery: Who experiences the feeling? (EXPERIENCER)")
    who_exp = EXPERIENCER.inverse().apply(model.rep_cls(event4))
    sim_mary_exp = cosine_similarity(who_exp.vec, memory["Mary"].vec)
    print(f"  'Mary': {sim_mary_exp:.3f} *")

    print("\nQuery: What is experienced? (THEME)")
    what_exp = THEME.inverse().apply(model.rep_cls(event4))
    sim_music = cosine_similarity(what_exp.vec, memory["music"].vec)
    print(f"  'music': {sim_music:.3f} *")
    print()

    # Example 6: Multiple sentences disambiguation
    print("-" * 80)
    print("Example 6: Disambiguating Multiple Sentences")
    print("-" * 80)
    print()

    print("Sentence 1: 'dog chases cat'")
    s1 = model.opset.bundle(
        AGENT.apply(memory["dog"]).vec,
        memory["chase"].vec,
        PATIENT.apply(memory["cat"]).vec,
    )

    print("Sentence 2: 'cat eats mouse'")
    s2 = model.opset.bundle(
        AGENT.apply(memory["cat"]).vec,
        memory["eat"].vec,
        PATIENT.apply(memory["mouse"]).vec,
    )

    print("\nQuery S1 PATIENT:")
    patient_s1 = PATIENT.inverse().apply(model.rep_cls(s1))
    sim_cat_s1 = cosine_similarity(patient_s1.vec, memory["cat"].vec)
    sim_mouse_s1 = cosine_similarity(patient_s1.vec, memory["mouse"].vec)
    print(f"  'cat': {sim_cat_s1:.3f}")
    print(f"  'mouse': {sim_mouse_s1:.3f}")
    print("  -> S1 PATIENT is 'cat' *")

    print("\nQuery S2 PATIENT:")
    patient_s2 = PATIENT.inverse().apply(model.rep_cls(s2))
    sim_cat_s2 = cosine_similarity(patient_s2.vec, memory["cat"].vec)
    sim_mouse_s2 = cosine_similarity(patient_s2.vec, memory["mouse"].vec)
    print(f"  'cat': {sim_cat_s2:.3f}")
    print(f"  'mouse': {sim_mouse_s2:.3f}")
    print("  -> S2 PATIENT is 'mouse' *")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("* Semantic operators encode 'who did what to whom'")
    print("* Multiple roles can be encoded in single event")
    print("* Each role can be queried independently")
    print("* Exact role extraction with high similarity")
    print("* Sentences can be disambiguated")
    print()
    print("Semantic roles covered:")
    print("  AGENT       - doer of action")
    print("  PATIENT     - undergoes action")
    print("  THEME       - thing moved/experienced")
    print("  EXPERIENCER - who has mental state")
    print("  INSTRUMENT  - tool used")
    print("  LOCATION    - where action occurs")
    print("  GOAL        - destination/recipient")
    print("  SOURCE      - starting point")
    print()
    print("Use cases:")
    print("  - Natural language understanding")
    print("  - Question answering systems")
    print("  - Event extraction from text")
    print("  - Semantic parsing")
    print()


if __name__ == "__main__":
    main()
