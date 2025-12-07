from nlqat.core.base_pipeline import NLP

def test_phase1():
    print("Initializing NLP...")
    nlp = NLP()
    text = "Apple is investing $5 billion in AI."
    print(f"Analyzing text: '{text}'")
    doc = nlp.analyze(text)

    print("\n--- Tokens ---")
    print(doc.tokens)

    print("\n--- POS Tags ---")
    print(doc.pos_tags)

    print("\n--- Entities ---")
    print(doc.entities)

    print("\n--- Dependencies ---")
    for dep in doc.dependencies:
        print(f"{dep['text']} --[{dep['dep']}]--> {dep['head']}")

    expected_entities = ["Apple", "$5 billion", "AI"]
    detected_entities = [e['text'] for e in doc.entities]
    
    if set(expected_entities).issubset(set(detected_entities)):
        print("\nSUCCESS: All expected entities detected.")
    else:
        print(f"\nFAILURE: Expected {expected_entities}, got {detected_entities}")

if __name__ == "__main__":
    test_phase1()
