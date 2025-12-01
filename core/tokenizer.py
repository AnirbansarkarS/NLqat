def tokenize_text(doc):
    return [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "is_stop": token.is_stop
        }
        for token in doc
    ]
