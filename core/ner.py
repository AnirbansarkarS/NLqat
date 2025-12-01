def extract_entities(doc):
    return [
        {
            "text": ent.text,
            "label": ent.label_
        }
        for ent in doc.ents
    ]
