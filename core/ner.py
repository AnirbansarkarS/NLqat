class NER:
    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def extract_entities(self, text: str) -> list:
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities
