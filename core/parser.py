class Parser:
    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def parse(self, text: str) -> list:
        doc = self.nlp(text)
        parsed_data = []
        for token in doc:
            parsed_data.append({
                "text": token.text,
                "dep": token.dep_,
                "head": token.head.text,
                "head_pos": token.head.pos_,
                "children": [child.text for child in token.children]
            })
        return parsed_data
