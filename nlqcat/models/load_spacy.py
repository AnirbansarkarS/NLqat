import spacy

def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        raise Exception("Run: python -m spacy download en_core_web_sm")
