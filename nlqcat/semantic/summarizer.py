from transformers import pipeline

class Summarizer:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """
        Initializes the summarization pipeline.
        Uses distilbart-cnn-12-6 by default for a good speed/quality trade-off.
        """
        print(f"Loading summarization model: {model_name}...")
        self.pipeline = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Summarizes the input text.
        """
        # handling short text gracefully
        if len(text.split()) < min_length:
            return text
            
        summary = self.pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
