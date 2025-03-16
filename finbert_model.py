from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd

class FINBERTSentimentAnalyzer:
    def __init__(self):        
        print("Initialize the FINBERT model and tokenizer.")

        print("Initializing FINBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def analyze_sentiment(self, text):
        print("Analyze the sentiment of the given text using FINBERT.")
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
        sentiment_scores = {
            "positive": probs[0][0].item(),
            "neutral": probs[0][1].item(),
            "negative": probs[0][2].item()
        }
        return sentiment_scores

    def batch_analyze_sentiment(self, texts):
        print("Analyze sentiment for a batch of texts.")

        results = []
        for text in texts:
            scores = self.analyze_sentiment(text)
            results.append(scores)
        return pd.DataFrame(results)