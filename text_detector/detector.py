from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

def detect_ai_text(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    # probs[0][1] → AI-generated, probs[0][0] → Human
    return {
        "human_prob": float(probs[0][0]),
        "ai_prob": float(probs[0][1])
    }

if __name__ == "__main__":
    sample = "This is a test sentence written by an AI."
    result = detect_ai_text(sample)
    print("Result:", result)

