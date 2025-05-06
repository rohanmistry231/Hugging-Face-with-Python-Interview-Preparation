# %% ethical_ai.py
# Setup: pip install transformers evaluate matplotlib pandas
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Mock text inputs for bias analysis
texts = [
    {"text": "He is a great engineer.", "group": "male"},
    {"text": "She is a great engineer.", "group": "female"},
    {"text": "They are a great engineer.", "group": "neutral"}
]

# Function to analyze bias and visualize results
def ethical_ai():
    print("Synthetic Data: Mock text inputs")
    print("Texts:", [t["text"] for t in texts])
    
    # Initialize classifier
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Analyze predictions for bias
    predictions = []
    success_counts = {"Successful": 0, "Failed": 0}
    
    for item in texts:
        try:
            result = classifier(item["text"])
            predictions.append({"group": item["group"], "label": result[0]["label"], "score": result[0]["score"]})
            success_counts["Successful"] += 1
            print(f"Text: {item['text']} -> Label: {result[0]['label']}, Score: {result[0]['score']:.4f}")
        except Exception as e:
            success_counts["Failed"] += 1
            print(f"Error processing {item['text']}: {e}")
    
    # Visualize bias metrics
    group_scores = Counter(p["group"] for p in predictions if p["label"] == "POSITIVE")
    plt.figure(figsize=(6, 4))
    plt.bar(group_scores.keys(), group_scores.values(), color='blue')
    plt.title("Positive Predictions by Group")
    plt.xlabel("Group")
    plt.ylabel("Count")
    plt.savefig("ethical_ai_output.png")
    print("Visualization: Bias analysis saved as ethical_ai_output.png")

if __name__ == "__main__":
    ethical_ai()