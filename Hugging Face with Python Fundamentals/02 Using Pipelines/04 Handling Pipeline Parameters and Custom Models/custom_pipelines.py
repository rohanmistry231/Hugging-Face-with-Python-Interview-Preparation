# %% custom_pipelines.py
# Setup: pip install transformers matplotlib pandas
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Sample texts for custom pipeline
texts = [
    "This is a great product!",
    "Not happy with the service.",
    "AI is revolutionary."
]

# Function to demonstrate custom pipelines and visualize results
def custom_pipelines():
    print("Synthetic Data: Sample texts")
    print("Texts:", texts)
    
    # Initialize custom pipeline with specific model and parameters
    try:
        classifier = pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            top_k=None  # Return all scores
        )
    except Exception as e:
        print("Error initializing custom pipeline:", e)
        return
    
    # Track pipeline success
    success_counts = {"Successful": 0, "Failed": 0}
    
    # Run custom pipeline
    label_counts = Counter()
    for text in texts:
        try:
            result = classifier(text)
            top_label = result[0]["label"]  # Get highest-scoring label
            label_counts[top_label] += 1
            success_counts["Successful"] += 1
            print(f"Text: {text} -> Label: {top_label}")
        except Exception as e:
            success_counts["Failed"] += 1
            print(f"Error processing text '{text}': {e}")
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.bar(label_counts.keys(), label_counts.values(), color='orange')
    plt.title("Custom Pipeline Classification Results")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.savefig("custom_pipelines_output.png")
    print("Visualization: Custom pipeline results saved as custom_pipelines_output.png")

if __name__ == "__main__":
    custom_pipelines()