%% pipeline_introduction.py
# Setup: pip install transformers matplotlib pandas
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Sample texts for pipeline testing
texts = [
    "Hugging Face is amazing!",
    "I dislike bugs in code.",
    "AI is transforming the world."
]

# Function to demonstrate pipeline basics and visualize results
def intro_pipelines():
    print("Synthetic Data: Sample texts")
    print("Texts:", texts)
    
    # Initialize pipeline
    try:
        classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        print("Error initializing pipeline:", e)
        return
    
    # Track pipeline success
    success_counts = {"Successful": 0, "Failed": 0}
    
    # Run pipeline on texts
    results = []
    for text in texts:
        try:
            result = classifier(text)
            results.append(result[0]["label"])
            success_counts["Successful"] += 1
            print(f"Text: {text} -> Label: {result[0]['label']}")
        except Exception as e:
            success_counts["Failed"] += 1
            print(f"Error processing text '{text}': {e}")
    
    # Visualization
    label_counts = Counter(results)
    plt.figure(figsize=(6, 4))
    plt.bar(label_counts.keys(), label_counts.values(), color=['green', 'red'])
    plt.title("Pipeline Text Classification Results")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("pipeline_introduction_output.png")
    print("Visualization: Pipeline results saved as pipeline_introduction_output.png")

if __name__ == "__main__":
    intro_pipelines()