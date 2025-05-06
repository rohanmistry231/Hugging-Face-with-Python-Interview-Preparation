# %% visualize_metrics.py
# Setup: pip install transformers scikit-learn matplotlib seaborn pandas
from transformers import pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Synthetic Data: Mock predictions
test_dataset = [
    {"text": "Great product!", "label": 1},
    {"text": "Awful experience.", "label": 0},
    {"text": "Fantastic service!", "label": 1},
    {"text": "Very disappointing.", "label": 0}
]

# Function to visualize performance metrics
def visualize_metrics():
    print("Synthetic Data: Test dataset")
    print("Dataset:", [d["text"] for d in test_dataset])
    
    # Initialize classifier
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Generate predictions
    true_labels = [d["label"] for d in test_dataset]
    pred_labels = []
    for item in test_dataset:
        try:
            result = classifier(item["text"])
            pred_labels.append(1 if result[0]["label"] == "POSITIVE" else 0)
            print(f"Text: {item['text']} -> Predicted: {pred_labels[-1]}")
        except Exception as e:
            print(f"Error processing {item['text']}: {e}")
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Visualization
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    plt.subplot(1, 2, 2)
    latencies = [0.1, 0.12, 0.09, 0.11]  # Simulated latencies
    plt.plot(range(1, len(latencies) + 1), latencies, marker='o', color='blue')
    plt.title("Inference Latency")
    plt.xlabel("Request Number")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("visualize_metrics_output.png")
    print("Visualization: Performance metrics saved as visualize_metrics_output.png")

if __name__ == "__main__":
    visualize_metrics()