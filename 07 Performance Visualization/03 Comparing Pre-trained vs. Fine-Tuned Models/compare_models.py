%% compare_models.py
# Setup: pip install transformers scikit-learn matplotlib pandas
from transformers import pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

# Synthetic Data: Mock test dataset
test_dataset = [
    {"text": "Great product!", "label": 1},
    {"text": "Awful experience.", "label": 0},
    {"text": "Fantastic service!", "label": 1},
    {"text": "Very disappointing.", "label": 0}
]

# Function to compare pre-trained vs. fine-tuned models and visualize results
def compare_models():
    print("Synthetic Data: Test dataset")
    print("Dataset:", [d["text"] for d in test_dataset])
    
    # Initialize models
    pretrained = pipeline("text-classification", model="distilbert-base-uncased")
    finetuned = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Measure performance
    true_labels = [d["label"] for d in test_dataset]
    pretrained_preds = []
    finetuned_preds = []
    pretrained_latencies = []
    finetuned_latencies = []
    
    for item in test_dataset:
        try:
            # Pre-trained model
            start_time = time.time()
            result = pretrained(item["text"])
            pretrained_preds.append(1 if result[0]["label"] == "POSITIVE" else 0)
            pretrained_latencies.append(time.time() - start_time)
            
            # Fine-tuned model
            start_time = time.time()
            result = finetuned(item["text"])
            finetuned_preds.append(1 if result[0]["label"] == "POSITIVE" else 0)
            finetuned_latencies.append(time.time() - start_time)
            
            print(f"Text: {item['text']} -> Pre-trained: {pretrained_preds[-1]}, Fine-tuned: {finetuned_preds[-1]}")
        except Exception as e:
            print(f"Error processing {item['text']}: {e}")
    
    # Calculate accuracies
    pretrained_accuracy = accuracy_score(true_labels, pretrained_preds)
    finetuned_accuracy = accuracy_score(true_labels, finetuned_preds)
    
    # Visualization
    metrics = {
        "Pre-trained Accuracy": pretrained_accuracy,
        "Fine-tuned Accuracy": finetuned_accuracy,
        "Pre-trained Avg Latency (s)": sum(pretrained_latencies) / len(pretrained_latencies),
        "Fine-tuned Avg Latency (s)": sum(finetuned_latencies) / len(finetuned_latencies)
    }
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red', 'purple'])
    plt.title("Pre-trained vs. Fine-tuned Model Performance")
    plt.xticks(rotation=45)
    plt.savefig("compare_models_output.png")
    print("Visualization: Model comparison saved as compare_models_output.png")

if __name__ == "__main__":
    compare_models()