%% evaluate_save_models.py
# Setup: pip install transformers scikit-learn matplotlib pandas
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Simulated test dataset
test_dataset = [
    {"text": "Really great product!", "label": 1},
    {"text": "Awful experience.", "label": 0},
    {"text": "Fantastic service!", "label": 1},
    {"text": "Very disappointing.", "label": 0}
]

# Function to evaluate and save model, and visualize results
def evaluate_save_model():
    print("Synthetic Data: Test dataset")
    print("Dataset:", [d["text"] for d in test_dataset])
    
    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Evaluate model
    predictions = []
    true_labels = [d["label"] for d in test_dataset]
    for item in test_dataset:
        try:
            inputs = tokenizer(item["text"], truncation=True, padding=True, return_tensors="pt")
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)
            print(f"Text: {item['text']} -> Predicted: {pred}")
        except Exception as e:
            print(f"Error evaluating {item['text']}: {e}")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    metrics = {"Accuracy": accuracy, "F1 Score": f1}
    
    # Simulate saving model
    try:
        model.save_pretrained("./fine_tuned_model")
        tokenizer.save_pretrained("./fine_tuned_model")
        print("Model and tokenizer saved to ./fine_tuned_model")
    except Exception as e:
        print("Error saving model:", e)
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green'])
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.savefig("evaluate_save_models_output.png")
    print("Visualization: Evaluation metrics saved as evaluate_save_models_output.png")

if __name__ == "__main__":
    evaluate_save_model()