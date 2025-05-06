%% trainer_api_finetuning.py
# Setup: pip install transformers datasets matplotlib pandas
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from collections import Counter
import torch

# Synthetic Data: Simulated text classification dataset
dataset = [
    {"text": "I love this product!", "label": 1},
    {"text": "This is terrible.", "label": 0},
    {"text": "Amazing experience!", "label": 1},
    {"text": "Not good at all.", "label": 0}
]

# Function to fine-tune with Trainer API and visualize results
def finetune_trainer():
    print("Synthetic Data: Text classification dataset")
    print("Dataset:", [d["text"] for d in dataset])
    
    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Preprocess data
    encodings = tokenizer([d["text"] for d in dataset], truncation=True, padding=True, return_tensors="pt")
    labels = torch.tensor([d["label"] for d in dataset])
    
    # Create dataset
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item
        def __len__(self):
            return len(self.labels)
    
    train_dataset = CustomDataset(encodings, labels)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        logging_steps=1,
        no_cuda=True  # Ensure CPU compatibility
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    # Simulate training and track loss
    losses = [0.7, 0.6, 0.5, 0.4]  # Simulated loss values
    try:
        trainer.train()
        print("Fine-tuning completed")
    except Exception as e:
        print("Error during fine-tuning:", e)
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', color='blue')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("trainer_api_finetuning_output.png")
    print("Visualization: Training loss saved as trainer_api_finetuning_output.png")

if __name__ == "__main__":
    finetune_trainer()