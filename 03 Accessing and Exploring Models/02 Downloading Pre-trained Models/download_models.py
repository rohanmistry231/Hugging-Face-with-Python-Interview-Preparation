%% download_models.py
# Setup: pip install transformers matplotlib pandas
from transformers import AutoModel, AutoTokenizer, pipeline
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Models to download
models = [
    {"name": "distilbert-base-uncased", "task": "text-classification"},
    {"name": "gpt2", "task": "text-generation"},
    {"name": "google/vit-base-patch16-224", "task": "image-classification"}
]

# Function to download models and visualize results
def download_models():
    print("Synthetic Data: Models to download")
    print("Models:", [m["name"] for m in models])
    
    # Track download success
    success_counts = {"Successful": 0, "Failed": 0}
    
    for model in models:
        model_name, task = model["name"], model["task"]
        try:
            # Download model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            print(f"Downloaded {model_name} successfully")
            
            # Verify with a sample task
            if task == "text-classification":
                pipe = pipeline("text-classification", model=model_name)
                result = pipe("This is a test")
                print(f"Sample result for {model_name}: {result}")
            success_counts["Successful"] += 1
        except Exception as e:
            success_counts["Failed"] += 1
            print(f"Error downloading {model_name}: {e}")
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.bar(success_counts.keys(), success_counts.values(), color=['green', 'red'])
    plt.title("Model Download Success")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.savefig("download_models_output.png")
    print("Visualization: Model download results saved as download_models_output.png")

if __name__ == "__main__":
    download_models()