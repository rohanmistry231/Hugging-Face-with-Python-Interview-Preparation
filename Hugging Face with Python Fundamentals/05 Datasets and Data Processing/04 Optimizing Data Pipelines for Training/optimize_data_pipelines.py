# %% optimize_data_pipelines.py
# Setup: pip install datasets transformers matplotlib pandas
from datasets import Dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import time

# Synthetic Data: Custom text classification dataset
custom_data = [
    {"text": "I love this app!", "label": 1},
    {"text": "Horrible user experience.", "label": 0},
    {"text": "Really intuitive design!", "label": 1},
    {"text": "Crashes all the time.", "label": 0}
]

# Function to optimize data pipeline and visualize performance
def optimize_pipeline():
    print("Synthetic Data: Custom text classification dataset")
    print("Dataset:", [d["text"] for d in custom_data])
    
    # Create Hugging Face Dataset
    dataset = Dataset.from_pandas(pd.DataFrame(custom_data))
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Optimize pipeline with map function
    start_time = time.time()
    try:
        tokenized_dataset = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=32),
            batched=True
        )
        print("Optimized pipeline completed:", tokenized_dataset)
    except Exception as e:
        print("Error in pipeline:", e)
        return
    
    # Measure performance
    processing_time = time.time() - start_time
    metrics = {"Processing Time (s)": processing_time}
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values(), color='blue')
    plt.title("Data Pipeline Performance")
    plt.ylabel("Time (seconds)")
    plt.savefig("optimize_data_pipelines_output.png")
    print("Visualization: Pipeline performance saved as optimize_data_pipelines_output.png")

if __name__ == "__main__":
    optimize_pipeline()