%% text_pipelines.py
# Setup: pip install transformers matplotlib pandas
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data
texts = [
    {"task": "text-classification", "input": "I love coding with Python!"},
    {"task": "ner", "input": "Elon Musk founded SpaceX in California."},
    {"task": "question-answering", "input": {"question": "Who founded SpaceX?", "context": "Elon Musk founded SpaceX in 2002."}}
]

# Function to demonstrate text pipelines and visualize results
def text_pipelines():
    print("Synthetic Data: Text inputs")
    print("Inputs:", [t["input"] for t in texts])
    
    # Initialize pipelines
    pipelines = {
        "text-classification": pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english"),
        "ner": pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english"),
        "question-answering": pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    }
    
    # Track task success
    success_counts = {"Successful": 0, "Failed": 0}
    
    # Run pipelines
    task_results = []
    for item in texts:
        task, input_data = item["task"], item["input"]
        try:
            if task == "question-answering":
                result = pipelines[task](question=input_data["question"], context=input_data["context"])
                task_results.append("Answer")
                print(f"QA: {input_data['question']} -> {result['answer']}")
            elif task == "ner":
                result = pipelines[task](input_data)
                entities = [r["entity"] for r in result]
                task_results.append("Entity")
                print(f"NER: {input_data} -> {entities}")
            else:
                result = pipelines[task](input_data)
                task_results.append(result[0]["label"])
                print(f"Classification: {input_data} -> {result[0]['label']}")
            success_counts["Successful"] += 1
        except Exception as e:
            success_counts["Failed"] += 1
            print(f"Error in {task}: {e}")
    
    # Visualization
    result_counts = Counter(task_results)
    plt.figure(figsize=(6, 4))
    plt.bar(result_counts.keys(), result_counts.values(), color='blue')
    plt.title("Text Pipeline Results")
    plt.xlabel("Result Type")
    plt.ylabel("Count")
    plt.savefig("text_pipelines_output.png")
    print("Visualization: Text pipeline results saved as text_pipelines_output.png")

if __name__ == "__main__":
    text_pipelines()