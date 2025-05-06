# %% vision_audio_pipelines.py
# Setup: pip install transformers matplotlib pandas
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Simulating vision and audio inputs
tasks = [
    {"task": "image-classification", "input": "Simulated image of a cat"},
    {"task": "automatic-speech-recognition", "input": "Simulated audio: Hello world"}
]

# Function to demonstrate vision/audio pipelines and visualize results
def vision_audio_pipelines():
    print("Synthetic Data: Vision and audio inputs")
    print("Inputs:", [t["input"] for t in tasks])
    
    # Initialize pipelines (using lightweight models)
    try:
        vision_pipeline = pipeline("image-classification", model="google/vit-base-patch16-224")
        audio_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    except Exception as e:
        print("Error initializing pipelines:", e)
        return
    
    # Track task success
    success_counts = {"Successful": 0, "Failed": 0}
    
    # Simulate pipeline execution (actual image/audio processing replaced with synthetic results)
    task_results = []
    for item in tasks:
        task, input_data = item["task"], item["input"]
        try:
            if task == "image-classification":
                # Simulate image classification result
                result = [{"label": "cat", "score": 0.95}]  # Mock result
                task_results.append(result[0]["label"])
                print(f"Image Classification: {input_data} -> {result[0]['label']}")
            else:
                # Simulate speech recognition result
                result = {"text": "Hello world"}  # Mock result
                task_results.append("Speech")
                print(f"Speech Recognition: {input_data} -> {result['text']}")
            success_counts["Successful"] += 1
        except Exception as e:
            success_counts["Failed"] += 1
            print(f"Error in {task}: {e}")
    
    # Visualization
    result_counts = Counter(task_results)
    plt.figure(figsize=(6, 4))
    plt.bar(result_counts.keys(), result_counts.values(), color='purple')
    plt.title("Vision and Audio Pipeline Results")
    plt.xlabel("Result Type")
    plt.ylabel("Count")
    plt.savefig("vision_audio_pipelines_output.png")
    print("Visualization: Vision/audio pipeline results saved as vision_audio_pipelines_output.png")

if __name__ == "__main__":
    vision_audio_pipelines()