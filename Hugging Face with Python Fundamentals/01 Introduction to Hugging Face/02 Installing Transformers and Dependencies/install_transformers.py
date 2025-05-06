# %% install_transformers.py
# Setup: pip install transformers matplotlib pandas
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import pkg_resources

# Function to verify installation and run a sample task
def verify_installation():
    print("Verifying Hugging Face Transformers installation...")
    
    # Check installed packages
    required_packages = ["transformers", "matplotlib", "pandas"]
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    package_status = {"Installed": 0, "Not Installed": 0}
    
    for pkg in required_packages:
        if pkg in installed_packages:
            print(f"{pkg}: {installed_packages[pkg]}")
            package_status["Installed"] += 1
        else:
            print(f"{pkg}: Not installed")
            package_status["Not Installed"] += 1
    
    # Run a sample text classification task
    try:
        classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = classifier("Hugging Face is awesome!")
        print("Sample Task Result:", result)
        package_status["Installed"] += 1  # Count successful task as validation
    except Exception as e:
        print("Error running sample task:", e)
        package_status["Not Installed"] += 1
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.bar(package_status.keys(), package_status.values(), color=['green', 'red'])
    plt.title("Transformers Installation Status")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.savefig("install_transformers_output.png")
    print("Visualization: Installation status saved as install_transformers_output.png")

if __name__ == "__main__":
    verify_installation()