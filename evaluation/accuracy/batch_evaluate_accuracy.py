import subprocess

"""
batch_evaluate_accuracy.py
==========================
Calls evaluate_accuracy.py for multiple GPT models sequentially.

Usage:
    python batch_evaluate_accuracy.py
"""

# Configuration
model_bases = ["recipe", "tiny"]
hom_counts = [10, 20, 30, 40, 50]

for base in model_bases:
    for count in hom_counts:
        model_name = f"gpt_{base}_{count}_HOM"

        print(f"\n=== Running evaluate_accuracy.py for: {model_name} ===")
        cmd = ["python", "evaluate_accuracy.py", model_name]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"⚠️ Evaluation failed for: {model_name}")
        else:
            print(f"✅ Finished: {model_name}")

print(f"🎉 All evaluations completed.")