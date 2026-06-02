import os
import re
import pandas as pd

# Base directory where all the model folders live
BASE_DIR = "/home/nina/ambiguity/gpt_pipelines/data"

# Regex patterns
doc_pattern = re.compile(r"(Train|Val|Test) documents: (\d+)")
tok_pattern = re.compile(r"(Train|Val|Test) tokens: ([\d,]+) \(([\d\.]+)%\)")

rows = []

# Walk through all model subfolders
for subdir in sorted(os.listdir(BASE_DIR)):
    log_path = os.path.join(BASE_DIR, subdir, "prepare_log.txt")
    if not os.path.isfile(log_path):
        continue  # skip if no log file present

    with open(log_path, "r") as f:
        content = f.read()

    # Extract document counts
    docs = {m.group(1): int(m.group(2)) for m in doc_pattern.finditer(content)}
    # Extract token counts + percentages
    toks = {}
    props = {}
    for m in tok_pattern.finditer(content):
        split, count, pct = m.groups()
        toks[split] = int(count.replace(",", ""))
        props[split] = float(pct)

    total_tokens = sum(toks.values())

    rows.append({
        "Dataset": subdir,
        "Train docs": docs.get("Train", 0),
        "Val docs": docs.get("Val", 0),
        "Test docs": docs.get("Test", 0),
        "Train tokens": toks.get("Train", 0),
        "Val tokens": toks.get("Val", 0),
        "Test tokens": toks.get("Test", 0),
        "Total tokens": total_tokens,
        "Train %": props.get("Train", 0),
        "Val %": props.get("Val", 0),
        "Test %": props.get("Test", 0),
    })

# Create a table
df = pd.DataFrame(rows)
df = df.sort_values("Dataset").reset_index(drop=True)

# Save to CSV for easy viewing
out_path = os.path.join(BASE_DIR, "gpt_logs_summary.csv")
df.to_csv(out_path, index=False)

print(f"✅ Summary written to {out_path}")
print(df.head(10))  # preview first rows