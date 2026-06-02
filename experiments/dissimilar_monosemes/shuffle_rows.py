import pandas as pd

"""
shuffle_rows.py
==========================

This script reorders the rows of CSV files containing monoseme word pairs and their cosine similarity scores.

For each input file, the script:
- Keeps the most dissimilar pair (i.e., the pair with the lowest cosine similarity) at the top.
- Randomly shuffles the order of the remaining rows.
- Saves the reordered table to a new CSV file.

Input:
------
CSV files with columns including at least: 'word1', 'word2', and 'cosine_similarity'.

Output:
-------
New CSV files with:
- The top row as the most dissimilar pair.
- All other rows shuffled.

Files:
------
- recipe_artificial_homonyms_sorted.csv → recipe_artificial_homonyms_shuffled.csv
- tiny_artificial_homonyms_sorted.csv → tiny_artificial_homonyms_shuffled.csv
"""

# File paths
INPUT_OUTPUT_FILES = [
    ("recipe_artificial_homonyms_sorted.csv", "recipe_artificial_homonyms_shuffled.csv"),
    ("tiny_artificial_homonyms_sorted.csv", "tiny_artificial_homonyms_shuffled.csv")
]

def reorder_csv(input_path, output_path):
    # Load the CSV file
    df = pd.read_csv(input_path)

    # Find and keep the row with the lowest cosine_similarity
    top_row = df.loc[df['cosine_similarity'].idxmin()]
    remaining_rows = df.drop(df['cosine_similarity'].idxmin())

    # Shuffle the remaining rows
    shuffled_rows = remaining_rows.sample(frac=1, random_state=42).reset_index(drop=True)

    # Combine and save
    reordered_df = pd.concat([top_row.to_frame().T, shuffled_rows], ignore_index=True)
    reordered_df.to_csv(output_path, index=False)
    print(f"Saved reordered file: {output_path}")

if __name__ == "__main__":
    for input_path, output_path in INPUT_OUTPUT_FILES:
        reorder_csv(input_path, output_path)