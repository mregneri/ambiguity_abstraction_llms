All-Kinds Dissimilar Word Pipeline (Artificial Homonyms)

This folder contains the scripts used to identify dissimilar word pairs across all
word types (M/P/H) and prepare them for creating artificial homonyms.

Pipeline Order
	1.	allkinds_homonym_analysis.py
	    -	Computes cosine dissimilarities for all M/P/H pairs.
	    -	Saves the 1000 most dissimilar pairs for each corpus.
	2.	allkinds_select_homonym_pairs.py
	    -	Selects non-overlapping pairs by preferring words with few alternatives.
	    -	Produces a full set and a trimmed top-N set.
	3.	sort_homonym_pairs.py
	    -	Ensures the more frequent word is in word1.
	    -	Sorts pairs by ascending cosine similarity (most dissimilar first).
	    -	Creates “top-N for naming” files with an empty name column.
	4. Manual name creation, manual removal of pairs with overly related words
	    - Removing pair 15 (gold - exercise) in tiny_top50_homonym_pairs_with_names.csv
	    - Add pair 51 (lemon - alert) to tiny_top50_homonym_pairs_with_names.csv
	    - Save lists with names as tiny_top50_homonym_pairs_with_names.csv and
	      recipe_top50_homonym_pairs_with_names.csv
	5.	shuffle_and_check_homonyms.py
	    -	Uses the manually named top-N files.
	    -	Checks whether each artificial homonym name already occurs in the corpus.
	    -	Keeps the first row fixed and shuffles the rest.
	    -	Produces the final shuffled files and summary reports.
	    -   Final homonym lists: recipe_homonym_pairs_final_shuffled.csv
	        and tiny_homonym_pairs_final_shuffled.csv

Notes
	- Manual changes in tiny_top50_homonym_pairs_with_names.csv: pair 15 (gold - exercise) removed,
	  pair 51 (lemon - alert) added