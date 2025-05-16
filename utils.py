import gender_guesser.detector as gender
import pandas as pd
import glob
import os

# Gender detector
d = gender.Detector()

def classify_gender(name):
    # Initial guess
    g = d.get_gender(name=name)

    if g != "unknown":
        return g
    
    # Fallback heuristic
    if name[-1].lower() in "aeiouy": # if last character in name is a vowel it's likely female
        return "female"
    elif name[-1].lower() in "korst": # if last character in name is one of these, it's likely male
        return "male"
    else:
        return "unknown"

def chunks_to_df():
    # Path to the output directory
    output_dir = "dataset"

    # Find all step parquet files and sort them by chunk number
    chunk_files = sorted(
        glob.glob(os.path.join(output_dir, "articles_entertainment_step_*.parquet")),
        key=lambda x: int(x.split("_step_")[-1].split(".")[0])
    )

    # Load and concatenate all the chunks
    dfs = []
    for f in chunk_files:
        print(f"Loading {f}")
        df_chunk = pd.read_parquet(f)
        dfs.append(df_chunk)

    # Combine into one DataFrame
    df_combined = pd.concat(dfs, ignore_index=True)

    # Save the full dataset
    combined_path = os.path.join(output_dir, "articles_entertainment_nlp.parquet")
    df_combined.to_parquet(combined_path)
    print(f"Combined DataFrame saved to {combined_path}")
