# Basics
import pandas as pd
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Scraping
import requests
from bs4 import BeautifulSoup

# Utilities
from imdb_utils import *

# Genderize
from utils import classify_gender

# Global variables
reality_stars_dataset_path = "dataset/reality_stars.parquet"



def get_reality_titles():
    # List of reality shows
    urls = [
        "https://www.imdb.com/search/title/?genres=reality-tv&countries=DK",                    # Top 25 most popular reality TV shows in Denmark
        "https://www.imdb.com/search/title/?genres=reality-tv&countries=DK&sort=num_votes,desc" # Top 25 most rated reality TV shows in Denmark
    ]

    # Make a titles list
    all_titles = []

    # Iterate through each URL to grab titles
    for url in urls:
        soup = get_soup(url)
        all_titles.extend(get_titles(soup))

    # Turn titles into DataFame and remove duplicates
    df_titles = pd.DataFrame(all_titles).drop_duplicates(subset="id").reset_index(drop=True)

    return df_titles

def get_reality_stars(max_workers=10):
    if os.path.isfile(reality_stars_dataset_path):
        return print("Reality stars dataset already exists.")

    # Get reality titles
    df_titles = get_reality_titles()

    # Make a cast list
    all_cast = []

    # Instead of looping one request at a time, fetch from multiple pages at a time
    # e.g. with max_workers=10, one fetches 10 pages at a time
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_cast, title): title for title in df_titles["id"]}
        for future in as_completed(futures):
            try:
                all_cast.extend(future.result())
            except Exception as e:
                print(f"Error fetching cast for {futures[future]}: {e}")
    
    # Collect reality stars in a DataFrame
    df_cast = pd.DataFrame(all_cast)

    # Remove duplicates and assign to a star all the titles they've been in
    df_people = (
        df_cast
        .drop_duplicates(subset=["id", "title"])                             # drop dupes
        .groupby(["id", "name"])["title"]                                    # group by "id", "name", "title"
        .apply(list)
        .reset_index()                                                       # reset indexing after dropping dupes
        .sort_values(by="title", key=lambda x: x.str.len(), ascending=False) # sort by number of titles
        .reset_index(drop=True)                                              # reset indexing one last time
    )

    # Save dataset as file
    df_people.to_parquet(reality_stars_dataset_path)

    return df_people

def genderize_reality_stars():
    # Get dataset
    if not os.path.isfile(reality_stars_dataset_path):
        reality_stars = get_reality_stars()
    else:
        reality_stars = pd.read_parquet(reality_stars_dataset_path)

    # Extract first names
    reality_stars["first_name"] = reality_stars["name"].str.split().str[0]

    # Get unique first names (saves computations)
    unique_first_names = reality_stars["first_name"].unique().tolist()

    # Generate gender results with rule fallback
    gender_results = [{"first_name": name, "gender": classify_gender(name)} for name in unique_first_names]

    # Create DataFrame
    gender_df = pd.DataFrame(gender_results)

    # Merge with original dataset
    reality_stars = reality_stars.merge(gender_df, on="first_name", how="left")

    # Remove rows where gender is still unknown
    reality_stars = reality_stars[reality_stars["gender"] != "unknown"].reset_index(drop=True)

    # Save as file
    reality_stars.to_parquet("dataset/reality_stars_genderized.parquet")

    return reality_stars