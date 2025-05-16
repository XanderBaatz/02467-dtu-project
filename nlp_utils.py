import pandas as pd
import re
from collections import defaultdict
import collections
from rapidfuzz import fuzz

# Function to find person descriptions
def find_descriptions(doc):
    per_descriptions = defaultdict(list)

    for sent in doc.sents:
        # Find all PER entities in the sentence
        pers_in_sent = [ent for ent in sent.ents if ent.label_ == "PER"]

        # For person in sentence
        for per in pers_in_sent:
            per_token_idxs = set(range(per.start, per.end)) # get the index range of person, i.e. where the person is positioned in a sentence

            for token in sent:
                # Get tokens related to person using the indexing
                if token.i in per_token_idxs:
                    # Look for adjectives and noun modifiers in children
                    for child in token.children:
                        if (child.dep_ in {"amod", "appos"} or child.pos_ in {"ADJ", "NOUN"}) and child.ent_type_ != "PER":
                            per_descriptions[per.lemma_].append(child.lemma_.lower())

                    # Look at head too
                    if (token.head.dep_ in {"amod", "appos"} or token.head.pos_ in {"ADJ", "NOUN"}) and token.head.ent_type_ != "PER":
                        per_descriptions[per.lemma_].append(token.head.lemma_.lower())

    return per_descriptions



# Function to clean person names before alias assignment
def clean_person_name(name):
    # Remove leading/trailing whitespace and newlines
    name = name.strip().replace('\n', ' ')
    
    # Remove common leading/trailing special characters
    name = re.sub(r"^[\W_]+|[\W_]+$", "", name)
    
    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)
    
    # Remove strings that look like product info, headlines, or descriptions
    if re.search(r"\d{3,}", name) or re.search(r"kr\.", name) or len(name.split()) > 6:
        return None  # Considered not a name
    
    # Optional: validate that name has at least one word starting with a capital letter
    if not re.search(r"[A-ZÆØÅ][a-zæøå]+", name):
        return None
    
    # Normalize to title case
    return name.title()

def clean_list(lst):
    return [n for n in (clean_person_name(name) for name in lst) if n is not None]



# Function to perform alias assignment
def is_valid_name(name):
    return bool(re.search(r'[a-zA-Z]', name))

def merge_aliases_in_article(names, threshold):
    names = sorted([n for n in names if is_valid_name(n)], key=len, reverse=True)
    alias_map = {}
    used = set()

    for i, name in enumerate(names):
        if name in used:
            continue
        group = [name]
        for other in names[i + 1:]:
            if other in used:
                continue
            if fuzz.WRatio(name, other) >= threshold:
                group.append(other)
                used.add(other)
        alias_map[name] = group
        used.add(name)

    return alias_map

def process_person_entities(df, name_col='persons', desc_col='person_descriptions', coref_col='coref_clusters', article_id_col='article_id', threshold=85):
    """
    Process a DataFrame to merge aliases, descriptions, and coreference clusters for person entities.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing person-related data.
        name_col (str): Column name for person names list.
        desc_col (str): Column name for person descriptions.
        coref_col (str): Column name for coreference clusters.
        article_id_col (str): Column name for article IDs.
        threshold (int): Similarity threshold for alias merging using fuzzy matching.
    
    Returns:
        pd.DataFrame: A DataFrame with canonical names, aliases, descriptions, and coref clusters.
    """

    # Initialize the person_dict to store data about each person
    person_dict = defaultdict(lambda: {
        'articles': set(),
        'descriptions': defaultdict(list),
        'coref_clusters': defaultdict(dict),
        'aliases': set() # Store aliases directly here
    })

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        article_id = row[article_id_col]
        people = row.get(name_col, [])
        descs = row.get(desc_col, {})
        coref_clusters = row.get(coref_col, {})

        # Get alias map for the current row's people
        alias_map = merge_aliases_in_article(people, threshold=threshold)

        # Process each canonical name and its aliases
        for canonical, aliases in alias_map.items():
            person_data = person_dict[canonical]
            person_data['articles'].add(article_id)

            # Merge descriptions for each alias
            for alias in aliases:
                if alias in descs:
                    descriptions = descs[alias]
                    if isinstance(descriptions, list):
                        person_data['descriptions'][article_id].extend(descriptions)
                    else:
                        person_data['descriptions'][article_id].append(descriptions)
            
            # Merge coref clusters if any alias is mentioned
            for cluster_id, mentions in coref_clusters.items():
                if any(any(alias in mention for alias in aliases) for mention in mentions):
                    person_data['coref_clusters'][article_id][cluster_id] = mentions

            # Store aliases for this canonical name
            person_data['aliases'].update(aliases)

    # ---- Final DataFrame ----
    final_rows = []
    for person, data in person_dict.items():
        # Prepare the aliases as a sorted list
        aliases = sorted(data['aliases'])

        final_rows.append({
            'canonical_name': person, # Store the canonical name in its own column
            'aliases': aliases, # Store the aliases in a list
            'article_ids': sorted(data['articles']),
            'person_descriptions': dict(data['descriptions']),
            'coref_clusters': dict(data['coref_clusters'])
        })

    result_df = pd.DataFrame(final_rows)

    # Sort the DataFrame by the canonical name
    return result_df.sort_values(by='canonical_name').reset_index(drop=True)



# Genderization based on coreference mentions
def genderization(result_df):
    """
    Infers gender for each person in the result DataFrame based on pronoun usage in coreference mentions.

    Args:
        result_df (pd.DataFrame): DataFrame with columns 'canonical_name', 'aliases', and 'coref_clusters'.

    Returns:
        pd.DataFrame: Updated DataFrame with an added 'gender' column.
    """

    # First, define pronouns
    male_pronouns = {"han", "ham", "hans"}
    female_pronouns = {"hun", "hende", "hendes"}

    first_name_pronouns = defaultdict(collections.Counter)

    # Next, build global pronoun statistics by first name of canonical name and all aliases
    for _, row in result_df.iterrows():
        names = row['aliases']
        coref_clusters = row['coref_clusters']

        first_names = set(n.split()[0] for n in names if isinstance(n, str) and n)

        for first_name in first_names:
            for article_clusters in coref_clusters.values():
                for mentions in article_clusters.values():
                    for mention in mentions:
                        words = re.findall(r'\w+', mention.lower())
                        first_name_pronouns[first_name].update(words)

    # Then infer gender per first name
    first_name_gender = {}
    for name, counter in first_name_pronouns.items():
        male_count = sum(counter[p] for p in male_pronouns)
        female_count = sum(counter[p] for p in female_pronouns)

        if male_count > female_count:
            first_name_gender[name] = 'male'
        elif female_count > male_count:
            first_name_gender[name] = 'female'
        else:
            first_name_gender[name] = 'unknown'

    # Helper to assign gender based on aliases
    def assign_gender_by_first_name(canonical_name, aliases):
        all_first_names = [n.split()[0] for n in [canonical_name] + aliases if isinstance(n, str) and n]
        for fn in all_first_names:
            if fn in first_name_gender and first_name_gender[fn] != 'unknown':
                return first_name_gender[fn]
        return 'unknown'

    # Apply gender assignment
    result_df['gender'] = result_df.apply(
        lambda row: assign_gender_by_first_name(row['canonical_name'], row['aliases']),
        axis=1
    )

    return result_df
