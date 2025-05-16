import pandas as pd
import numpy as np
import json
import spacy
import dacy
from tqdm import tqdm
import os
from nlp_utils import find_descriptions

########## Dataset ##########

# Load and filter data
df_articles = pd.read_parquet("ebnerd_large/articles.parquet")   # Load parquet as dataframe
df = df_articles[df_articles["category_str"] == "underholdning"] # Filter to only use category "underholding" (entertainment)
df = df[df["body"].str.strip().astype(bool)].copy()              # Remove article with empty bodies



########## NLP pipeline ##########

# Force SpaCy to use GPU to speed up transformer matrix computations
spacy.require_gpu()

# Load large DaCy model and add NLP components
nlp = dacy.load("large")                       # Loads transformer model (incl. NER, coref, lemmatizer, POS tagger)
nlp.add_pipe("dacy/polarity")                  # Add sentiment analysis, polarity
nlp.add_pipe("dacy/emotionally_laden")         # Add emotion detection
nlp.add_pipe("dacy/emotion")                   # Add emotion classification
nlp.add_pipe("dacy/hatespeech_detection")      # Add hate speech detection
nlp.add_pipe("dacy/hatespeech_classification") # Add hate speech classification



########## Functions ##########

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    else:
        return obj



########## Chunking ##########

# Chunking and processing
chunk_size = 256
num_chunks = (len(df) + chunk_size - 1) // chunk_size  # Ceiling division

output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

for i in range(num_chunks):
    output_path = os.path.join(output_dir, f"articles_entertainment_step_{i+1}.parquet")
    
    # Skip chunk if already processed
    if os.path.exists(output_path):
        print(f"Skipping chunk {i+1}/{num_chunks}, file exists: {output_path}")
        continue

    print(f"Processing chunk {i+1}/{num_chunks}")
    chunk = df.iloc[i*chunk_size : (i+1)*chunk_size].copy()
    texts = chunk["body"].tolist()
    docs = nlp.pipe(texts)

    # Make lists
    ner_clusters = []
    ner_clusters_lemma = []
    entity_groups = []
    coref_clusters = []
    sentiment_scores = []
    sentiment_labels = []
    emotion = []
    hate_speech = []
    person_descriptions = []

    for doc in tqdm(docs, desc=f"Chunk {i+1}"):
        # NER
        ent_texts = [ent.text for ent in doc.ents]
        ent_texts_lemma = [ent.lemma_ for ent in doc.ents]
        ent_labels = [ent.label_ for ent in doc.ents]

        # Coreference resolution
        coref_data = {}
        for key, span_group in doc.spans.items():
            coref_data[key] = [span.text for span in span_group]

        # Sentiment analysis - polarity
        polarity_probs = doc._.polarity_prob # get polarity probabilities for negative, neutral and positive
        probabilities = polarity_probs["prob"] # get probs
        polarity_index = int(probabilities.argmax()) # find most probable sentiment
        polarity = polarity_probs["labels"][polarity_index] # get the corresponding sentiment label
        score = round(float(probabilities[polarity_index]), 4) # round the probability score

        # Emotion
        emotion_type = doc._.emotion

        # Hate speech classification
        hate_speech_type = doc._.hate_speech_type

        # Person descriptions (modifiers)
        per_desc = find_descriptions(doc)

        # Append NLP stuff
        ner_clusters.append(ent_texts)
        ner_clusters_lemma.append(ent_texts_lemma)
        entity_groups.append(ent_labels)
        coref_clusters.append(coref_data)
        sentiment_scores.append(score)
        sentiment_labels.append(polarity)
        emotion.append(emotion_type)
        hate_speech.append(hate_speech_type)
        person_descriptions.append(per_desc)

    chunk["ner_clusters"] = ner_clusters
    chunk["ner_clusters_lemma"] = ner_clusters_lemma
    chunk["entity_groups"] = entity_groups
    chunk["coref_clusters"] = coref_clusters
    chunk["sentiment_score"] = sentiment_scores
    chunk["sentiment_label"] = sentiment_labels
    chunk["emotion"] = emotion
    chunk["hate_speech"] = hate_speech
    chunk["person_descriptions"] = person_descriptions
    
    # Serialize dicts to JSON strings for Parquet compatibility
    #chunk["coref_clusters"] = chunk["coref_clusters"].apply(lambda x: json.dumps(to_serializable(x), ensure_ascii=False))
    #chunk["person_descriptions"] = chunk["person_descriptions"].apply(lambda x: json.dumps(to_serializable(x), ensure_ascii=False))

    chunk.to_parquet(output_path)