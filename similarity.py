import torch
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

###############################################
###############################################

# Functions:
# - encode_text
# - compute_cosine_distance_matrix
# - compute_avg_cosine_similarity
# - compute_sentences_similarity
# - compute_sentences_sim_per_object

# Function to encode text into embeddings using a pre-trained language model
def encode_text(texts, model_name):
    '''
    Transform each sentence into embeddings
    inputs: 
        - texts: list of sentences
        - model_name: model to use for embeddings
    output:
        - embeddings: torch tensor of embeddings of size [# sentences, embedding dim]
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input texts and generate embeddings
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_input)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling to get sentence embeddings
    return embeddings

# Function to compute cosine distance matrix
def compute_cosine_distance_matrix(embeddings):
    cosine_dist_matrix = cosine_distances(embeddings.numpy())
    return cosine_dist_matrix

# Function to compute average cosine similarity for each sentence
def compute_avg_cosine_similarity(cosine_dist_matrix):
    num_sentences = cosine_dist_matrix.shape[0]
    avg_cosine_similarities = np.zeros(num_sentences)

    for i in tqdm(range(num_sentences)):
        avg_cosine_similarities[i] = np.mean(np.delete(cosine_dist_matrix[i], i))  # Exclude self-similarity

    return avg_cosine_similarities

def compute_sentences_similarity(df, model_name):
    # Encode the textual answers into embeddings
    embeddings = encode_text(df['response'].tolist(), model_name)

    # Compute cosine distance matrix
    cosine_dist_matrix = compute_cosine_distance_matrix(embeddings)

    # Compute average cosine similarity for each sentence
    avg_cosine_similarities = compute_avg_cosine_similarity(cosine_dist_matrix)
    
    return avg_cosine_similarities

def compute_sentences_sim_per_object(df, model_name):
    for object in df['prompt'].unique():
        avg_cosine_similarities = compute_sentences_similarity(df[df['prompt'] == object], model_name)
        df.loc[df['prompt'] == object, 'similarity'] = avg_cosine_similarities
    
    return df
