import torch
import string
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

# Download NLTK resources
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  
nltk.download('omw-1.4')  
nltk.download('averaged_perceptron_tagger')

# Define stop words, lemmatizer
stop_words = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

from elaboration import *
from similarity import *
from flexibility import *


# Compute Elaboration 
# with or without stop_words
def calculate_elaboration(sentence, remove_stop_words):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    
    if remove_stop_words:
        # Remove stop words
        tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Count the remaining words
    return len(tokens)

# Compute all metrics for humans at once
def compute_all_metrics(df, objects, dict_kw_coeff, num_topics):
    
    # Compute Elaboration
    print("Compute elaboration...")
    df['elaboration'] = df['response'].apply(lambda x: calculate_elaboration(x, remove_stop_words = False))
    print("Compute elaboration without stop words...")
    df['elaboration_SW'] = df['response'].apply(lambda x: calculate_elaboration(x, remove_stop_words = True))
    
    # Compute Similarity
    #print("Compute similarity...")
    #embeddings_model_name = "distilbert-base-uncased"
    #df['similarity'] = compute_sentences_sim_per_object(df, embeddings_model_name)
    
    # Compute Flexibility
    print("Compute flexibility...")
    df = compute_flexibility_score(df, dict_kw_coeff, num_topics, objects)
    
    return df