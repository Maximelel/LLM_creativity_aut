import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import pi
import string
import re

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

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

###############################################
###############################################

# Functions:


def elaboration_SW(df):
    df['elaboration_SW'] = df['response'].apply(count_words_wo_SW)
    return df

def count_words_wo_SW(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return len(tokens)

# POS tagging
def tag_sentence(sentence):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    return pos_tags

# Function to calculate the average number of specific POS tags in sentences
def analyze_pos_distribution(sentences):
    # Initialize counts for each POS tag
    noun_count = 0
    verb_count = 0
    adjective_count = 0
    adverb_count = 0
    determiner_count = 0
    preposition_count = 0
    other_count = 0
    total_sentences = len(sentences)
    
    # Iterate over each sentence and calculate POS tag counts
    for sentence in tqdm(sentences):
        pos_tags = tag_sentence(sentence)
        tag_counts = nltk.Counter(tag for word, tag in pos_tags)
        
        # Update counts
        noun_count += tag_counts['NN'] + tag_counts['NNS']  # Nouns
        verb_count += tag_counts['VB'] + tag_counts['VBD'] + tag_counts['VBG'] + tag_counts['VBN'] + tag_counts['VBP'] + tag_counts['VBZ']  # Verbs
        adjective_count += tag_counts['JJ']  # Adjectives
        adverb_count += tag_counts['RB'] + tag_counts['RBR'] + tag_counts['RBS']  # Adverbs
        determiner_count += tag_counts['DT'] # Determiners
        preposition_count += tag_counts['IN'] # Prepositions
        other_count += sum(tag_counts[tag] for tag in tag_counts if tag not in ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'RB', 'RBR', 'RBS', 'DT', 'IN'])
        
    
    # Calculate average POS tag counts per sentence
    avg_nouns = noun_count / total_sentences
    avg_verbs = verb_count / total_sentences
    avg_adjectives = adjective_count / total_sentences
    avg_adverbs = adverb_count / total_sentences
    avg_determiners = determiner_count / total_sentences
    avg_prepositions = preposition_count / total_sentences
    avg_other_count = other_count / total_sentences
    
    return avg_nouns, avg_verbs, avg_adjectives, avg_adverbs, avg_determiners, avg_prepositions, avg_other_count

def compute_pos_tagging(combined_data):
    df_pos = pd.DataFrame(columns = ['model', 'avg_nouns', 'avg_verbs', 'avg_adj', 'avg_adv', 'avg_det', 'avg_prepos', 'other'])
    #df_pos['model'] = ['Humans', 'GPT-3.5', 'GPT-4', 'Mistral', 'Vicuna']
    for i, model in enumerate(['Humans', 'GPT-3.5', 'GPT-4', 'Mistral', 'Vicuna']):
        sentences = combined_data[combined_data['dataset'] == model]['response']
        avg_nouns, avg_verbs, avg_adjectives, avg_adverbs, avg_determiners, avg_prepositions, avg_other_count = analyze_pos_distribution(sentences)
        df_pos.loc[i] = [model] + [avg_nouns, avg_verbs, avg_adjectives, avg_adverbs, avg_determiners, avg_prepositions, avg_other_count]
    return df_pos


################################################
################ VISUALIZATION #################
################################################

def plot_POS(combined_data, df_pos):
    list = []
    for i, model in enumerate(['Humans', 'GPT-3.5', 'GPT-4', 'Mistral', 'Vicuna']):
        sentences = combined_data[combined_data['dataset'] == model]['response']
        avg_nb_words = sum([len(sentence.split()) for sentence in sentences]) / len(sentences)
        list.append(avg_nb_words)
        
    # check
    print(f"POS tagging: {df_pos[df_pos.columns[1:]].sum(axis = 1).values}")
    print(f"Ground truth: {list}")

    # make 2 sublots with df_pos and plt_bar
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey = True)
    df_pos.plot(x='model', kind='bar', stacked=True, title='POS tagging Humans VS LLMs', ax=ax[0])
    ax[1].bar(['Humans', 'GPT-3.5', 'GPT-4', 'Mistral', 'Vicuna'], list)
    ax[1].set_title('Average number of words per sentence')
    plt.tight_layout()
    plt.show()

def plot_POS_proportions(df_pos):
    proportions_df = df_pos.copy()
    cols_to_transform = ['avg_nouns', 'avg_verbs', 'avg_adj', 'avg_adv', 'avg_det', 'avg_prepos', 'other']
    proportions_df[cols_to_transform] = proportions_df[cols_to_transform].div(proportions_df[cols_to_transform].sum(axis=1), axis=0).round(3)
    proportions_df.rename(columns = {'avg_nouns': 'Nouns', 'avg_verbs': 'Verbs', 'avg_adj': 'Adjectives', 'avg_adv': 'Adverbs', 'avg_det': 'Determiners', 'avg_prepos': 'Prepositions', 'other': 'Other'}, inplace = True)

    # plot
    ax = proportions_df.plot(x='model', kind='bar', stacked=True, title='POS tagging Humans VS LLMs (in %)', figsize=(10, 7))
    # rotate x axis
    plt.xticks(rotation=0)

    # Put values on the bars
    for container in ax.containers:
        ax.bar_label(container, label_type='center')
                
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.show()
