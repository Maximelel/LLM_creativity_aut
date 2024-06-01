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

# import gensim
from pprint import pprint# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel# spaCy for preprocessing
import spacy# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
from gensim.models import LdaModel

###############################################
###############################################

# Functions:
# - clean
# - create_lda_model
# - generate_lda_models
# - kw_in_sentence
# - assign_topic_all
# - compute_coeff_topic_object
# - compute_coeff_topic_all_objects
# - run_LDA_on_humans_data
# - plot_originality_per_topic
# - visu_with_pyldavis

# clean text
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop_words])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def create_lda_model(df, object, num_topics):
    # create corpus
    texts = [sent for sent in df[df['prompt'] == object]['response']]

    clean_texts = [clean(text).split() for text in texts]
    print(f"Number of documents in corpus for object \"{object}\": {len(clean_texts)}")
    
    # Creating dictionary
    dictionary = corpora.Dictionary(clean_texts)
    # create term document frequency
    corpus = [dictionary.doc2bow(text) for text in clean_texts]
    
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=num_topics, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
    
    return lda_model

def generate_lda_models(df, num_topics):
    '''
    Generates the LDA models for each objects in df with num_topics topics
    
    input:
    - df: pandas dataframe
    - num_topics: Number of topics to run LDA algorithm

    output:
    - lda_model_list: list of LDA models for each object
    '''
    lda_model_brick = create_lda_model(df, 'brick', num_topics=num_topics)
    lda_model_box = create_lda_model(df, 'box', num_topics=num_topics)
    lda_model_knife = create_lda_model(df, 'knife', num_topics=num_topics)
    lda_model_rope = create_lda_model(df, 'rope', num_topics=num_topics)

    lda_model_list = [lda_model_brick, lda_model_box, lda_model_knife, lda_model_rope]
    
    return lda_model_list
    
def kw_in_sentence(sentence, keywords):
    return any(word in sentence for word in keywords)

def assign_topic_all(df_model, lda_model_list, print_keywords, num_topics, num_words):

    df_output = pd.DataFrame()
    df_kw_per_topic = pd.DataFrame(columns = ['object', 'topic', 'keywords'])#, 'coherence score'])
    objects = ['brick', 'box', 'knife', 'rope']
    # evaluate performance
    perplexity = []
    coherence_score = []
    coherence_score_per_topic = []
    
    for i, lda_model in enumerate(lda_model_list):
        object = objects[i]
        df_object = df_model[df_model['prompt'] == object]
        # initialize column topic
        df_object['topic'] = None
        for j in range(num_topics):
            topic_keywords = [w[0] for w  in lda_model.show_topic(topicid = j, topn = num_words)]
            df_kw_per_topic.loc[len(df_kw_per_topic)] = [object, j, topic_keywords]
            if print_keywords:
                print(f"Object: {object}, Topic {j+1}, Keywords: {topic_keywords}")

            mask = df_object['response'].apply(lambda x: kw_in_sentence(x, topic_keywords))
            df_object.loc[mask, 'topic'] = j 
        if print_keywords:
            print("\n")

        df_output = pd.concat([df_output, df_object])
        
        # Evaluate performance
        #print("Evaluate LDA model")
        # create corpus
        texts = [sent for sent in df_model[df_model['prompt'] == object]['response']]
        clean_texts = [clean(text).split() for text in texts]
        # Creating dictionary
        dictionary = corpora.Dictionary(clean_texts)
        # create term document frequency
        corpus = [dictionary.doc2bow(text) for text in clean_texts]
        
        # Compute Perplexity: a measure of how good the model is. lower the better.
        perplexity.append(lda_model.log_perplexity(corpus))  
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=clean_texts, dictionary=dictionary, coherence='c_v')
        coherence_score.append(coherence_model_lda.get_coherence())
        #df_kw_per_topic.loc[df_kw_per_topic, 'coherence score'] = coherence_model_lda.get_coherence()
        coherence_score_per_topic.append(coherence_model_lda.get_coherence_per_topic())

    return df_output, df_kw_per_topic, perplexity, coherence_score, coherence_score_per_topic

def compute_coeff_topic_object(df_kw_per_topic, humans, object, num_topics):
    topic_freq = [0]*num_topics
    df_object = df_kw_per_topic[df_kw_per_topic['object'] == object]
    humans_object = humans[humans['prompt'] == object]
    for N in range(num_topics):
        keywords = df_object[df_object['topic'] == N]['keywords'].tolist()[0]
        
        # add 1 if there is at least 1 keyword in the sentence
        #topic_freq[N] = humans_object['response'].apply(lambda x: sum([1 for kw in keywords if kw in x])).sum()
        topic_freq[N] = humans_object['response'].apply(lambda x: sum([1 for kw in keywords if kw in x]) > 0).sum()
    print(f"Frequency of topics: {topic_freq}")
    # softmax to get probabilities
    topic_prob = [freq/sum(topic_freq) for freq in topic_freq]
    topic_coeff = 1 - np.array(topic_prob)
    return topic_coeff

def compute_coeff_topic_all_objects(df_kw_per_topic, humans, num_topics):
    dict_topic_coeff = {"brick": compute_coeff_topic_object(df_kw_per_topic, humans, 'brick', num_topics),
                    "box": compute_coeff_topic_object(df_kw_per_topic, humans, 'box', num_topics),
                    "knife": compute_coeff_topic_object(df_kw_per_topic, humans, 'knife', num_topics),
                    "rope": compute_coeff_topic_object(df_kw_per_topic, humans, 'rope', num_topics)}
    return dict_topic_coeff

def run_LDA_on_humans_data(df, num_topics, num_words, print_keywords, objects):
    # generate the LDA models for each object
    lda_model_list = generate_lda_models(df, num_topics)
    
    # assign topic to all sentences in df
    humans_topic,  df_kw_per_topic, perplexity, coherence_score, coherence_score_per_topic = assign_topic_all(df, lda_model_list, print_keywords = print_keywords, num_topics = num_topics, num_words = num_words)
    df_kw_per_topic['coherence_score'] = np.array(coherence_score_per_topic).reshape(-1)
    # compute the coefficient for each topic for each object
    dict_topic_coeff = compute_coeff_topic_all_objects(df_kw_per_topic, df, num_topics)
    
    # compute the originality with score from topic coefficients
    dict_kw_coeff = []
    for object in objects:
        for i in range(num_topics):
            #dict_kw_coeff.append({"keywords": df_kw_per_topic[df_kw_per_topic['object'] == object]['keywords'][i], "coeff": dict_topic_coeff[object][i]})
            keywords = df_kw_per_topic[(df_kw_per_topic['object'] == object) & (df_kw_per_topic['topic'] == i)]['keywords'].tolist()[0]
            coeff = dict_topic_coeff[object][i]
            dict_kw_coeff.append({"object": object, "topic": i, "keywords": keywords, "coeff": coeff})
            
    return dict_kw_coeff


def compute_flexibility_per_sentence(sentence, dict_kw_coeff_object, num_topics):
    flex_score = 0
    for i in range(num_topics):
        keywords = dict_kw_coeff_object[i]['keywords']
        is_topic_covered = 0
        for kw in keywords:
            if kw in sentence.lower().split():
                is_topic_covered = 1
        flex_score += is_topic_covered 
    return flex_score

def compute_flexibility_augmented_per_sentence(sentence, dict_kw_coeff_object, num_topics):
    flex_score_augmented = 0
    for i in range(num_topics):
        keywords = dict_kw_coeff_object[i]['keywords']
        N_words_in_topic = 0
        for kw in keywords:
            if kw in sentence.lower().split():
                N_words_in_topic += 1 # count the number of times the sentence contains a keyword of the topic
        flex_score_augmented += N_words_in_topic * dict_kw_coeff_object[i]['coeff'] # sum
    return flex_score_augmented

# flexibility score: number of topics covered by each sentence
def compute_flexibility_score(df, dict_kw_coeff, num_topics, objects):
    df_output = pd.DataFrame()
    for i, object in enumerate(objects):
        df_tmp = df[df['prompt'] == object]
        df_tmp['flexibility'] = df_tmp['response'].apply(lambda x: compute_flexibility_per_sentence(x, dict_kw_coeff[i*num_topics:(i+1)*num_topics], num_topics))
        df_output = pd.concat([df_output, df_tmp])
    return df_output

# flexibility augmented score: number of topics covered by each sentence multipled by a coefficient inversely proportional to the frequency of the topic in the humans data
def compute_flexibility_augmented_score(df, dict_kw_coeff, num_topics, objects):
    df_output = pd.DataFrame()
    for i, object in enumerate(objects):
        df_tmp = df[df['prompt'] == object]
        df_tmp['flexibility_augmented'] = df_tmp['response'].apply(lambda x: compute_flexibility_augmented_per_sentence(x, dict_kw_coeff[i*num_topics:(i+1)*num_topics], num_topics))
        df_output = pd.concat([df_output, df_tmp])
    return df_output

################################################
################ VISUALIZATION #################
################################################

def plot_originality_per_topic(df, lda_model_list, name_model, print_keywords, num_topics):
    
    objects = ['brick', 'box', 'knife', 'rope']
    
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    #df_keywords = pd.DataFrame(columns = ['topic', 'keywords', 'originality', 'elaboration', 'elaboration_SW', 'cosine_dist'])
    
    # evaluate performance
    perplexity = []
    coherence_score = []
    
    for i, lda_model in enumerate(lda_model_list):
        object = objects[i]
        for j in range(num_topics):
            topic_keywords = [w[0] for w  in lda_model.show_topic(topicid = j, topn = 5)]
            if print_keywords:
                print(f"Object: {object}, Topic {j+1}, Keywords: {topic_keywords}")

            # keep only sentences in humans['response'] that have those words
            texts = [sent for sent in df[df['prompt'] == object]['response']]
            mask = [any(word in text for word in topic_keywords) for text in texts]
            
            #plot
            sns.histplot(df[df['prompt'] == object][mask]['originality'], kde = True, ax = axs[i], label = f"Topic {j+1}")
            axs[i].set_title(f"{object}")
        if print_keywords:
            print("\n")
            
        # Evaluate performance
        #print("Evaluate LDA model")
        # create corpus
        texts = [sent for sent in df[df['prompt'] == object]['response']]
        clean_texts = [clean(text).split() for text in texts]
        # Creating dictionary
        dictionary = corpora.Dictionary(clean_texts)
        # create term document frequency
        corpus = [dictionary.doc2bow(text) for text in clean_texts]
        
        # Compute Perplexity: a measure of how good the model is. lower the better.
        perplexity.append(lda_model.log_perplexity(corpus))  
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=clean_texts, dictionary=dictionary, coherence='c_v')
        coherence_score.append(coherence_model_lda.get_coherence())
            
    for j in range(4):
        axs[j].legend()
    plt.suptitle(f"Originality per topic for {name_model} for {num_topics} topics, Perplexity: {np.array(perplexity).mean().round(3)}, Coherence score: {np.array(coherence_score).mean().round(3)}", fontsize = 16) 
    plt.tight_layout()
    plt.show()
    
def visu_with_pyldavis(lda_model, df, object):
    # create corpus
    texts = [sent for sent in df[df['prompt'] == object]['response']]
    clean_texts = [clean(text).split() for text in texts]
    # Creating dictionary
    dictionary = corpora.Dictionary(clean_texts)
    # create term document frequency
    corpus = [dictionary.doc2bow(text) for text in clean_texts]
    
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    
    return vis

