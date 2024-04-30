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

################################################
################# ELABORATION ##################
################################################

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

################################################
################# FLEXIBILITY ##################
################################################

# topic modeling

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

def plot_originality_per_topic(df, lda_model_list, model, print_keywords, num_topics):
    
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
    plt.suptitle(f"Originality per topic for {model} for {num_topics} topics, Perplexity: {np.array(perplexity).mean().round(3)}, Coherence score: {np.array(coherence_score).mean().round(3)}", fontsize = 16) 
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

def kw_in_sentence(sentence, keywords):
    return any(word in sentence for word in keywords)

def assign_topic_all(df_model, lda_model_list, print_keywords, num_topics):

    df_output = pd.DataFrame()
    model_list = ['Humans', 'GPT-3.5', 'GPT-4', 'Mistral', 'Vicuna']
    objects = ['brick', 'box', 'knife', 'rope']
    # evaluate performance
    perplexity = []
    coherence_score = []
    
    for i, lda_model in enumerate(lda_model_list):
        object = objects[i]
        df_object = df_model[df_model['prompt'] == object]
        # initialize column topic
        df_object['topic'] = None
        for j in range(num_topics):
            topic_keywords = [w[0] for w  in lda_model.show_topic(topicid = j, topn = 5)]
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

    return df_output, perplexity, coherence_score


################################################
################ VISUALIZATION #################
################################################

# RADAR CHARTS

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

########################################
####### 1 radar chart per model ########
########################################
def example_data_per_model(humans_norm, gpt_35_100_norm, gpt_4_100_norm, mistral_norm, vicuna_norm, features):
    data = [
        features,
        ('Humans', [
            #humans_norm[humans_norm['prompt'] == 'brick'][["originality", "elaboration", "elaboration_SW", "cosine_dist"]].mean().values, # brick
            humans_norm[humans_norm['prompt'] == 'brick'][features].mean().values, # brick
            humans_norm[humans_norm['prompt'] == 'box'][features].mean().values, # box
            humans_norm[humans_norm['prompt'] == 'knife'][features].mean().values, # knife
            humans_norm[humans_norm['prompt'] == 'rope'][features].mean().values]), # rope
        ('GPT-3', [
            gpt_35_100_norm[gpt_35_100_norm['prompt'] == 'brick'][features].mean().values,
            gpt_35_100_norm[gpt_35_100_norm['prompt'] == 'box'][features].mean().values,
            gpt_35_100_norm[gpt_35_100_norm['prompt'] == 'knife'][features].mean().values,
            gpt_35_100_norm[gpt_35_100_norm['prompt'] == 'rope'][features].mean().values]),
        ('GPT-4', [
            gpt_4_100_norm[gpt_4_100_norm['prompt'] == 'brick'][features].mean().values,
            gpt_4_100_norm[gpt_4_100_norm['prompt'] == 'box'][features].mean().values,
            gpt_4_100_norm[gpt_4_100_norm['prompt'] == 'knife'][features].mean().values,
            gpt_4_100_norm[gpt_4_100_norm['prompt'] == 'rope'][features].mean().values]),
        ('Mistral', [
            mistral_norm[mistral_norm['prompt'] == 'brick'][features].mean().values,
            mistral_norm[mistral_norm['prompt'] == 'box'][features].mean().values,
            mistral_norm[mistral_norm['prompt'] == 'knife'][features].mean().values,
            mistral_norm[mistral_norm['prompt'] == 'rope'][features].mean().values]),
        ('Vicuna', [
            vicuna_norm[vicuna_norm['prompt'] == 'brick'][features].mean().values,
            vicuna_norm[vicuna_norm['prompt'] == 'box'][features].mean().values,
            vicuna_norm[vicuna_norm['prompt'] == 'knife'][features].mean().values,
            vicuna_norm[vicuna_norm['prompt'] == 'rope'][features].mean().values])
    ]
    return data

def radar_charts_per_model(humans_norm, gpt_35_100_norm, gpt_4_100_norm, mistral_norm, vicuna_norm, features):
    N = len(features)
    theta = radar_factory(N, frame='polygon')
    data_visu = example_data_per_model(humans_norm, gpt_35_100_norm, gpt_4_100_norm, mistral_norm, vicuna_norm, features)
    spoke_labels = data_visu.pop(0)
    fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    #colors = ['b', 'o', 'g', 'r', 'y']
    colors = ['b', 'r', 'g', 'm', 'y']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data_visu):
        #ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        #ax.set_rgrids([1, 2, 3, 4, 5])
        ax.set_title(title, weight='bold', size='xx-large', position=(0.5, 1.1),
                    horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels)
        
        ## Line to set the grid
        ax.set_rticks(np.linspace(0, 0.6, 4))  # Set radial ticks from 0 to 1 with 5 intervals
    # add legend relative to top-left plot
    labels = ('Brick', 'Box', 'Knife', 'Rope')
    legend = axs[0, 0].legend(labels, loc=(0.9, .95),
                            labelspacing=0.1, fontsize='x-large')
    fig.text(0.5, 0.965, 'Creativity comparison per model',
            horizontalalignment='center', color='black', weight='bold',
            size='xx-large')
    plt.tight_layout()
    plt.show()
        

########################################
####### 1 radar chart per object #######
########################################
def example_data_per_object(humans_norm, gpt_35_100_norm, gpt_4_100_norm, mistral_norm, features):
    data = [
        features,
        ('Brick', [
            humans_norm[humans_norm['prompt'] == 'brick'][features].mean().values, # Humans
            gpt_35_100_norm[gpt_35_100_norm['prompt'] == 'brick'][features].mean().values, # gpt-3.5
            gpt_4_100_norm[gpt_4_100_norm['prompt'] == 'brick'][features].mean().values, # gpt-4
            mistral_norm[mistral_norm['prompt'] == 'brick'][features].mean().values]), # mistral
        ('Box', [
            humans_norm[humans_norm['prompt'] == 'box'][features].mean().values,
            gpt_35_100_norm[gpt_35_100_norm['prompt'] == 'box'][features].mean().values,
            gpt_4_100_norm[gpt_4_100_norm['prompt'] == 'box'][features].mean().values,
            mistral_norm[mistral_norm['prompt'] == 'box'][features].mean().values]),
        ('Knife', [
            humans_norm[humans_norm['prompt'] == 'knife'][features].mean().values,
            gpt_35_100_norm[gpt_35_100_norm['prompt'] == 'knife'][features].mean().values,
            gpt_4_100_norm[gpt_4_100_norm['prompt'] == 'knife'][features].mean().values,
            mistral_norm[mistral_norm['prompt'] == 'knife'][features].mean().values]),
        ('Rope', [
            humans_norm[humans_norm['prompt'] == 'rope'][features].mean().values,
            gpt_35_100_norm[gpt_35_100_norm['prompt'] == 'rope'][features].mean().values,
            gpt_4_100_norm[gpt_4_100_norm['prompt'] == 'rope'][features].mean().values,
            mistral_norm[mistral_norm['prompt'] == 'rope'][features].mean().values])
    ]
    return data

def example_data_per_object_2(brick_norm, box_norm, knife_norm, rope_norm, features):
    data = [
        features,
        ('Brick', [
            brick_norm[brick_norm['dataset'] == 'Humans'][features].mean().values, # Humans
            brick_norm[brick_norm['dataset'] == 'GPT-3.5'][features].mean().values, # gpt-3.5
            brick_norm[brick_norm['dataset'] == 'GPT-4'][features].mean().values, # gpt-4
            brick_norm[brick_norm['dataset'] == 'Mistral'][features].mean().values, # mistral
            brick_norm[brick_norm['dataset'] == 'Vicuna'][features].mean().values]), # vicuna
        ('Box', [
            box_norm[box_norm['dataset'] == 'Humans'][features].mean().values,
            box_norm[box_norm['dataset'] == 'GPT-3.5'][features].mean().values,
            box_norm[box_norm['dataset'] == 'GPT-4'][features].mean().values,
            box_norm[box_norm['dataset'] == 'Mistral'][features].mean().values,
            box_norm[box_norm['dataset'] == 'Vicuna'][features].mean().values]),
        ('Knife', [
            knife_norm[knife_norm['dataset'] == 'Humans'][features].mean().values,
            knife_norm[knife_norm['dataset'] == 'GPT-3.5'][features].mean().values,
            knife_norm[knife_norm['dataset'] == 'GPT-4'][features].mean().values,
            knife_norm[knife_norm['dataset'] == 'Mistral'][features].mean().values,
            knife_norm[knife_norm['dataset'] == 'Vicuna'][features].mean().values]),
        ('Rope', [
            rope_norm[rope_norm['dataset'] == 'Humans'][features].mean().values,
            rope_norm[rope_norm['dataset'] == 'GPT-3.5'][features].mean().values,
            rope_norm[rope_norm['dataset'] == 'GPT-4'][features].mean().values,
            rope_norm[rope_norm['dataset'] == 'Mistral'][features].mean().values,
            rope_norm[rope_norm['dataset'] == 'Vicuna'][features].mean().values])
    ]
    return data

def radar_charts_per_object(brick_norm, box_norm, knife_norm, rope_norm, features):
#def radar_charts_per_object(humans_norm, gpt_35_100_norm, gpt_4_100_norm, mistral_norm, features):
    N = len(features)
    theta = radar_factory(N, frame='polygon')
    #data_visu = example_data_per_object(humans_norm, gpt_35_100_norm, gpt_4_100_norm, mistral_norm, features)
    data_visu = example_data_per_object_2(brick_norm, box_norm, knife_norm, rope_norm, features)
    spoke_labels = data_visu.pop(0)
    fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    colors = ['b', 'r', 'g', 'm', 'y']
    #colors = ['b', 'o', 'g', 'r', 'y']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data_visu):
        #ax.set_rgrids(np.arange(0, 1.2, 0.2).round(2), labels=np.arange(0, 1.2, 0.2).round(2))
        ax.set_varlabels(spoke_labels)
        #ax.set_rgrids([1, 2, 3, 4, 5])
        ax.set_title(title, weight='bold', size='xx-large', position=(0.5, 1.1),
                    horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        #ax.set_varlabels(spoke_labels)
        
        ## Line to set the grid
        ax.set_rticks(np.linspace(0, 0.6, 4))  # Set radial ticks from 0 to 1 with 5 intervals
    
    # add legend relative to top-left plot
    labels = ('Humans', 'GPT-3.5', 'GPT-4', 'Mistral', 'Vicuna')
    legend = axs[0, 0].legend(labels, loc=(0.9, .95),
                            labelspacing=0.1, fontsize='x-large')
    fig.text(0.5, 0.965, 'Creativity comparison per object',
            horizontalalignment='center', color='black', weight='bold',
            size='xx-large')
    plt.tight_layout()
    plt.show()

########################################
########### 1 chart in total ###########
########################################

def plot_radar_chart(dataframes, titles, avg_per_object):
    # Define categories (assuming all dataframes have the same columns)
    categories = list(dataframes[0].columns)[1:] # to not take into account the prompt column
    num_categories = len(categories)

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # To close the circle for radar chart

    # Create subplot with polar projection
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ['b', 'r', 'g', 'm', 'y']

    # Plot each dataframe as a polygon on the radar chart
    for i, df in enumerate(dataframes):
        
        if avg_per_object == True:
            # dimensions averaged by considering different objects
            values = []
            for object in ['brick', 'box', 'knife', 'rope']:
                df_object = df[df['prompt'] == object]
                # average per object
                values += df_object[categories].mean().values.flatten().tolist()
            # average across the 4 objects
            values = np.array(values).reshape(4, 4).mean(axis=0).tolist()
        else:
            # dimensions averaged without considering different objects
            values = df[categories].mean().values.flatten().tolist()
        
        values += values[:1]  # To close the circle for radar chart
        #ax.fill(angles, values, alpha=0.2, label=titles[i]) # Plot by filling polygons
        ax.plot(angles, values, label=titles[i], color = colors[i])  # Plot edges only

    # Set radial ticks and labels
    ax.set_yticks(np.arange(0, 0.7, 0.2).round(2))  # Set radial ticks
    ax.set_yticklabels(np.arange(0, 0.7, 0.2).round(2), fontsize='small', color='gray')  # Set radial labels

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    #ax.set_xticklabels(['         originality', 'elaboration        ', 'elaboration_SW            ', 'cosine_dist']) # to fit the labels outside the circle

    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title('LLMs vs Humans on the AUT')
    plt.show()
        

########################################
########### Normalization ##############
########################################

def normalization_1(df, check_norm):
    # normalize by features: min max scaling
    result = df.copy()
    for feature_name in df.columns:
        if feature_name in ["originality", "elaboration", "elaboration_SW", "cosine_dist"]:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    
    if check_norm:
        features = ['originality', 'elaboration', 'elaboration_SW', 'cosine_dist']

        # before normalization
        fig, axs = plt.subplots(1, 4, figsize = (15,4))
        for i, dim in enumerate(features):
            sns.boxplot(data=df, y=dim, x='dataset', ax=axs[i])

        plt.suptitle("Before normalization")
        plt.tight_layout()
        plt.show()

        # after normalization
        fig, axs = plt.subplots(1, 4, figsize = (15,4))
        for i, dim in enumerate(features):
            sns.boxplot(data=result, y=dim, x='dataset', ax=axs[i])

        plt.suptitle("After normalization")
        plt.tight_layout()
        plt.show()
    
    return result

def normalization_per_model(df):
    # normalize by features: min max scaling
    result = df.copy()
    for feature_name in df.columns:
        if feature_name in ["originality", "elaboration", "elaboration_SW", "cosine_dist"]:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    
    return result

