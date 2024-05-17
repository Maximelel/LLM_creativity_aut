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
    fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=3,
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
def prepare_data_for_radar_chart_per_object(data_dict, objects):
    result_dict = {}

    for object in objects:
        data_frames = []
        for model_name, df in data_dict.items():
            data_frames.append(df[df['prompt'] == object].assign(dataset=model_name))
        
        concatenated_data = pd.concat(data_frames)
        normalized_data = normalization_1(concatenated_data, check_norm=False) # normalize data per object and per feature
        result_dict[f'{object}_norm'] = normalized_data

    return result_dict

def example_data_per_object(brick_norm, box_norm, knife_norm, rope_norm, features, model_names):
    if len(model_names) != 5:
        print("Error: model_names should contain 5 model names...")
        return 0
    
    data = [
        features,
        ('Brick', [
            brick_norm[brick_norm['dataset'] == model_names[0]][features].mean().values, # Humans
            brick_norm[brick_norm['dataset'] == model_names[1]][features].mean().values, # model 1
            brick_norm[brick_norm['dataset'] == model_names[2]][features].mean().values, # model 2
            brick_norm[brick_norm['dataset'] == model_names[3]][features].mean().values, # model 3
            brick_norm[brick_norm['dataset'] == model_names[4]][features].mean().values]), # model 4
        ('Box', [
            box_norm[box_norm['dataset'] == model_names[0]][features].mean().values,
            box_norm[box_norm['dataset'] == model_names[1]][features].mean().values,
            box_norm[box_norm['dataset'] == model_names[2]][features].mean().values,
            box_norm[box_norm['dataset'] == model_names[3]][features].mean().values,
            box_norm[box_norm['dataset'] == model_names[4]][features].mean().values]),
        ('Knife', [
            knife_norm[knife_norm['dataset'] == model_names[0]][features].mean().values,
            knife_norm[knife_norm['dataset'] == model_names[1]][features].mean().values,
            knife_norm[knife_norm['dataset'] == model_names[2]][features].mean().values,
            knife_norm[knife_norm['dataset'] == model_names[3]][features].mean().values,
            knife_norm[knife_norm['dataset'] == model_names[4]][features].mean().values]),
        ('Rope', [
            rope_norm[rope_norm['dataset'] == model_names[0]][features].mean().values,
            rope_norm[rope_norm['dataset'] == model_names[1]][features].mean().values,
            rope_norm[rope_norm['dataset'] == model_names[2]][features].mean().values,
            rope_norm[rope_norm['dataset'] == model_names[3]][features].mean().values,
            rope_norm[rope_norm['dataset'] == model_names[4]][features].mean().values])
    ]
    return data

def radar_charts_per_object(brick_norm, box_norm, knife_norm, rope_norm, features, model_names, colors):
    N = len(features)
    theta = radar_factory(N, frame='polygon')
    data_visu = example_data_per_object(brick_norm, box_norm, knife_norm, rope_norm, features, model_names)
    spoke_labels = data_visu.pop(0)
    fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    #colors = ['b', 'r', 'g', 'm', 'y']
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
    #labels = ('Humans', 'GPT-3.5', 'GPT-4', 'Mistral', 'Vicuna')
    labels = tuple(model_names)
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

def plot_radar_chart(dataframes, titles, colors, avg_per_object):
    # Define categories (assuming all dataframes have the same columns)
    categories = list(dataframes[0].columns)[1:] # to not take into account the prompt column
    num_categories = len(categories)

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # To close the circle for radar chart

    # Create subplot with polar projection
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    #colors = ['b', 'r', 'g', 'm', 'y']

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
        
def plot_per_object(type, data_dict, features, combined_data_norm_per_object):
    if type == "kde":
        for feat in features:
            fig, axs = plt.subplots(1, 4, figsize=(15, 5))
            for name, df in data_dict.items():
                for i, object in enumerate(['brick', 'box', 'knife','rope']):
                    sns.kdeplot(df[feat], label=name, ax = axs[i])
                    axs[i].set_title(object)
                    axs[i].legend()
        plt.suptitle(f"Comparison on {feat}")
        plt.tight_layout()
        plt.show()
    elif type == "boxplot":
        for feat in features:
            plt.figure(figsize=(12, 5))
            sns.boxplot(data=combined_data_norm_per_object, x='dataset', y=feat, hue = 'prompt')
            plt.suptitle(f"Comparison on {feat}")
            plt.tight_layout()
            plt.show()
    elif type == "violinplot":
        for feat in features:
            plt.figure(figsize=(12, 5))
            sns.violinplot(data=combined_data_norm_per_object, x='dataset', y=feat, hue = 'prompt')
            plt.suptitle(f"Comparison on {feat}")
            plt.tight_layout()
            plt.show()
    else:
        print("Wrong type of plot")
        return 0


########################################
########### Normalization ##############
########################################

def normalization_1(df, check_norm):
    # normalize by features: min max scaling
    result = df.copy()
    for feature_name in df.columns:
        if feature_name in ["originality", "elaboration", "elaboration_SW", "dissimilarity", "flexibility"]:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    
    if check_norm:
        features = ['originality', 'elaboration', 'elaboration_SW', 'dissimilarity', 'flexibility']

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
        if feature_name in ["originality", "elaboration", "elaboration_SW", "dissimilarity", "flexibility"]:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    
    return result

