# to call LMM API

import requests
import json
import pandas as pd
import os
import openai
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# plot statistical difference
def plot_kde(ocsai_eval_merged_gpt35, ocsai_eval_merged_gpt4, ground_truth, objects):
    fig, axs = plt.subplots(1,4, figsize=(15, 4), sharey=True)
    for i, object_name in enumerate(objects):
        # normalize the values in ground truth and ocsai_eval_merged
        object_ground_truth = ground_truth[ground_truth['prompt'] == object_name]
        object_ocsai_eval_gpt35 = ocsai_eval_merged_gpt35[ocsai_eval_merged_gpt35['prompt'] == object_name]
        object_ocsai_eval_gpt4 = ocsai_eval_merged_gpt4[ocsai_eval_merged_gpt4['prompt'] == object_name]
        
        sns.kdeplot(object_ground_truth['target'], label=f'Humans, N = {len(object_ground_truth)}', ax = axs[i])
        sns.kdeplot(object_ocsai_eval_gpt35['originality'], label=f'GPT3.5, N = {len(object_ocsai_eval_gpt35)}', ax = axs[i])
        sns.kdeplot(object_ocsai_eval_gpt4['originality'], label=f'GPT4, N = {len(object_ocsai_eval_gpt4)}', ax = axs[i])
        axs[i].set_title(f'{object_name}')
        axs[i].legend()

    plt.suptitle('Statistical difference of Creativity on AUT between LLM and humans')
    plt.tight_layout()
    plt.show()
    
# plot violin plots
def plot_violin(ocsai_eval_merged_gpt35, ocsai_eval_merged_gpt4, ground_truth, objects):
    fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    # Originality of humans
    sns.violinplot(x='prompt', y='target', data=ground_truth[ground_truth['prompt'].isin(objects)], ax = axs[0], order = objects)
    axs[0].set_title('Human originality')

    # Originality of LLM GPT3.5
    sns.violinplot(x='prompt', y='originality', data=ocsai_eval_merged_gpt35[ocsai_eval_merged_gpt35['prompt'].isin(objects)], ax = axs[1], order=objects)
    axs[1].set_title('LLM originality (GPT3.5)')
    
    # Originality of LLM GPT4
    sns.violinplot(x='prompt', y='originality', data=ocsai_eval_merged_gpt4[ocsai_eval_merged_gpt4['prompt'].isin(objects)], ax = axs[2], order=objects)
    axs[2].set_title('LLM originality (GPT4)')

    plt.tight_layout()
    plt.show()
    
#plot_kde
def plot_kde_exp1(model, ocsai_eval_merged_gpt35_30, ocsai_eval_merged_gpt35_100, ocsai_eval_merged_gpt35_4x25, ocsai_eval_merged_gpt4_30, ocsai_eval_merged_gpt4_100, ocsai_eval_merged_gpt4_4x25):
    
    ground_truth = pd.read_csv('./../data/cleaned_all_data.csv')
    objects = list(ground_truth['prompt'].value_counts().head(4).index)
    
    if model == 'gpt_35':
        fig, axs = plt.subplots(1,4, figsize=(15, 4), sharey=True)
        for i, object_name in enumerate(objects):
            # normalize the values in ground truth and ocsai_eval_merged
            object_ground_truth = ground_truth[ground_truth['prompt'] == object_name]
            object_ocsai_eval_gpt35_30 = ocsai_eval_merged_gpt35_30[ocsai_eval_merged_gpt35_30['prompt'] == object_name]
            object_ocsai_eval_gpt35_100 = ocsai_eval_merged_gpt35_100[ocsai_eval_merged_gpt35_100['prompt'] == object_name]
            object_ocsai_eval_gpt35_4x25 = ocsai_eval_merged_gpt35_4x25[ocsai_eval_merged_gpt35_4x25['prompt'] == object_name]
            
            sns.kdeplot(object_ground_truth['target'], label=f'Humans, N = {len(object_ground_truth)}', ax = axs[i])
            sns.kdeplot(object_ocsai_eval_gpt35_30['originality'], label=f'GPT3.5, 30 ex', ax = axs[i])
            sns.kdeplot(object_ocsai_eval_gpt35_100['originality'], label=f'GPT3.5, 100 ex', ax = axs[i])
            sns.kdeplot(object_ocsai_eval_gpt35_4x25['originality'], label=f'GPT3.5, 4x25 ex', ax = axs[i])
            
            axs[i].set_title(f'{object_name}')
            axs[i].legend()
        plt.suptitle(f'Statistical difference of Creativity on AUT between {model} and humans')
        plt.tight_layout()
        plt.show()
    elif model == 'gpt_4':
        fig, axs = plt.subplots(1,4, figsize=(15, 4), sharey=True)
        for i, object_name in enumerate(objects):
            # normalize the values in ground truth and ocsai_eval_merged
            object_ground_truth = ground_truth[ground_truth['prompt'] == object_name]
            object_ocsai_eval_gpt4_30 = ocsai_eval_merged_gpt4_30[ocsai_eval_merged_gpt4_30['prompt'] == object_name]
            object_ocsai_eval_gpt4_100 = ocsai_eval_merged_gpt4_100[ocsai_eval_merged_gpt4_100['prompt'] == object_name]
            object_ocsai_eval_gpt4_4x25 = ocsai_eval_merged_gpt4_4x25[ocsai_eval_merged_gpt4_4x25['prompt'] == object_name]
            
            sns.kdeplot(object_ground_truth['target'], label=f'Humans, N = {len(object_ground_truth)}', ax = axs[i])
            sns.kdeplot(object_ocsai_eval_gpt4_30['originality'], label=f'GPT4, 30 ex', ax = axs[i])
            sns.kdeplot(object_ocsai_eval_gpt4_100['originality'], label=f'GPT4, 100 ex', ax = axs[i])
            sns.kdeplot(object_ocsai_eval_gpt4_4x25['originality'], label=f'GPT4, 4x25 ex', ax = axs[i])
            
            axs[i].set_title(f'{object_name}')
            axs[i].legend()
        plt.suptitle(f'Statistical difference of Creativity on AUT between {model} and humans')
        plt.tight_layout()
        plt.show()