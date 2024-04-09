# import packages
import requests
import json
import pandas as pd
import os
import openai
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

# If OpenAI used
# Get API key from the environment variable
api_key = os.environ.get('OPENAI_API_KEY')

# Set the API key
openai.api_key = api_key

# Define function to create a prompt
def create_prompt(object_name, N_responses):
    #prompt = f"Generate alternative uses for the object [{object_name}]."
    prompt = f"""
        You are meant to assist students in group ideation. They are asked to propose alternative
        uses for an object and you should propose yours to give them ideas as well as inspire them to 
        explore other uses. 
        You are a very creative, open-minded person and can propose creative, out-of-the-box ideas while staying realistic. 
        Your ideas will be even more appreciated if they are original or useful in real-life or both.
        
        Generate alternative uses for the object [{object_name}]. 
        
        Provide exactly {N_responses} alternative uses, each explained in a concise sentence and following these examples: 
        Sock, Color it and maybe make a snake
        Sock, Use it as a puppet
        Sock, Use it as a dusting cloth
        
        Be careful to respect the exact same format as the examples above and the exact number of alternative uses requested.
        """
        
    return prompt

def create_prompt_exp2(object_name, N_responses, fs_examples):
    #prompt = f"Generate alternative uses for the object [{object_name}]."
    prompt = f"""
        You are meant to assist students in group ideation. They are asked to propose alternative
        uses for an object and you should propose yours to give them ideas as well as inspire them to 
        explore other uses. 
        You are a very creative, open-minded person and can propose creative, out-of-the-box ideas while staying realistic. 
        Your ideas will be even more appreciated if they are original or useful in real-life or both.
        
        Generate alternative uses for the object [{object_name}]. 
        
        Provide exactly {N_responses} alternative uses, each explained in a concise sentence and following these examples: 
{fs_examples}
        
        Be careful to respect the exact same format as the examples above and the exact number of alternative uses requested.
        """
        
    return prompt


# Define function to call the LLM API
def call_openai_api(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        #model="gpt-4-0125-preview", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ]#,
        #seed=seed,
        #max_tokens=200,
        #temperature=temperature,
        )
    return response


# Define function to store the result in a JSON file
def store_result_json(object_name, alternative_uses):
    result = [{"object_name": object_name, "alternative_uses": use} for use in  alternative_uses.split("\n") if use.strip()]
    with open(f"./data_ocsai/input_ocsai/aut_{object_name}.json", "w") as json_file:
        json.dump(result, json_file)


# Define function to create a CSV file and Pandas DataFrame
def create_csv_and_dataframe(object_name):
    with open(f"./data_ocsai/input_ocsai/aut_{object_name}.json", "r") as json_file:
        data = json.load(json_file)
    df = pd.DataFrame(data)
    
    # remove prefix
    df['alternative_uses']= df['alternative_uses'].apply(lambda x: x.split(", ", 1)[1])
    #df['alternative_uses']= df['alternative_uses'].apply(lambda x: x.split(":**", 1)[1])
    #df['alternative_uses'] = df['alternative_uses'].str.split(f': ').str[1]
    #df['alternative_uses'] = df['alternative_uses'].str.split(f'{object_name}: ').str[1]
    
    #rename the columns to fit ocsai format
    df = df.rename(columns={"object_name": "prompt", "alternative_uses": "response"})
    
    # save for evaluation
    #df.to_csv(f"./data_ocsai/input_ocsai/aut_{object_name}.csv", index=False)
    
    # Delete the JSON file
    os.remove(f"./data_ocsai/input_ocsai/aut_{object_name}.json")
    
    return df

# Define function to estimate the price of an OpenAI API request
def estimate_price(prompt_tokens, response_tokens):
    # Define your pricing model (e.g., price per token)
    #price_per_input_token = 0.50 / 1e6 # gpt3.5-turbo-0125
    #price_per_output_token = 1.5 / 1e6 # gpt3.5-turbo-0125
    price_per_input_token = 10 / 1e6 # gpt4-0125-preview
    price_per_output_token = 30 / 1e6 # gpt4-0125-preview
    
    # Estimate the price based on the total number of tokens
    price = (prompt_tokens * price_per_input_token) + (response_tokens * price_per_output_token)
    return price

# plot mean from outputs LLMs in the histogram of ground truth
def plot_mean_histogram(ocsai_eval, ground_truth, object_name, ax):
    plt.figure(figsize=(10, 6))
    
    object_ground_truth = ground_truth[ground_truth['prompt'] == object_name]
    
    sns.histplot(object_ground_truth['target'], kde=True, label='Humans', ax = ax)
    ax.vlines(ocsai_eval['originality'].mean(), 0, ax.get_ylim()[1], colors='red', linestyles='dashed', label='LLM')
    ax.set_title(f'Creativity on AUT for {object_name}')
    ax.legend()
    #plt.xlabel('Mean')
    #plt.ylabel('Frequency')
    
def pipeline_object(object_name, N_responses):
    prompt = create_prompt(object_name, N_responses)
    response = call_openai_api(prompt)
    
    response_content = response.choices[0].message.content
    system_fingerprint = response.system_fingerprint
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.total_tokens - response.usage.prompt_tokens

    print(f"System fingerprint: {system_fingerprint}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total number of tokens: {response.usage.total_tokens}")

    price = estimate_price(prompt_tokens, completion_tokens)
    print(f"Estimated price: ${price}")
    
    # store
    store_result_json(object_name, response_content)
    df = create_csv_and_dataframe(object_name)
    return df

def pipeline_object_exp2(object_name, N_responses, fs_examples, print_prompt):
    #prompt = create_prompt(object_name, N_responses)
    prompt = create_prompt_exp2(object_name, N_responses, fs_examples)
    if print_prompt:
        print(prompt)
    response = call_openai_api(prompt)
    
    response_content = response.choices[0].message.content
    system_fingerprint = response.system_fingerprint
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.total_tokens - response.usage.prompt_tokens

    print(f"System fingerprint: {system_fingerprint}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total number of tokens: {response.usage.total_tokens}")

    price = estimate_price(prompt_tokens, completion_tokens)
    print(f"Estimated price: ${price}")
    
    # store
    store_result_json(object_name, response_content)
    df = create_csv_and_dataframe(object_name)
    return df

def call_api_ocsai(prompt, output_llm):
    base_url = 'https://openscoring.du.edu/llm'
    model = 'ocsai-chatgpt'
    input_value = f"{prompt}, {output_llm}"
    input_type = 'csv'
    elab_method = 'whitespace'
    language = 'English'
    task = 'uses'
    question_in_input = 'false'
    
    # Encode special characters in the input value
    input_value_encoded = input_value.replace(' ', '%20').replace(',', '%2C')
    
    # Construct the URL
    #url = f"{base_url}?model={model}&prompt={prompt}&input={input_value_encoded}&input_type={input_type}&elab_method={elab_method}&language={language}&task={task}&question_in_input={question_in_input}"
    url = f"{base_url}?model={model}&input={input_value_encoded}&input_type={input_type}&elab_method={elab_method}&question_in_input={question_in_input}"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    
    return response.json()


def load_data_exp1(model):
    """
    Load the data for the first experiment
    """
    # load data with 100 responses
    aut_box_100 = pd.read_csv(f'./data_ocsai/data_exp1/{model}/aut_box_100.csv')
    aut_brick_100 = pd.read_csv(f'./data_ocsai/data_exp1/{model}/aut_brick_100.csv')
    aut_knife_100 = pd.read_csv(f'./data_ocsai/data_exp1/{model}/aut_knife_100.csv')
    aut_rope_100 = pd.read_csv(f'./data_ocsai/data_exp1/{model}/aut_rope_100.csv')

    # load data with 4x25 responses
    aut_box_4x25 = pd.read_csv(f'./data_ocsai/data_exp1/{model}/aut_box_4x25.csv')
    aut_brick_4x25 = pd.read_csv(f'./data_ocsai/data_exp1/{model}/aut_brick_4x25.csv')
    aut_knife_4x25 = pd.read_csv(f'./data_ocsai/data_exp1/{model}/aut_knife_4x25.csv')
    aut_rope_4x25 = pd.read_csv(f'./data_ocsai/data_exp1/{model}/aut_rope_4x25.csv')

    # merge the dataframes
    ocsai_eval_merged_gpt35_100 = pd.concat([aut_box_100, aut_brick_100, aut_knife_100, aut_rope_100])
    ocsai_eval_merged_gpt35_4x25 = pd.concat([aut_box_4x25, aut_brick_4x25, aut_knife_4x25, aut_rope_4x25])
    #rename the columns originality
    ocsai_eval_merged_100 = ocsai_eval_merged_gpt35_100.rename(columns={"originality_ocsai": "originality", "elaboration_ocsai": "elaboration"})
    ocsai_eval_merged_4x25 = ocsai_eval_merged_gpt35_4x25.rename(columns={"originality_ocsai": "originality", "elaboration_ocsai": "elaboration"})
    
    return ocsai_eval_merged_100, ocsai_eval_merged_4x25