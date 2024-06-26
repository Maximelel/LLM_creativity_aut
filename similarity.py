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
#def encode_text(texts, model_name):
#    '''
#    Transform each sentence into embeddings
#    inputs: 
#        - texts: list of sentences
#        - model_name: model to use for embeddings
#    output:
#        - embeddings: torch tensor of embeddings of size [# sentences, embedding dim]
#    '''
#    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    model = AutoModel.from_pretrained(model_name)
#
#    # Tokenize input texts and generate embeddings
#    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
#    with torch.no_grad():
#        outputs = model(**encoded_input)
#    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling to get sentence embeddings
#    return embeddings
#
## Function to compute cosine distance matrix
#def compute_cosine_distance_matrix(embeddings):
#    cosine_dist_matrix = cosine_distances(embeddings.numpy())
#    return cosine_dist_matrix
#
## Function to compute average cosine similarity for each sentence
#def compute_avg_cosine_similarity(cosine_dist_matrix):
#    num_sentences = cosine_dist_matrix.shape[0]
#    avg_cosine_similarities = np.zeros(num_sentences)
#
#    for i in tqdm(range(num_sentences)):
#        avg_cosine_similarities[i] = np.mean(np.delete(cosine_dist_matrix[i], i))  # Exclude self-similarity
#
#    return avg_cosine_similarities
#
#def compute_sentences_similarity(df, model_name):
#    # Encode the textual answers into embeddings
#    embeddings = encode_text(df['response'].tolist(), model_name)
#
#    # Compute cosine distance matrix
#    cosine_dist_matrix = compute_cosine_distance_matrix(embeddings)
#
#    # Compute average cosine similarity for each sentence
#    avg_cosine_similarities = compute_avg_cosine_similarity(cosine_dist_matrix)
#    
#    return avg_cosine_similarities
#
#def compute_sentences_sim_per_object(df, model_name):
#    for object in df['prompt'].unique():
#        avg_cosine_similarities = compute_sentences_similarity(df[df['prompt'] == object], model_name)
#        df.loc[df['prompt'] == object, 'dissimilarity_old'] = avg_cosine_similarities
#    
#    return df

#def compute_avg_similarity_each_sentence(model, tokenizer, sentence_ref, other_sentences):
#
#    #inputs_ref = tokenizer(sentence_ref, return_tensors="pt", padding=True, truncation=True)
#    #outputs_ref = model(**inputs_ref)
#
#    # Obtain sentence embeddings = each sentence is represented by a 768 elements vector
#    # mean: average of the hidden states of the tokens
#    # detach: returns tensor
#    # numpy: transform to numpy array
#    #embeddings_ref = outputs_ref.last_hidden_state.mean(dim=1).detach().numpy()
#    
#    similarity_sentence = []
#    for i in range(len(other_sentences)):
#        text2 = other_sentences[i]
#        inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
#        outputs2 = model(**inputs2)
#        embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()
#
#        #Calculate cosine similarity (similar sentence => cosine similarity = 1)
#        similarity_sentence.append(np.dot(embeddings_ref, embeddings2.T) / (np.linalg.norm(embeddings_ref) * np.linalg.norm(embeddings2)))
#
#    return np.mean(similarity_sentence)
#

def compute_cosine_similarity_from_sentence_embedding(embedding_ref, embedding_other):
    return np.dot(embedding_ref, embedding_other) / (np.linalg.norm(embedding_ref) * np.linalg.norm(embedding_other))
    
    
def compute_dissimilarity(df, embeddings_model_name):

    tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name)
    model = AutoModel.from_pretrained(embeddings_model_name)    
    
    # List to store all sentence embeddings
    sentence_embeddings = []
    
    # compute sentence embedding for all ideas
    print("---- Compute sentence embeddings...")
    for i in tqdm(range(len(df))):
        sentence = df.loc[i, 'response']
        #print(sentence)
        input_ids = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        output = model(**input_ids)
        
        # Obtain sentence embeddings = each sentence is represented by a 768 elements vector
        # mean: average of the hidden states of the tokens
        # detach: returns tensor
        # numpy: transform to numpy array
        #embeddings_ref = outputs_ref.last_hidden_state.mean(dim=1).detach().numpy()
        sentence_embedding = output.last_hidden_state.mean(dim=1).detach().numpy().flatten()
        
        # Append the embedding to the list
        sentence_embeddings.append(sentence_embedding)
    
    # Assign the list of embeddings to a new column in the DataFrame
    df['sentence_embedding'] = sentence_embeddings
    
    print("---- Compute similarity between sentences...")
    df_output = pd.DataFrame()
    for object in df['prompt'].unique():
        df_object = df[df['prompt'] == object].reset_index(drop = True)
        for i in tqdm(range(len(df_object))):
            df_tmp = df_object.copy()
            df_tmp = df_tmp.drop(i).reset_index(drop = True)
            sentence_similarity = []
            for j in range(len(df_tmp)):
                sentence_similarity.append(compute_cosine_similarity_from_sentence_embedding(df_object.loc[i, 'sentence_embedding'], df_tmp.loc[j, 'sentence_embedding']))
            # Take 1 - similarity to get dissimilarity
            df_object.loc[i, 'dissimilarity'] = 1 - np.mean(sentence_similarity)
        df_output = pd.concat([df_output, df_object])
    
    # drop column "sentence_embedding"
    df_output.drop('sentence_embedding', axis = 1, inplace = True)
    # reset index
    df_output.reset_index(drop = True, inplace = True)
    return df_output
