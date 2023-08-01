import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np
import ast
import re

import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

import spacy
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.preprocessing.text import Tokenizer # clean and tokenize data
from keras.preprocessing.sequence import pad_sequences #
from keras.initializers import he_normal
import tensorflow as tf
import keras.backend as K # facilitate the computation of performance metrics

import string
import nltk

from nltk.corpus import stopwords
nltk.download('stopwords')

from pathlib import Path
import joblib


# display purposes

# Parameter mappings to override the values in the preset seaborn style dictionaries
color = 'white'
sns.set_theme(rc={'grid.color': 'white', 
                  'xtick.color': color,
                  'ytick.color': color,
                  'axes.labelcolor': color,
                  'text.color': color,
                  'axes.facecolor':(0,0,0,0),
                  'figure.facecolor':(0,0,0,0)})


# definitions / lists

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['welcome','luck'])

remove_entities = ['one','two','first','1','2','3','4','5','6','7','10','30',
                   'ref','kak','juga','ga','bro','kami','sa']

def top_n_grams(corpus, n, ngram, stop):
    # Create a CountVectorizer object with specified n-gram range and stop words
    vec = CountVectorizer(ngram_range=ngram, stop_words=stop)
    # Convert the corpus into a bag of words representation
    bag_of_words = vec.fit_transform(corpus)
    # Calculate the sum of words across all documents
    sum_words = bag_of_words.sum(axis=0) 
    # Create a list of (word, count) pairs from the vocabulary and word counts
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    # Sort the list of (word, count) pairs by count in descending order and select top n
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)[:n]
    # Store top n common n-grams in a dataframe
    df = pd.DataFrame(words_freq, columns=['text', 'count'])
    # Sort the dataframe by the count column in descending order
    df = df.sort_values(by='count', ascending=False)

    return df

# define custom functions to calculate R2
def R2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) # sum of squares of residuals
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) # total sum of squares
    # K.epsilon: Epsilon is small value that makes very little difference to the value of 
    # the denominator, but ensures that it isn't equal to exactly zero.
    r2_score = (1 - (SS_res / (SS_tot + K.epsilon())))
    return r2_score

# empty list function
def create_empty_list():
    return []

# define function that creates new dataframe with NPS of each entity
def create_nps_dataframe(sentence_df,sg_entities_patterns_df):
    
    # convert string of list back to a list
    sentence_df['entity'] = sentence_df['entity'].map(lambda x: ast.literal_eval(x))
    sentence_df['pos_sentiment_words'] = sentence_df['pos_sentiment_words'].map(lambda x: ast.literal_eval(x))
    sentence_df['neg_sentiment_words'] = sentence_df['neg_sentiment_words'].map(lambda x: ast.literal_eval(x))

    list_of_lists_all = [entities for entities in sentence_df['entity']]

    unique_entities_count = {}

    # Loop through each sublist in the list of lists
    for sublist in list_of_lists_all:
        # Loop through each string in the sublist
        for word in sublist:
            # need to swap out the word with the replacement from my other list
            # we will only keep entities that are in 'sg_entities_patterns_df'
            if word in sg_entities_patterns_df['pattern'].values:
                # find the row number in which this word is found
                row_num = (sg_entities_patterns_df.index[sg_entities_patterns_df['pattern'] == word])[0]
                # replace word with value in 'sublocation'
                word = sg_entities_patterns_df['sublocation'][row_num]
                if word not in remove_entities:
                    # Update the count in the dictionary                 
                    unique_entities_count[word] = unique_entities_count.get(word, 0) + 1

    # save entity and its corresponding count in a dataframe
    unique_entities_count_df = {'entity': [],'count': []}
    for k,v in unique_entities_count.items():
        unique_entities_count_df['entity'].append(k)
        unique_entities_count_df['count'].append(v)
    unique_entities_count_df = pd.DataFrame(unique_entities_count_df)

    # add new columns
    unique_entities_count_df['promoter_count'] = 0
    unique_entities_count_df['detractor_count'] = 0
    unique_entities_count_df['total_count'] = 0
    unique_entities_count_df['pos_sentiment_words'] = 0
    unique_entities_count_df['neg_sentiment_words'] = 0
    unique_entities_count_df['interest_1'] = ''
    unique_entities_count_df['interest_2'] = ''
    unique_entities_count_df['indoor_outdoor'] = ''

    # Apply the function to each cell
    unique_entities_count_df['pos_sentiment_words'] = unique_entities_count_df['pos_sentiment_words'].apply(lambda x: create_empty_list())
    unique_entities_count_df['neg_sentiment_words'] = unique_entities_count_df['neg_sentiment_words'].apply(lambda x: create_empty_list())

    # calculate nps for each entity
    for index, entities in enumerate(sentence_df['entity']):
        # extract the positive and negative sentiment words as a list
        pos_list = sentence_df['pos_sentiment_words'].loc[index]
        neg_list = sentence_df['neg_sentiment_words'].loc[index]
        for entity in entities:
            # we only want the locations mentioned in sg_entities_patterns_df
            if entity in sg_entities_patterns_df['pattern'].values:
                # find the row number in which this word is found
                row_num = (sg_entities_patterns_df.index[sg_entities_patterns_df['pattern'] == entity])[0]
                # replace word with value in 'sublocation'
                entity = sg_entities_patterns_df['sublocation'][row_num]        
                # find the row in which this entity is saved in for unique_entities_count_df
                row_num = (unique_entities_count_df.index[unique_entities_count_df['entity'] == entity])
                unique_entities_count_df['total_count'].loc[row_num] = unique_entities_count_df['total_count'].loc[row_num] + 1

                # save the extracted pos and neg words in a list
                unique_entities_count_df['pos_sentiment_words'].loc[row_num[0]] = pos_list + unique_entities_count_df['pos_sentiment_words'].loc[row_num[0]]
                unique_entities_count_df['neg_sentiment_words'].loc[row_num[0]] = neg_list + unique_entities_count_df['neg_sentiment_words'].loc[row_num[0]]

                # add to the respective count if the sentence is a promoter / detractor
                if sentence_df['sentiment'].loc[index] >= 9:
                    unique_entities_count_df['promoter_count'].loc[row_num] = unique_entities_count_df['promoter_count'].loc[row_num] + 1
                elif sentence_df['sentiment'].loc[index] < 6:
                    unique_entities_count_df['detractor_count'].loc[row_num] = unique_entities_count_df['detractor_count'].loc[row_num] + 1

    # nps formula
    unique_entities_count_df['nps'] = ((unique_entities_count_df['promoter_count'] - unique_entities_count_df['detractor_count']) / unique_entities_count_df['count']) * 100

    # add the 'interest_1', 'interest_2' and 'indoor_outdoor' info to the dataframe
    for index, entity in enumerate(unique_entities_count_df['entity']):
        # find the matching entity in 'sg_entities_patterns_df' and get the row_num
        row_num = sg_entities_patterns_df.index[sg_entities_patterns_df['sublocation'] == entity][0]
        unique_entities_count_df['interest_1'].loc[index] = sg_entities_patterns_df['interest_1'].loc[row_num]
        unique_entities_count_df['interest_2'].loc[index] = sg_entities_patterns_df['interest_2'].loc[row_num]
        unique_entities_count_df['indoor_outdoor'].loc[index] = sg_entities_patterns_df['indoor_outdoor'].loc[row_num]

    # for positive words
    for index, list_of_strings in enumerate(unique_entities_count_df['pos_sentiment_words']):
        dictionary = {}
        final_dict = {'sentiment_word': [],'count': []}
        for sentiment_word in list_of_strings:
            dictionary[sentiment_word] = dictionary.get(sentiment_word,0) + 1
        for k, v in dictionary.items():
            final_dict['sentiment_word'].append(k)
            final_dict['count'].append(v)
        # save in the cell
        unique_entities_count_df['pos_sentiment_words'][index] = final_dict

    # for negative words
    for index, list_of_strings in enumerate(unique_entities_count_df['neg_sentiment_words']):
        dictionary = {}
        final_dict = {'sentiment_word': [],'count': []}
        for sentiment_word in list_of_strings:
            dictionary[sentiment_word] = dictionary.get(sentiment_word,0) + 1
        for k, v in dictionary.items():
            final_dict['sentiment_word'].append(k)
            final_dict['count'].append(v)
        # save in the cell
        unique_entities_count_df['neg_sentiment_words'][index] = final_dict
    return unique_entities_count_df

# define function that creates new dataframe with NPS of each entity
def create_nps_dataframe_without_ast_conversion(sentence_df,sg_entities_patterns_df):

    list_of_lists_all = [entities for entities in sentence_df['entity']]

    unique_entities_count = {}

    # Loop through each sublist in the list of lists
    for sublist in list_of_lists_all:
        # Loop through each string in the sublist
        for word in sublist:
            # need to swap out the word with the replacement from my other list
            # we will only keep entities that are in 'sg_entities_patterns_df'
            if word in sg_entities_patterns_df['pattern'].values:
                # find the row number in which this word is found
                row_num = (sg_entities_patterns_df.index[sg_entities_patterns_df['pattern'] == word])[0]
                # replace word with value in 'sublocation'
                word = sg_entities_patterns_df['sublocation'][row_num]
                if word not in remove_entities:
                    # Update the count in the dictionary                 
                    unique_entities_count[word] = unique_entities_count.get(word, 0) + 1

    # save entity and its corresponding count in a dataframe
    unique_entities_count_df = {'entity': [],'count': []}
    for k,v in unique_entities_count.items():
        unique_entities_count_df['entity'].append(k)
        unique_entities_count_df['count'].append(v)
    unique_entities_count_df = pd.DataFrame(unique_entities_count_df)

    # add new columns
    unique_entities_count_df['promoter_count'] = 0
    unique_entities_count_df['detractor_count'] = 0
    unique_entities_count_df['total_count'] = 0
    unique_entities_count_df['pos_sentiment_words'] = 0
    unique_entities_count_df['neg_sentiment_words'] = 0
    unique_entities_count_df['interest_1'] = ''
    unique_entities_count_df['interest_2'] = ''
    unique_entities_count_df['indoor_outdoor'] = ''

    # Apply the function to each cell
    unique_entities_count_df['pos_sentiment_words'] = unique_entities_count_df['pos_sentiment_words'].apply(lambda x: create_empty_list())
    unique_entities_count_df['neg_sentiment_words'] = unique_entities_count_df['neg_sentiment_words'].apply(lambda x: create_empty_list())

    # calculate nps for each entity
    for index, entities in enumerate(sentence_df['entity']):
        # extract the positive and negative sentiment words as a list
        pos_list = sentence_df['pos_sentiment_words'].loc[index]
        neg_list = sentence_df['neg_sentiment_words'].loc[index]
        for entity in entities:
            # we only want the locations mentioned in sg_entities_patterns_df
            if entity in sg_entities_patterns_df['pattern'].values:
                # find the row number in which this word is found
                row_num = (sg_entities_patterns_df.index[sg_entities_patterns_df['pattern'] == entity])[0]
                # replace word with value in 'sublocation'
                entity = sg_entities_patterns_df['sublocation'][row_num]        
                # find the row in which this entity is saved in for unique_entities_count_df
                row_num = (unique_entities_count_df.index[unique_entities_count_df['entity'] == entity])
                unique_entities_count_df['total_count'].loc[row_num] = unique_entities_count_df['total_count'].loc[row_num] + 1

                # save the extracted pos and neg words in a list
                unique_entities_count_df['pos_sentiment_words'].loc[row_num[0]] = pos_list + unique_entities_count_df['pos_sentiment_words'].loc[row_num[0]]
                unique_entities_count_df['neg_sentiment_words'].loc[row_num[0]] = neg_list + unique_entities_count_df['neg_sentiment_words'].loc[row_num[0]]

                # add to the respective count if the sentence is a promoter / detractor
                if sentence_df['sentiment'].loc[index] >= 9:
                    unique_entities_count_df['promoter_count'].loc[row_num] = unique_entities_count_df['promoter_count'].loc[row_num] + 1
                elif sentence_df['sentiment'].loc[index] < 6:
                    unique_entities_count_df['detractor_count'].loc[row_num] = unique_entities_count_df['detractor_count'].loc[row_num] + 1

    # nps formula
    unique_entities_count_df['nps'] = ((unique_entities_count_df['promoter_count'] - unique_entities_count_df['detractor_count']) / unique_entities_count_df['count']) * 100

    # add the 'interest_1', 'interest_2' and 'indoor_outdoor' info to the dataframe
    for index, entity in enumerate(unique_entities_count_df['entity']):
        # find the matching entity in 'sg_entities_patterns_df' and get the row_num
        row_num = sg_entities_patterns_df.index[sg_entities_patterns_df['sublocation'] == entity][0]
        unique_entities_count_df['interest_1'].loc[index] = sg_entities_patterns_df['interest_1'].loc[row_num]
        unique_entities_count_df['interest_2'].loc[index] = sg_entities_patterns_df['interest_2'].loc[row_num]
        unique_entities_count_df['indoor_outdoor'].loc[index] = sg_entities_patterns_df['indoor_outdoor'].loc[row_num]

    # for positive words
    for index, list_of_strings in enumerate(unique_entities_count_df['pos_sentiment_words']):
        dictionary = {}
        final_dict = {'sentiment_word': [],'count': []}
        for sentiment_word in list_of_strings:
            dictionary[sentiment_word] = dictionary.get(sentiment_word,0) + 1
        for k, v in dictionary.items():
            final_dict['sentiment_word'].append(k)
            final_dict['count'].append(v)
        # save in the cell
        unique_entities_count_df['pos_sentiment_words'][index] = final_dict

    # for negative words
    for index, list_of_strings in enumerate(unique_entities_count_df['neg_sentiment_words']):
        dictionary = {}
        final_dict = {'sentiment_word': [],'count': []}
        for sentiment_word in list_of_strings:
            dictionary[sentiment_word] = dictionary.get(sentiment_word,0) + 1
        for k, v in dictionary.items():
            final_dict['sentiment_word'].append(k)
            final_dict['count'].append(v)
        # save in the cell
        unique_entities_count_df['neg_sentiment_words'][index] = final_dict
    return unique_entities_count_df

# define function that will combine the sentiment words dictionary
def add_counts(sentiment_words, counts, combined_dict):
    for word, count in zip(sentiment_words, counts):
        if word in combined_dict['sentiment_word']:
            # find the index of the word in the list saved in combined_dict['sentiment_words']
            index_of_word = combined_dict['sentiment_word'].index(word)
            combined_dict['count'][index_of_word] += count
        else:
            combined_dict['sentiment_word'].append(word)
            combined_dict['count'].append(count)
    return combined_dict

# define function that will create a new dataframe that contains sentiment words and count
def create_sentiment_words_df(entities_df,pos_or_neg_sentiment_words):
    # reset sorting
    entities_df = entities_df.sort_index()
    
    # new dictionary
    sentiment_words_dict = {'sentiment_word':[],'count':[]}
    
    # for positive words
    for index, interest_1 in enumerate(entities_df[pos_or_neg_sentiment_words]):
        # Add pos_sentiment_words_dict to the combined_dict
        combined_dict = sentiment_words_dict

        # Add counts from unique_entities_count_df['pos_sentiment_words'][index] to the combined_dict
        add_counts(entities_df[pos_or_neg_sentiment_words][index]['sentiment_word'], entities_df[pos_or_neg_sentiment_words][index]['count'],combined_dict)

        # save the combined_dict in pos_sentiment_words_dict
        sentiment_words_dict = combined_dict
        
        # save as dataframe
        sentiment_words_df = pd.DataFrame(sentiment_words_dict)
        sentiment_words_df = sentiment_words_df.sort_values(by=['count'],ascending=False)
    return sentiment_words_df

# define function for lemmatization using spaCy
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join(token.lemma_ for token in doc)
    return lemmatized_text

# define function for splitting, lowercase, lemma sentences and extract entity, pos and neg sentiment words
def clean_extract_dataframe(data_n):
    # instantiate dataframe
    data_n_mod_df = pd.DataFrame(columns=['sentence','entity','pos_sentiment_words','neg_sentiment_words'])
    
    # find no. of rows in 'video_info_audio_caption_cleaned_df'
    n_rows = len(data_n)

    for i in range(0,n_rows):
        # define text
        text = data_n['sentence'].loc[i]

        # if there are no words in 'final_text_description', it becomes a math nan which is a float
        # thus, to check if there are words, check if type is string or float
        if type(text) == str:
            # split text into smaller sentences
            sentences = re.split(r'[^\w\s\'-,]', text) # NOT a word character, whitespace, single quote, hyphen, or comma

            for sentence in sentences:
                # instantiate a dictionary
                dictionary = {}

                # convert characters to lowercase
                sentence = sentence.lower()
                # remove punctuations
                sentence = sentence.translate(str.maketrans('','',string.punctuation))

                # check if there are words in the sentence
                if sentence != '' and sentence != ' ':
                    # save the sentence in dictionary
                    dictionary['sentence'] = sentence

                    # instantiate 'pos_sentiment_words' and 'neg_sentiment_words'
                    dictionary['pos_sentiment_words'] = []
                    dictionary['neg_sentiment_words'] = []
                    for token in sentence.split():
                        textblob_token = TextBlob(token)
                        if textblob_token.sentiment.polarity > 0:
                            dictionary['pos_sentiment_words'].append(token)
                        elif textblob_token.sentiment.polarity < 0:
                            dictionary['neg_sentiment_words'].append(token)

                    # lemmatize text using function defined above
                    lemmatized_text = lemmatize_text(sentence)
                    lemmatized_doc = nlp(lemmatized_text)

                    # extract entities 
                    entities = [entity for entity in lemmatized_doc.ents]

                    for entity in entities:

                        # if word happens to be in the name of entity, it should not be a sentiment word, remove from pos words
                        for word in dictionary['pos_sentiment_words']:
                            if word in str(entity):
                                dictionary['pos_sentiment_words'].remove(word)
                                sentence = sentence.replace(word,'')
                        # if word happens to be in the name of entity, it should not be a sentiment word, remove from neg words
                        for word in dictionary['neg_sentiment_words']:
                            if word in str(entity):
                                dictionary['neg_sentiment_words'].remove(word)
                                sentence = sentence.replace(word,'')

                    dictionary['entity'] = []
                    for entity in entities:
                        # save the entity in the dictionary
                        dictionary['entity'].append(str(entity))

                    # save the entire dictionary to train_df
                    data_n_mod_df.loc[len(data_n_mod_df)] = dictionary
    return data_n_mod_df





# import datasets
data_path = Path(__file__).parent.parent / 'datasets'
# df = pd.read_csv(data_path,lineterminator='\n')

train_df = pd.read_csv(f'{data_path}/train_df.csv',lineterminator='\n')
sg_entities_patterns_df = pd.read_csv(f'{data_path}/sg_entities_patterns.csv',lineterminator='\n')

# load the models
model_path = Path(__file__).parent / 'models'

# keras model
model = tf.keras.models.load_model(f"{model_path}/keras_model_k.h5", custom_objects={"R2": R2 })

# scaler - minmax
mm_scaler = joblib.load(f'{model_path}/mm_scaler.gz')

# keras tokenizer
keras_tokenizer = joblib.load(f'{model_path}/keras_tokenizer.gz')

# load spaCy English model
nlp = spacy.load('en_core_web_sm')

# create entity ruler
ruler = nlp.add_pipe('entity_ruler',"ruleActions", config={"overwrite_ents": True})

# list of entities and patterns
# note that the text are lemmatized before pulling for entities. Thus, patterns should be in root form
lst_of_patterns = sg_entities_patterns_df.to_dict('records') # convert df to list of dictionary
patterns = lst_of_patterns

ruler.add_patterns(patterns)




# create some new dataframes to help plot graphs
# create df for top 10% and bottom 10% sentences with regards to sentiment
pct = 0.1
rows = len(train_df)

top_sentences = train_df.sort_values(by=['sentiment'],
                                     ascending=False).head(round(rows * pct))
bottom_sentences = train_df.sort_values(by=['sentiment'],
                                        ascending=True).head(round(rows * pct))



# create new dataframe for NPS
# generate a list of all entities mentioned
unique_entities_count_df = create_nps_dataframe(train_df,sg_entities_patterns_df)

# Create 2 new dataframes that contain positive and negative sentiment words and the corresponding count
pos_sentiment_words_df = create_sentiment_words_df(unique_entities_count_df,'pos_sentiment_words')
neg_sentiment_words_df = create_sentiment_words_df(unique_entities_count_df,'neg_sentiment_words')




# streamlit shell (layouts etc)
# set webpage name and icon
st.set_page_config(
    page_title='Singapore Tourist Attractions: an Analysis',
    page_icon='🏖',
    layout='wide',
    initial_sidebar_state='expanded'
    )

# top navigation bar
selected = option_menu(
    menu_title = None,
    options = ['NPS Predictor'], # 'Popularity Predictor'
    icons = ['search-heart','star-half'],
    default_index = 0, # which tab it should open when page is first loaded
    orientation = 'horizontal',
    styles={
        'nav-link-selected': {'background-color': '#4f8b84'}
        }
    )

if selected == 'NPS Predictor':
    # title
    st.title('Net Promoter Score (NPS) Predictor')
    st.subheader('for Singapore Tourist Attractions Based on TikTok Posts and Comments')
    style = "<div style='background-color:#fcefa7; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    # add a file uploader
    st.write('')
    st.subheader('Upload your own dataset and get predicted NPS')
    st.write('Upload a .csv file featuring a designated column titled "sentence." To ensure the highest level of accuracy in the predictions and enable thorough analysis, each row in the column should contain a succinct sentence that mentions a maximum of 1 tourist attraction. To compare across 2 different datasets, we have provided the option to upload 2 different files.')
    # split into 2 cols
    col_1_upload,space_upload,col_2_upload = st.columns([1,0.1,1])
    
    with col_1_upload:
        data_1 = st.file_uploader('Upload Data 1')
        if data_1 is not None:
            data_1_mod_df = pd.read_csv(data_1)
            st.success('Successfully Uploaded')
            # using clean_extract_dataframe function, clean and extract entities, pos, neg sentiment words
            data_1_mod_df = clean_extract_dataframe(data_1_mod_df)
            
            # params
            padding_type = 'post'
            trunc_type = 'post'
            max_length = 200 

            # padding
            # turn sentences into sequences of tokens
            data_1_mod_seq = keras_tokenizer.texts_to_sequences(data_1_mod_df['sentence'])
            # pad sentences so they are all the same length
            data_1_mod_seq_padded = pad_sequences(data_1_mod_seq, maxlen=max_length,
                                               padding=padding_type,truncating=trunc_type)

            # predict sentiment score
            data_1_mod_pred = model.predict(data_1_mod_seq_padded) # array
            # add predicted sentiment score
            data_1_mod_pred_df = pd.DataFrame(data_1_mod_pred,columns=['sentiment'])
            data_1_mod_df = pd.concat([data_1_mod_df, data_1_mod_pred_df],axis=1)
            # reverse scaling
            data_1_mod_pred_y = data_1_mod_df['sentiment']
            data_1_mod_pred_y = data_1_mod_pred_y.array
            data_1_mod_pred_y = mm_scaler.inverse_transform(data_1_mod_pred_y.reshape(-1,1))
            data_1_mod_df['sentiment'] = pd.DataFrame(data_1_mod_pred_y)
            # reverse normalization
            data_1_mod_df['sentiment'] = data_1_mod_df['sentiment'] ** 1.25

            # create some new dataframes to help plot graphs
            # create df for top 10% and bottom 10% sentences with regards to sentiment
            data_1_mod_top_sentences = data_1_mod_df.sort_values(by=['sentiment'],
                                                                 ascending=False).head(round(rows * pct))
            data_1_mod_bottom_sentences = data_1_mod_df.sort_values(by=['sentiment'],
                                                                    ascending=True).head(round(rows * pct))

            # create new dataframe for NPS
            # generate a list of all entities mentioned
            data_1_mod_unique_entities_count_df = create_nps_dataframe_without_ast_conversion(data_1_mod_df,sg_entities_patterns_df)

            # Create 2 new dataframes that contain positive and negative sentiment words and the corresponding count
            data_1_mod_pos_sentiment_words_df = create_sentiment_words_df(data_1_mod_unique_entities_count_df,'pos_sentiment_words')
            data_1_mod_neg_sentiment_words_df = create_sentiment_words_df(data_1_mod_unique_entities_count_df,'neg_sentiment_words')

            
            
    with col_2_upload:
        data_2 = st.file_uploader('Upload Data 2')
        if data_2 is not None:
            data_2_mod_df = pd.read_csv(data_2)
            st.success('Successfully Uploaded')
            
            # using clean_extract_dataframe function, clean and extract entities, pos, neg sentiment words
            data_2_mod_df = clean_extract_dataframe(data_2_mod_df)
            
            # params
            padding_type = 'post'
            trunc_type = 'post'
            max_length = 200 

            # padding
            # turn sentences into sequences of tokens
            data_2_mod_seq = keras_tokenizer.texts_to_sequences(data_2_mod_df['sentence'])
            # pad sentences so they are all the same length
            data_2_mod_seq_padded = pad_sequences(data_2_mod_seq, maxlen=max_length,
                                               padding=padding_type,truncating=trunc_type)

            # predict sentiment score
            data_2_mod_pred = model.predict(data_2_mod_seq_padded) # array
            # add predicted sentiment score
            data_2_mod_pred_df = pd.DataFrame(data_2_mod_pred,columns=['sentiment'])
            data_2_mod_df = pd.concat([data_2_mod_df, data_2_mod_pred_df],axis=1)
            # reverse scaling
            data_2_mod_pred_y = data_2_mod_df['sentiment']
            data_2_mod_pred_y = data_2_mod_pred_y.array
            data_2_mod_pred_y = mm_scaler.inverse_transform(data_2_mod_pred_y.reshape(-1,1))
            data_2_mod_df['sentiment'] = pd.DataFrame(data_2_mod_pred_y)
            # reverse normalization
            data_2_mod_df['sentiment'] = data_2_mod_df['sentiment'] ** 1.25

            # create some new dataframes to help plot graphs
            # create df for top 10% and bottom 10% sentences with regards to sentiment
            data_2_mod_top_sentences = data_2_mod_df.sort_values(by=['sentiment'],
                                                                 ascending=False).head(round(rows * pct))
            data_2_mod_bottom_sentences = data_2_mod_df.sort_values(by=['sentiment'],
                                                                    ascending=True).head(round(rows * pct))            

            # create new dataframe for NPS
            # generate a list of all entities mentioned
            data_2_mod_unique_entities_count_df = create_nps_dataframe_without_ast_conversion(data_2_mod_df,sg_entities_patterns_df)

            # Create 2 new dataframes that contain positive and negative sentiment words and the corresponding count
            data_2_mod_pos_sentiment_words_df = create_sentiment_words_df(data_2_mod_unique_entities_count_df,'pos_sentiment_words')
            data_2_mod_neg_sentiment_words_df = create_sentiment_words_df(data_2_mod_unique_entities_count_df,'neg_sentiment_words')

    
            
    
    st.write('')

    st.subheader('Analysis and Prediction')
    st.write('There are 6 views to choose from:')
    st.write('Sample Data: For illustration, we have provided data collected from 400 plus TikTok posts with the hashtags #singapore and #travel. These posts were uploaded between November 2011 and July 2023. You may use this as a reference point.')
    st.write('Uploaded Data 1: This wil be populated based on your uploaded dataset 1 above.')
    st.write('Uploaded Data 2: This wil be populated based on your uploaded dataset 2 above.')
    st.write('Sample vs Uploaded Data 1: Side-by-side analysis of Sample Data and Uploaded Data 1.')
    st.write('Sample vs Uploaded Data 2: Side-by-side analysis of Sample Data and Uploaded Data 2.')
    st.write('Uploaded Data 1 vs Uploaded Data 2: Side-by-side analysis of Uploaded Data 1 and Uploaded Data 2.')
    st.write('')

    # 3 different tabs to show 3 different views
    tab_1,tab_2,tab_3,tab_4,tab_5,tab_6 = st.tabs(['Sample Data', 'Uploaded Data 1','Uploaded Data 2','Sample vs Uploaded Data 1','Sample vs Uploaded Data 2','Uploaded Data 1 vs Uploaded Data 2'])

    # change font size of the tabs
    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.05rem;
        }
    </style>
    '''

    st.markdown(css, unsafe_allow_html=True)
    
    with tab_1:
        # START PLOTTING ANALYSIS
        
        # compare count of selected entities
        st.write('')
        st.title('Compare Tourist Attractions')
        st.subheader('Sample Data')
        
        # change the color of the boxes that surrounds the selected tourist attraction
        st.markdown(
            """
        <style>
        span[data-baseweb="tag"] {
          background-color: #4f8b84 !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        # create a multiselect function
        options = st.multiselect(
        'Select Tourist Attractions to Compare (Max. Selection: 5)',
        options=unique_entities_count_df['entity'].unique(),
        default=unique_entities_count_df['entity'].loc[0:1],
        max_selections=5,
        key='tab_1'
        )
        # using the selected options, compare them in various ways
        # first, create dataframe based on selection
        selected_df = unique_entities_count_df[unique_entities_count_df['entity'].isin(options)]
        # compare the frequency of occurrence
        st.subheader(f'Compare Frequency of Occurence')
        
        compare_freq_fig = px.bar(selected_df,
                             x='count',y='entity',
                             color='count',
                             color_continuous_scale="YlGn",
               )
        compare_freq_fig.update_layout(
            autosize=False,
            height=250,
            yaxis_title='Tourist Attractions',
                yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(compare_freq_fig,use_container_width=True)

        # compare the NPS
        st.subheader(f'Compare NPS')
        
        compare_nps_fig = px.bar(selected_df,
                             x='nps',y='entity',
                             color='nps',
                             color_continuous_scale="YlGn",
               )
        compare_nps_fig.update_layout(
            autosize=False,
            height=250,
            yaxis_title='Tourist Attractions',
                yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(compare_nps_fig,use_container_width=True)
            
        
        
        st.title('General Analysis of the Entire Dataset')
        st.subheader('Sample Data')
        # slider (unigram)
        st.subheader('Top n Frequently Occurring Unigrams')
        top_n_uni = st.slider('Select n', 1, 10, 6,key='unigram')

        # st.write(train_df.columns)
        
        
        # create bar charts for 90th percentile (unigram)
        st.write(f'Top {top_n_uni} Frequently Occurring Unigrams - Sentiment Score: 90th percentile (top 10%)')
        
        top_uni_fig = px.bar(top_n_grams(top_sentences['sentence'],
                           n=top_n_uni, ngram=(1,1), stop=stopwords),
                             x='count',y='text',
                             color='count',
                             color_continuous_scale="Reds",
               )
        top_uni_fig.update_layout(
            autosize=False,
            height=300,
            yaxis_title='Unigram',
                yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(top_uni_fig,use_container_width=True)

        # create bar charts for 10th percentile (unigram)

        st.write(f'Top {top_n_uni} Frequently Occurring Unigrams - Sentiment Score: 10th percentile (bottom 10%)')
        
        bottom_uni_fig = px.bar(top_n_grams(bottom_sentences['sentence'],
                           n=top_n_uni, ngram=(1,1), stop=stopwords),
                             x='count',y='text',
                             color='count',
                             color_continuous_scale="Blues",
               )
        bottom_uni_fig.update_layout(
            autosize=False,
            height=300,
            yaxis_title='Unigram',
                yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(bottom_uni_fig,use_container_width=True)


        # slider (bigram)
        st.subheader('Top n Frequently Occurring Bigrams')
        top_n_bi = st.slider('Select n', 1, 10, 6,key='bigram')

        # create bar charts for 90th percentile (Bigram)
        st.write(f'Top {top_n_bi} Frequently Occurring Bigrams - Sentiment Score: 90th percentile (top 10%)')
        
        top_bi_fig = px.bar(top_n_grams(top_sentences['sentence'],
                           n=top_n_bi, ngram=(2,2), stop=stopwords),
                             x='count',y='text',
                             color='count',
                             color_continuous_scale="Reds",
               )
        top_bi_fig.update_layout(
            autosize=False,
            height=300,
            yaxis_title='Bigram',
                yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(top_bi_fig,use_container_width=True)

        # create bar charts for 10th percentile (bigram)

        st.write(f'Top {top_n_bi} Frequently Occurring Bigrams - Sentiment Score: 10th percentile (bottom 10%)')
        
        bottom_bi_fig = px.bar(top_n_grams(bottom_sentences['sentence'],
                           n=top_n_bi, ngram=(2,2), stop=stopwords),
                             x='count',y='text',
                             color='count',
                             color_continuous_scale="Blues",
               )
        bottom_bi_fig.update_layout(
            autosize=False,
            height=300,
            yaxis_title='Bigram',
                yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(bottom_bi_fig,use_container_width=True)


        # slider (Top n Occuring Sentiment Words)
        st.subheader('Top n Occuring Sentiment Words')
        top_n_sentiment = st.slider('Select n', 1, 15, 12,key='sentiment')

        # plot top mentioned sentiment words
        st.write(f'Top {top_n_sentiment} Occuring Positive Sentiment Words')
            
        top_pos_sentiment_fig = px.bar(pos_sentiment_words_df.head(top_n_sentiment),x='count',y='sentiment_word', 
                        color='count',
                        color_continuous_scale="Reds")
        top_pos_sentiment_fig.update_layout(
                autosize=False,
                height=380,
                yaxis_title='Positive Sentiment Words',
                yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(top_pos_sentiment_fig,use_container_width=True)

        # plot top mentioned sentiment words
        st.write(f'Top {top_n_sentiment} Occuring Negative Sentiment Words')
            
        top_neg_sentiment_fig = px.bar(neg_sentiment_words_df.head(top_n_sentiment),x='count',y='sentiment_word', 
                        color='count',
                        color_continuous_scale="Blues")
        top_neg_sentiment_fig.update_layout(
                autosize=False,
                height=380,
                yaxis_title='Negative Sentiment Words',
                yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(top_neg_sentiment_fig,use_container_width=True)
        
        
        # slider (NPS of Top n Occuring Tourist Attractions)
        st.subheader('NPS of Top n Occuring Tourist Attractions')
        top_n_entities = st.slider('Select n', 1, 20, 8,key='entities')
        # sort by count
        unique_entities_count_df = unique_entities_count_df.sort_values(by=['count','nps','entity'],
                                                                        ascending=False)
        
        # plot top mentioned entities using 'unique_entities_count_df'
        st.write(f'NPS of Top {top_n_entities} Occuring Tourist Attractions - Entity Level')
        
        top_entities_fig = px.bar(unique_entities_count_df.head(top_n_entities),x='nps',y='entity',
                                  color='interest_1',
                                  color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
        top_entities_fig.update_layout(
            autosize=False,
            height=500,
            yaxis_title='Tourist Attraction',
            legend_title_text='Type of Tourist Attraction',
                yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(top_entities_fig,use_container_width=True)

        # plot top mentioned entities grouped by type of tourist attraction 
        st.write(f'NPS of Top {top_n_entities} Occuring Tourist Attractions - Grouped by Type')
        
        top_entities_type_fig = px.bar(unique_entities_count_df.filter(['interest_1','count','total_count','nps']).head(top_n_entities).groupby('interest_1').mean().reset_index(),
                                       x='nps',y='interest_1',
                                       color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
        top_entities_type_fig.update_layout(
            autosize=False,
            height=300,
            yaxis_title='Type of Tourist Attraction',
            legend_title_text='Type of Tourist Attraction',
                yaxis={'categoryorder': 'total ascending'},
            showlegend=False)
        st.plotly_chart(top_entities_type_fig,use_container_width=True)







    with tab_2:
        if data_1 is None:
            st.write('Please upload your own dataset in the section above (Data 1).')
        elif data_1 is not None:

            # START PLOTTING ANALYSIS
            
            # compare count of selected entities
            st.write('')
            st.title('Compare Tourist Attractions')
            st.subheader('Uploaded Data 1')
            
            # change the color of the boxes that surrounds the selected tourist attraction
            st.markdown(
                """
            <style>
            span[data-baseweb="tag"] {
              background-color: #4f8b84 !important;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )
            # create a multiselect function
            options_tab2 = st.multiselect(
            'Select Tourist Attractions to Compare (Max. Selection: 5)',
            options=data_1_mod_unique_entities_count_df['entity'].unique(),
            default=data_1_mod_unique_entities_count_df['entity'].loc[0:1],
            max_selections=5,
            key='tab_2'
            )
            # using the selected options, compare them in various ways
            # first, create dataframe based on selection
            selected_df_tab2 = data_1_mod_unique_entities_count_df[data_1_mod_unique_entities_count_df['entity'].isin(options_tab2)]
            # compare the frequency of occurrence
            st.subheader(f'Compare Frequency of Occurence')
            
            compare_freq_fig_tab2 = px.bar(selected_df_tab2,
                                 x='count',y='entity',
                                 color='count',
                                 color_continuous_scale="YlGn",
                   )
            compare_freq_fig_tab2.update_layout(
                autosize=False,
                height=250,
                yaxis_title='Tourist Attractions',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(compare_freq_fig_tab2,use_container_width=True)

            # compare the NPS
            st.subheader(f'Compare NPS')
            
            compare_nps_fig_tab2 = px.bar(selected_df_tab2,
                                 x='nps',y='entity',
                                 color='nps',
                                 color_continuous_scale="YlGn",
                   )
            compare_nps_fig_tab2.update_layout(
                autosize=False,
                height=250,
                yaxis_title='Tourist Attractions',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(compare_nps_fig_tab2,use_container_width=True)
                
            
            
            st.title('General Analysis of the Entire Dataset')
            st.subheader('Uploaded Data 1')
            # slider (unigram)
            st.subheader('Top n Frequently Occurring Unigrams')
            top_n_uni_tab2 = st.slider('Select n', 1, 10, 6,key='unigram_tab2')

            # st.write(train_df.columns)
            
            
            # create bar charts for 90th percentile (unigram)
            st.write(f'Top {top_n_uni_tab2} Frequently Occurring Unigrams - Sentiment Score: 90th percentile (top 10%)')
            
            top_uni_fig_tab2 = px.bar(top_n_grams(data_1_mod_top_sentences['sentence'],
                               n=top_n_uni_tab2, ngram=(1,1), stop=stopwords),
                                 x='count',y='text',
                                 color='count',
                                 color_continuous_scale="Reds",
                   )
            top_uni_fig_tab2.update_layout(
                autosize=False,
                height=300,
                yaxis_title='Unigram',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(top_uni_fig_tab2,use_container_width=True)

            # create bar charts for 10th percentile (unigram)

            st.write(f'Top {top_n_uni_tab2} Frequently Occurring Unigrams - Sentiment Score: 10th percentile (bottom 10%)')
            
            bottom_uni_fig_tab2 = px.bar(top_n_grams(data_1_mod_bottom_sentences['sentence'],
                               n=top_n_uni_tab2, ngram=(1,1), stop=stopwords),
                                 x='count',y='text',
                                 color='count',
                                 color_continuous_scale="Blues",
                   )
            bottom_uni_fig_tab2.update_layout(
                autosize=False,
                height=300,
                yaxis_title='Unigram',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(bottom_uni_fig_tab2,use_container_width=True)


            # slider (bigram)
            st.subheader('Top n Frequently Occurring Bigrams')
            top_n_bi_tab2 = st.slider('Select n', 1, 10, 6,key='bigram_tab2')

            # create bar charts for 90th percentile (Bigram)
            st.write(f'Top {top_n_bi_tab2} Frequently Occurring Bigrams - Sentiment Score: 90th percentile (top 10%)')
            
            top_bi_fig_tab2 = px.bar(top_n_grams(data_1_mod_top_sentences['sentence'],
                               n=top_n_bi_tab2, ngram=(2,2), stop=stopwords),
                                 x='count',y='text',
                                 color='count',
                                 color_continuous_scale="Reds",
                   )
            top_bi_fig_tab2.update_layout(
                autosize=False,
                height=300,
                yaxis_title='Bigram',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(top_bi_fig_tab2,use_container_width=True)

            # create bar charts for 10th percentile (bigram)

            st.write(f'Top {top_n_bi_tab2} Frequently Occurring Bigrams - Sentiment Score: 10th percentile (bottom 10%)')
            
            bottom_bi_fig_tab2 = px.bar(top_n_grams(data_1_mod_bottom_sentences['sentence'],
                               n=top_n_bi_tab2, ngram=(2,2), stop=stopwords),
                                 x='count',y='text',
                                 color='count',
                                 color_continuous_scale="Blues",
                   )
            bottom_bi_fig_tab2.update_layout(
                autosize=False,
                height=300,
                yaxis_title='Bigram',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(bottom_bi_fig_tab2,use_container_width=True)


            # slider (Top n Occuring Sentiment Words)
            st.subheader('Top n Occuring Sentiment Words')
            top_n_sentiment_tab2 = st.slider('Select n', 1, 15, 12,key='sentiment_tab2')

            # plot top mentioned sentiment words
            st.write(f'Top {top_n_sentiment_tab2} Occuring Positive Sentiment Words')
                
            top_pos_sentiment_fig_tab2 = px.bar(data_1_mod_pos_sentiment_words_df.head(top_n_sentiment_tab2),x='count',y='sentiment_word', 
                            color='count',
                            color_continuous_scale="Reds")
            top_pos_sentiment_fig_tab2.update_layout(
                    autosize=False,
                    height=380,
                    yaxis_title='Positive Sentiment Words',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(top_pos_sentiment_fig_tab2,use_container_width=True)

            # plot top mentioned sentiment words
            st.write(f'Top {top_n_sentiment_tab2} Occuring Negative Sentiment Words')
                
            top_neg_sentiment_fig_tab2 = px.bar(data_1_mod_neg_sentiment_words_df.head(top_n_sentiment_tab2),x='count',y='sentiment_word', 
                            color='count',
                            color_continuous_scale="Blues")
            top_neg_sentiment_fig_tab2.update_layout(
                    autosize=False,
                    height=380,
                    yaxis_title='Negative Sentiment Words',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(top_neg_sentiment_fig_tab2,use_container_width=True)
            
            
            # slider (NPS of Top n Occuring Tourist Attractions)
            st.subheader('NPS of Top n Occuring Tourist Attractions')
            top_n_entities_tab2 = st.slider('Select n', 1, 20, 8,key='entities_tab2')
            # sort by count
            data_1_mod_unique_entities_count_df = data_1_mod_unique_entities_count_df.sort_values(by=['count','nps','entity'],
                                                                            ascending=False)
            
            # plot top mentioned entities using 'unique_entities_count_df'
            st.write(f'NPS of Top {top_n_entities_tab2} Occuring Tourist Attractions - Entity Level')
            
            top_entities_fig_tab2 = px.bar(data_1_mod_unique_entities_count_df.head(top_n_entities_tab2),x='nps',y='entity', 
                        color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
            top_entities_fig_tab2.update_layout(
                autosize=False,
                height=500,
                yaxis_title='Tourist Attraction',
                legend_title_text='Type of Tourist Attraction',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(top_entities_fig_tab2,use_container_width=True)

            # plot top mentioned entities grouped by type of tourist attraction 
            st.write(f'NPS of Top {top_n_entities_tab2} Occuring Tourist Attractions - Grouped by Type')
            
            top_entities_type_fig_tab2 = px.bar(data_1_mod_unique_entities_count_df.filter(['interest_1','count','total_count','nps']).head(top_n_entities_tab2).groupby('interest_1').mean().reset_index(),
                                                x='nps',y='interest_1',
                                                color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
            top_entities_type_fig_tab2.update_layout(
                autosize=False,
                height=300,
                yaxis_title='Type of Tourist Attraction',
                legend_title_text='Type of Tourist Attraction',
                    yaxis={'categoryorder': 'total ascending'},
            showlegend=False)
            st.plotly_chart(top_entities_type_fig_tab2,use_container_width=True)







    with tab_3:
        if data_2 is None:
            st.write('Please upload your own dataset in the section above (Data 2).')
        elif data_2 is not None:

            # START PLOTTING ANALYSIS
            
            # compare count of selected entities
            st.write('')
            st.title('Compare Tourist Attractions')
            st.subheader('Uploaded Data 2')
            
            # change the color of the boxes that surrounds the selected tourist attraction
            st.markdown(
                """
            <style>
            span[data-baseweb="tag"] {
              background-color: #4f8b84 !important;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )
            # create a multiselect function
            options_tab3 = st.multiselect(
            'Select Tourist Attractions to Compare (Max. Selection: 5)',
            options=data_2_mod_unique_entities_count_df['entity'].unique(),
            default=data_2_mod_unique_entities_count_df['entity'].loc[0:1],
            max_selections=5,
            key='tab_3'
            )
            # using the selected options, compare them in various ways
            # first, create dataframe based on selection
            selected_df_tab3 = data_2_mod_unique_entities_count_df[data_2_mod_unique_entities_count_df['entity'].isin(options_tab3)]
            # compare the frequency of occurrence
            st.subheader(f'Compare Frequency of Occurence')
            
            compare_freq_fig_tab3 = px.bar(selected_df_tab3,
                                 x='count',y='entity',
                                 color='count',
                                 color_continuous_scale="YlGn",
                   )
            compare_freq_fig_tab3.update_layout(
                autosize=False,
                height=250,
                yaxis_title='Tourist Attractions',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(compare_freq_fig_tab3,use_container_width=True)

            # compare the NPS
            st.subheader(f'Compare NPS')
            
            compare_nps_fig_tab3 = px.bar(selected_df_tab3,
                                 x='nps',y='entity',
                                 color='nps',
                                 color_continuous_scale="YlGn",
                   )
            compare_nps_fig_tab3.update_layout(
                autosize=False,
                height=250,
                yaxis_title='Tourist Attractions',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(compare_nps_fig_tab3,use_container_width=True)
                
            
            
            st.title('General Analysis of the Entire Dataset')
            st.subheader('Uploaded Data 2')
            # slider (unigram)
            st.subheader('Top n Frequently Occurring Unigrams')
            top_n_uni_tab3 = st.slider('Select n', 1, 10, 6,key='unigram_tab3')

            # st.write(train_df.columns)
            
            
            # create bar charts for 90th percentile (unigram)
            st.write(f'Top {top_n_uni_tab3} Frequently Occurring Unigrams - Sentiment Score: 90th percentile (top 10%)')
            
            top_uni_fig_tab3 = px.bar(top_n_grams(data_2_mod_top_sentences['sentence'],
                               n=top_n_uni_tab3, ngram=(1,1), stop=stopwords),
                                 x='count',y='text',
                                 color='count',
                                 color_continuous_scale="Reds",
                   )
            top_uni_fig_tab3.update_layout(
                autosize=False,
                height=300,
                yaxis_title='Unigram',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(top_uni_fig_tab3,use_container_width=True)

            # create bar charts for 10th percentile (unigram)

            st.write(f'Top {top_n_uni_tab3} Frequently Occurring Unigrams - Sentiment Score: 10th percentile (bottom 10%)')
            
            bottom_uni_fig_tab3 = px.bar(top_n_grams(data_2_mod_bottom_sentences['sentence'],
                               n=top_n_uni_tab3, ngram=(1,1), stop=stopwords),
                                 x='count',y='text',
                                 color='count',
                                 color_continuous_scale="Blues",
                   )
            bottom_uni_fig_tab3.update_layout(
                autosize=False,
                height=300,
                yaxis_title='Unigram',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(bottom_uni_fig_tab3,use_container_width=True)


            # slider (bigram)
            st.subheader('Top n Frequently Occurring Bigrams')
            top_n_bi_tab3 = st.slider('Select n', 1, 10, 6,key='bigram_tab3')

            # create bar charts for 90th percentile (Bigram)
            st.write(f'Top {top_n_bi_tab3} Frequently Occurring Bigrams - Sentiment Score: 90th percentile (top 10%)')
            
            top_bi_fig_tab3 = px.bar(top_n_grams(data_2_mod_top_sentences['sentence'],
                               n=top_n_bi_tab3, ngram=(2,2), stop=stopwords),
                                 x='count',y='text',
                                 color='count',
                                 color_continuous_scale="Reds",
                   )
            top_bi_fig_tab3.update_layout(
                autosize=False,
                height=300,
                yaxis_title='Bigram',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(top_bi_fig_tab3,use_container_width=True)

            # create bar charts for 10th percentile (bigram)

            st.write(f'Top {top_n_bi_tab3} Frequently Occurring Bigrams - Sentiment Score: 10th percentile (bottom 10%)')
            
            bottom_bi_fig_tab3 = px.bar(top_n_grams(data_2_mod_bottom_sentences['sentence'],
                               n=top_n_bi_tab3, ngram=(2,2), stop=stopwords),
                                 x='count',y='text',
                                 color='count',
                                 color_continuous_scale="Blues",
                   )
            bottom_bi_fig_tab3.update_layout(
                autosize=False,
                height=300,
                yaxis_title='Bigram',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(bottom_bi_fig_tab3,use_container_width=True)


            # slider (Top n Occuring Sentiment Words)
            st.subheader('Top n Occuring Sentiment Words')
            top_n_sentiment_tab3 = st.slider('Select n', 1, 15, 12,key='sentiment_tab3')

            # plot top mentioned sentiment words
            st.write(f'Top {top_n_sentiment_tab3} Occuring Positive Sentiment Words')
                
            top_pos_sentiment_fig_tab3 = px.bar(data_2_mod_pos_sentiment_words_df.head(top_n_sentiment_tab3),x='count',y='sentiment_word', 
                            color='count',
                            color_continuous_scale="Reds")
            top_pos_sentiment_fig_tab3.update_layout(
                    autosize=False,
                    height=380,
                    yaxis_title='Positive Sentiment Words',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(top_pos_sentiment_fig_tab3,use_container_width=True)

            # plot top mentioned sentiment words
            st.write(f'Top {top_n_sentiment_tab3} Occuring Negative Sentiment Words')
                
            top_neg_sentiment_fig_tab3 = px.bar(data_2_mod_neg_sentiment_words_df.head(top_n_sentiment_tab3),x='count',y='sentiment_word', 
                            color='count',
                            color_continuous_scale="Blues")
            top_neg_sentiment_fig_tab3.update_layout(
                    autosize=False,
                    height=380,
                    yaxis_title='Negative Sentiment Words',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(top_neg_sentiment_fig_tab3,use_container_width=True)
            
            
            # slider (NPS of Top n Occuring Tourist Attractions)
            st.subheader('NPS of Top n Occuring Tourist Attractions')
            top_n_entities_tab3 = st.slider('Select n', 1, 20, 8,key='entities_tab3')
            # sort by count
            data_2_mod_unique_entities_count_df = data_2_mod_unique_entities_count_df.sort_values(by=['count','nps','entity'],
                                                                            ascending=False)
            
            # plot top mentioned entities using 'unique_entities_count_df'
            st.write(f'NPS of Top {top_n_entities_tab3} Occuring Tourist Attractions - Entity Level')
            
            top_entities_fig_tab3 = px.bar(data_2_mod_unique_entities_count_df.head(top_n_entities_tab3),x='nps',y='entity', 
                        color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
            top_entities_fig_tab3.update_layout(
                autosize=False,
                height=500,
                yaxis_title='Tourist Attraction',
                legend_title_text='Type of Tourist Attraction',
                    yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(top_entities_fig_tab3,use_container_width=True)

            # plot top mentioned entities grouped by type of tourist attraction 
            st.write(f'NPS of Top {top_n_entities_tab3} Occuring Tourist Attractions - Grouped by Type')
            
            top_entities_type_fig_tab3 = px.bar(data_2_mod_unique_entities_count_df.filter(['interest_1','count','total_count','nps']).head(top_n_entities_tab3).groupby('interest_1').mean().reset_index(),
                                                x='nps',y='interest_1',
                                                color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
            top_entities_type_fig_tab3.update_layout(
                autosize=False,
                height=300,
                yaxis_title='Type of Tourist Attraction',
                legend_title_text='Type of Tourist Attraction',
                    yaxis={'categoryorder': 'total ascending'},
            showlegend=False)
            st.plotly_chart(top_entities_type_fig_tab3,use_container_width=True)






    with tab_4:
        if data_1 is None:
            st.write('Please upload your own dataset in the section above (Data 1).')
        elif data_1 is not None:
            col_1_tab4, space_tab4, col_2_tab4 = st.columns([1,0.1,1])
            with col_1_tab4:
                # START PLOTTING ANALYSIS
        
                # compare count of selected entities
                st.write('')
                st.title('Compare Tourist Attractions')
                st.subheader('Sample Data')
                
                # change the color of the boxes that surrounds the selected tourist attraction
                st.markdown(
                    """
                <style>
                span[data-baseweb="tag"] {
                  background-color: #4f8b84 !important;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )
                # create a multiselect function
                options_col_1_tab4 = st.multiselect(
                'Select Tourist Attractions to Compare (Max. Selection: 5)',
                options=unique_entities_count_df['entity'].unique(),
                default=[unique_entities_count_df['entity'].loc[0],unique_entities_count_df['entity'].loc[1]],
                max_selections=5,
                key='col_1_tab4'
                )
                # using the selected options, compare them in various ways
                # first, create dataframe based on selection
                selected_df_col_1_tab4 = unique_entities_count_df[unique_entities_count_df['entity'].isin(options_col_1_tab4)]
                # compare the frequency of occurrence
                st.subheader(f'Compare Frequency of Occurence')
                
                compare_freq_fig_col_1_tab4 = px.bar(selected_df_col_1_tab4,
                                     x='count',y='entity',
                                     color='count',
                                     color_continuous_scale="YlGn",
                       )
                compare_freq_fig_col_1_tab4.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_freq_fig_col_1_tab4,use_container_width=True)

                # compare the NPS
                st.subheader(f'Compare NPS')
                
                compare_nps_fig_col_1_tab4 = px.bar(selected_df_col_1_tab4,
                                     x='nps',y='entity',
                                     color='nps',
                                     color_continuous_scale="YlGn",
                       )
                compare_nps_fig_col_1_tab4.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_nps_fig_col_1_tab4,use_container_width=True)
                    
                
                
                st.title('General Analysis of the Entire Dataset')
                st.subheader('Sample Data')
                # slider (unigram)
                st.subheader('Top n Frequently Occurring Unigrams')
                top_n_uni_col_1_tab4 = st.slider('Select n', 1, 10, 6,key='unigram_col_1_tab4')

                # st.write(train_df.columns)
                
                
                # create bar charts for 90th percentile (unigram)
                st.write(f'Top {top_n_uni_col_1_tab4} Frequently Occurring Unigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_uni_fig_col_1_tab4 = px.bar(top_n_grams(top_sentences['sentence'],
                                   n=top_n_uni_col_1_tab4, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_uni_fig_col_1_tab4.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_uni_fig_col_1_tab4,use_container_width=True)

                # create bar charts for 10th percentile (unigram)

                st.write(f'Top {top_n_uni_col_1_tab4} Frequently Occurring Unigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_uni_fig_col_1_tab4 = px.bar(top_n_grams(bottom_sentences['sentence'],
                                   n=top_n_uni_col_1_tab4, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_uni_fig_col_1_tab4.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_uni_fig_col_1_tab4,use_container_width=True)


                # slider (bigram)
                st.subheader('Top n Frequently Occurring Bigrams')
                top_n_bi_col_1_tab4 = st.slider('Select n', 1, 10, 6,key='bigram_col_1_tab4')

                # create bar charts for 90th percentile (Bigram)
                st.write(f'Top {top_n_bi_col_1_tab4} Frequently Occurring Bigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_bi_fig_col_1_tab4 = px.bar(top_n_grams(top_sentences['sentence'],
                                   n=top_n_bi_col_1_tab4, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_bi_fig_col_1_tab4.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_bi_fig_col_1_tab4,use_container_width=True)

                # create bar charts for 10th percentile (bigram)

                st.write(f'Top {top_n_bi_col_1_tab4} Frequently Occurring Bigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_bi_fig_col_1_tab4 = px.bar(top_n_grams(bottom_sentences['sentence'],
                                   n=top_n_bi_col_1_tab4, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_bi_fig_col_1_tab4.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_bi_fig_col_1_tab4,use_container_width=True)


                # slider (Top n Occuring Sentiment Words)
                st.subheader('Top n Occuring Sentiment Words')
                top_n_sentiment_col_1_tab4 = st.slider('Select n', 1, 15, 12,key='sentiment_col_1_tab4')

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_1_tab4} Occuring Positive Sentiment Words')
                    
                top_pos_sentiment_fig_col_1_tab4 = px.bar(pos_sentiment_words_df.head(top_n_sentiment_col_1_tab4),x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Reds")
                top_pos_sentiment_fig_col_1_tab4.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Positive Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_pos_sentiment_fig_col_1_tab4,use_container_width=True)

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_1_tab4} Occuring Negative Sentiment Words')
                    
                top_neg_sentiment_fig_col_1_tab4 = px.bar(neg_sentiment_words_df.head(top_n_sentiment_col_1_tab4),x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Blues")
                top_neg_sentiment_fig_col_1_tab4.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Negative Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_neg_sentiment_fig_col_1_tab4,use_container_width=True)
                
                
                # slider (NPS of Top n Occuring Tourist Attractions)
                st.subheader('NPS of Top n Occuring Tourist Attractions')
                top_n_entities_col_1_tab4 = st.slider('Select n', 1, 20, 8,key='entities_col_1_tab4')
                # sort by count
                unique_entities_count_df = unique_entities_count_df.sort_values(by=['count','nps','entity'],
                                                                                ascending=False)
                
                # plot top mentioned entities using 'unique_entities_count_df'
                st.write(f'NPS of Top {top_n_entities_col_1_tab4} Occuring Tourist Attractions - Entity Level')
                
                top_entities_fig_col_1_tab4 = px.bar(unique_entities_count_df.head(top_n_entities_col_1_tab4),x='nps',y='entity', 
                            color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
                top_entities_fig_col_1_tab4.update_layout(
                    autosize=False,
                    height=500,
                    yaxis_title='Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_entities_fig_col_1_tab4,use_container_width=True)

                # plot top mentioned entities grouped by type of tourist attraction 
                st.write(f'NPS of Top {top_n_entities_col_1_tab4} Occuring Tourist Attractions - Grouped by Type')
                
                top_entities_type_fig_col_1_tab4 = px.bar(unique_entities_count_df.filter(['interest_1','count','total_count','nps']).head(top_n_entities_col_1_tab4).groupby('interest_1').mean().reset_index(),
                                                    x='nps',y='interest_1',
                                                    color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
                top_entities_type_fig_col_1_tab4.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Type of Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'},
            showlegend=False)
                st.plotly_chart(top_entities_type_fig_col_1_tab4,use_container_width=True)

                
        
            with col_2_tab4:
                # START PLOTTING ANALYSIS
                
                # compare count of selected entities
                st.write('')
                st.title('Compare Tourist Attractions')
                st.subheader('Uploaded Data 1')
                
                # change the color of the boxes that surrounds the selected tourist attraction
                st.markdown(
                    """
                <style>
                span[data-baseweb="tag"] {
                  background-color: #4f8b84 !important;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )
                # create a multiselect function
                options_col_2_tab4 = st.multiselect(
                'Select Tourist Attractions to Compare (Max. Selection: 5)',
                options=data_1_mod_unique_entities_count_df['entity'].unique(),
                default=[data_1_mod_unique_entities_count_df['entity'].loc[0],data_1_mod_unique_entities_count_df['entity'].loc[1]],
                max_selections=5,
                key='col_2_tab4'
                )
                # using the selected options, compare them in various ways
                # first, create dataframe based on selection
                selected_df_col_2_tab4 = data_1_mod_unique_entities_count_df[data_1_mod_unique_entities_count_df['entity'].isin(options_col_2_tab4)]
                # compare the frequency of occurrence
                st.subheader(f'Compare Frequency of Occurence')
                
                compare_freq_fig_col_2_tab4 = px.bar(selected_df_col_2_tab4,
                                     x='count',y='entity',
                                     color='count',
                                     color_continuous_scale="YlGn",
                       )
                compare_freq_fig_col_2_tab4.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_freq_fig_col_2_tab4,use_container_width=True)

                # compare the NPS
                st.subheader(f'Compare NPS')
                
                compare_nps_fig_col_2_tab4 = px.bar(selected_df_col_2_tab4,
                                     x='nps',y='entity',
                                     color='nps',
                                     color_continuous_scale="YlGn",
                       )
                compare_nps_fig_col_2_tab4.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_nps_fig_col_2_tab4,use_container_width=True)
                    
                
                
                st.title('General Analysis of the Entire Dataset')
                st.subheader('Uploaded Data 1')
                # slider (unigram)
                st.subheader('Top n Frequently Occurring Unigrams')
                top_n_uni_col_2_tab4 = st.slider('Select n', 1, 10, 6,key='unigram_col_2_tab4')

                # st.write(train_df.columns)
                
                
                # create bar charts for 90th percentile (unigram)
                st.write(f'Top {top_n_uni_col_2_tab4} Frequently Occurring Unigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_uni_fig_col_2_tab4 = px.bar(top_n_grams(data_1_mod_top_sentences['sentence'],
                                   n=top_n_uni_col_2_tab4, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_uni_fig_col_2_tab4.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_uni_fig_col_2_tab4,use_container_width=True)

                # create bar charts for 10th percentile (unigram)

                st.write(f'Top {top_n_uni_col_2_tab4} Frequently Occurring Unigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_uni_fig_col_2_tab4 = px.bar(top_n_grams(data_1_mod_bottom_sentences['sentence'],
                                   n=top_n_uni_col_2_tab4, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_uni_fig_col_2_tab4.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_uni_fig_col_2_tab4,use_container_width=True)


                # slider (bigram)
                st.subheader('Top n Frequently Occurring Bigrams')
                top_n_bi_col_2_tab4 = st.slider('Select n', 1, 10, 6,key='bigram_col_2_tab4')

                # create bar charts for 90th percentile (Bigram)
                st.write(f'Top {top_n_bi_col_2_tab4} Frequently Occurring Bigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_bi_fig_col_2_tab4 = px.bar(top_n_grams(data_1_mod_top_sentences['sentence'],
                                   n=top_n_bi_col_2_tab4, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_bi_fig_col_2_tab4.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_bi_fig_col_2_tab4,use_container_width=True)

                # create bar charts for 10th percentile (bigram)

                st.write(f'Top {top_n_bi_col_2_tab4} Frequently Occurring Bigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_bi_fig_col_2_tab4 = px.bar(top_n_grams(data_1_mod_bottom_sentences['sentence'],
                                   n=top_n_bi_col_2_tab4, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_bi_fig_col_2_tab4.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_bi_fig_col_2_tab4,use_container_width=True)


                # slider (Top n Occuring Sentiment Words)
                st.subheader('Top n Occuring Sentiment Words')
                top_n_sentiment_col_2_tab4 = st.slider('Select n', 1, 15, 12,key='sentiment_col_2_tab4')

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_2_tab4} Occuring Positive Sentiment Words')
                    
                top_pos_sentiment_fig_col_2_tab4 = px.bar(data_1_mod_pos_sentiment_words_df.head(top_n_sentiment_col_2_tab4),x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Reds")
                top_pos_sentiment_fig_col_2_tab4.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Positive Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_pos_sentiment_fig_col_2_tab4,use_container_width=True)

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_2_tab4} Occuring Negative Sentiment Words')
                    
                top_neg_sentiment_fig_col_2_tab4 = px.bar(data_1_mod_neg_sentiment_words_df.head(top_n_sentiment_col_2_tab4),x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Blues")
                top_neg_sentiment_fig_col_2_tab4.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Negative Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_neg_sentiment_fig_col_2_tab4,use_container_width=True)
                
                
                # slider (NPS of Top n Occuring Tourist Attractions)
                st.subheader('NPS of Top n Occuring Tourist Attractions')
                top_n_entities_col_2_tab4 = st.slider('Select n', 1, 20, 8,key='entities_col_2_tab4')
                # sort by count
                data_1_mod_unique_entities_count_df = data_1_mod_unique_entities_count_df.sort_values(by=['count','nps','entity'],
                                                                                ascending=False)
                
                # plot top mentioned entities using 'unique_entities_count_df'
                st.write(f'NPS of Top {top_n_entities_col_2_tab4} Occuring Tourist Attractions - Entity Level')
                
                top_entities_fig_col_2_tab4 = px.bar(data_1_mod_unique_entities_count_df.head(top_n_entities_col_2_tab4),x='nps',y='entity', 
                            color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
                top_entities_fig_col_2_tab4.update_layout(
                    autosize=False,
                    height=500,
                    yaxis_title='Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_entities_fig_col_2_tab4,use_container_width=True)

                # plot top mentioned entities grouped by type of tourist attraction 
                st.write(f'NPS of Top {top_n_entities_col_2_tab4} Occuring Tourist Attractions - Grouped by Type')
                
                top_entities_type_fig_col_2_tab4 = px.bar(data_1_mod_unique_entities_count_df.filter(['interest_1','count','total_count','nps']).head(top_n_entities_col_2_tab4).groupby('interest_1').mean().reset_index(),
                                                    x='nps',y='interest_1',
                                                    color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
                top_entities_type_fig_col_2_tab4.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Type of Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'},
            showlegend=False)
                st.plotly_chart(top_entities_type_fig_col_2_tab4,use_container_width=True)




    with tab_5:
        if data_2 is None:
            st.write('Please upload your own dataset in the section above (Data 2).')
        elif data_2 is not None:
            col_1_tab5, space_tab5, col_2_tab5 = st.columns([1,0.1,1])
            with col_1_tab5:
                # START PLOTTING ANALYSIS
        
                # compare count of selected entities
                st.write('')
                st.title('Compare Tourist Attractions')
                st.subheader('Sample Data')
                
                # change the color of the boxes that surrounds the selected tourist attraction
                st.markdown(
                    """
                <style>
                span[data-baseweb="tag"] {
                  background-color: #4f8b84 !important;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )
                # create a multiselect function
                options_col_1_tab5 = st.multiselect(
                'Select Tourist Attractions to Compare (Max. Selection: 5)',
                options=unique_entities_count_df['entity'].unique(),
                default=[unique_entities_count_df['entity'].loc[0],unique_entities_count_df['entity'].loc[1]],
                max_selections=5,
                key='col_1_tab5'
                )
                # using the selected options, compare them in various ways
                # first, create dataframe based on selection
                selected_df_col_1_tab5 = unique_entities_count_df[unique_entities_count_df['entity'].isin(options_col_1_tab5)]
                # compare the frequency of occurrence
                st.subheader(f'Compare Frequency of Occurence')
                
                compare_freq_fig_col_1_tab5 = px.bar(selected_df_col_1_tab5,
                                     x='count',y='entity',
                                     color='count',
                                     color_continuous_scale="YlGn",
                       )
                compare_freq_fig_col_1_tab5.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_freq_fig_col_1_tab5,use_container_width=True)

                # compare the NPS
                st.subheader(f'Compare NPS')
                
                compare_nps_fig_col_1_tab5 = px.bar(selected_df_col_1_tab5,
                                     x='nps',y='entity',
                                     color='nps',
                                     color_continuous_scale="YlGn",
                       )
                compare_nps_fig_col_1_tab5.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_nps_fig_col_1_tab5,use_container_width=True)
                    
                
                
                st.title('General Analysis of the Entire Dataset')
                st.subheader('Sample Data')
                # slider (unigram)
                st.subheader('Top n Frequently Occurring Unigrams')
                top_n_uni_col_1_tab5 = st.slider('Select n', 1, 10, 6,key='unigram_col_1_tab5')

                # st.write(train_df.columns)
                
                
                # create bar charts for 90th percentile (unigram)
                st.write(f'Top {top_n_uni_col_1_tab5} Frequently Occurring Unigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_uni_fig_col_1_tab5 = px.bar(top_n_grams(top_sentences['sentence'],
                                   n=top_n_uni_col_1_tab5, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_uni_fig_col_1_tab5.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_uni_fig_col_1_tab5,use_container_width=True)

                # create bar charts for 10th percentile (unigram)

                st.write(f'Top {top_n_uni_col_1_tab5} Frequently Occurring Unigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_uni_fig_col_1_tab5 = px.bar(top_n_grams(bottom_sentences['sentence'],
                                   n=top_n_uni_col_1_tab5, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_uni_fig_col_1_tab5.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_uni_fig_col_1_tab5,use_container_width=True)


                # slider (bigram)
                st.subheader('Top n Frequently Occurring Bigrams')
                top_n_bi_col_1_tab5 = st.slider('Select n', 1, 10, 6,key='bigram_col_1_tab5')

                # create bar charts for 90th percentile (Bigram)
                st.write(f'Top {top_n_bi_col_1_tab5} Frequently Occurring Bigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_bi_fig_col_1_tab5 = px.bar(top_n_grams(top_sentences['sentence'],
                                   n=top_n_bi_col_1_tab5, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_bi_fig_col_1_tab5.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_bi_fig_col_1_tab5,use_container_width=True)

                # create bar charts for 10th percentile (bigram)

                st.write(f'Top {top_n_bi_col_1_tab5} Frequently Occurring Bigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_bi_fig_col_1_tab5 = px.bar(top_n_grams(bottom_sentences['sentence'],
                                   n=top_n_bi_col_1_tab5, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_bi_fig_col_1_tab5.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_bi_fig_col_1_tab5,use_container_width=True)


                # slider (Top n Occuring Sentiment Words)
                st.subheader('Top n Occuring Sentiment Words')
                top_n_sentiment_col_1_tab5 = st.slider('Select n', 1, 15, 12,key='sentiment_col_1_tab5')

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_1_tab5} Occuring Positive Sentiment Words')
                    
                top_pos_sentiment_fig_col_1_tab5 = px.bar(pos_sentiment_words_df.head(top_n_sentiment_col_1_tab5),
                                                          x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Reds")
                top_pos_sentiment_fig_col_1_tab5.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Positive Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_pos_sentiment_fig_col_1_tab5,use_container_width=True)

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_1_tab5} Occuring Negative Sentiment Words')
                    
                top_neg_sentiment_fig_col_1_tab5 = px.bar(neg_sentiment_words_df.head(top_n_sentiment_col_1_tab5),
                                                          x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Blues")
                top_neg_sentiment_fig_col_1_tab5.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Negative Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_neg_sentiment_fig_col_1_tab5,use_container_width=True)
                
                
                # slider (NPS of Top n Occuring Tourist Attractions)
                st.subheader('NPS of Top n Occuring Tourist Attractions')
                top_n_entities_col_1_tab5 = st.slider('Select n', 1, 20, 8,key='entities_col_1_tab5')
                # sort by count
                unique_entities_count_df = unique_entities_count_df.sort_values(by=['count','nps','entity'],
                                                                                ascending=False)
                
                # plot top mentioned entities using 'unique_entities_count_df'
                st.write(f'NPS of Top {top_n_entities_col_1_tab5} Occuring Tourist Attractions - Entity Level')
                
                top_entities_fig_col_1_tab5 = px.bar(unique_entities_count_df.head(top_n_entities_col_1_tab5),x='nps',y='entity', 
                            color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
                top_entities_fig_col_1_tab5.update_layout(
                    autosize=False,
                    height=500,
                    yaxis_title='Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_entities_fig_col_1_tab5,use_container_width=True)

                # plot top mentioned entities grouped by type of tourist attraction 
                st.write(f'NPS of Top {top_n_entities_col_1_tab5} Occuring Tourist Attractions - Grouped by Type')
                
                top_entities_type_fig_col_1_tab5 = px.bar(unique_entities_count_df.filter(['interest_1','count','total_count','nps']).head(top_n_entities_col_1_tab5).groupby('interest_1').mean().reset_index(),
                                                    x='nps',y='interest_1',
                                                    color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
                top_entities_type_fig_col_1_tab5.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Type of Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'},
            showlegend=False)
                st.plotly_chart(top_entities_type_fig_col_1_tab5,use_container_width=True)

                
        
            with col_2_tab5:
                # START PLOTTING ANALYSIS
                
                # compare count of selected entities
                st.write('')
                st.title('Compare Tourist Attractions')
                st.subheader('Uploaded Data 2')
                
                # change the color of the boxes that surrounds the selected tourist attraction
                st.markdown(
                    """
                <style>
                span[data-baseweb="tag"] {
                  background-color: #4f8b84 !important;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )
                # create a multiselect function
                options_col_2_tab5 = st.multiselect(
                'Select Tourist Attractions to Compare (Max. Selection: 5)',
                options=data_2_mod_unique_entities_count_df['entity'].unique(),
                default=[data_2_mod_unique_entities_count_df['entity'].loc[0],data_2_mod_unique_entities_count_df['entity'].loc[1]],
                max_selections=5,
                key='col_2_tab5'
                )
                # using the selected options, compare them in various ways
                # first, create dataframe based on selection
                selected_df_col_2_tab5 = data_2_mod_unique_entities_count_df[data_2_mod_unique_entities_count_df['entity'].isin(options_col_2_tab5)]
                # compare the frequency of occurrence
                st.subheader(f'Compare Frequency of Occurence')
                
                compare_freq_fig_col_2_tab5 = px.bar(selected_df_col_2_tab5,
                                     x='count',y='entity',
                                     color='count',
                                     color_continuous_scale="YlGn",
                       )
                compare_freq_fig_col_2_tab5.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_freq_fig_col_2_tab5,use_container_width=True)

                # compare the NPS
                st.subheader(f'Compare NPS')
                
                compare_nps_fig_col_2_tab5 = px.bar(selected_df_col_2_tab5,
                                     x='nps',y='entity',
                                     color='nps',
                                     color_continuous_scale="YlGn",
                       )
                compare_nps_fig_col_2_tab5.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_nps_fig_col_2_tab5,use_container_width=True)
                    
                
                
                st.title('General Analysis of the Entire Dataset')
                st.subheader('Uploaded Data 2')
                # slider (unigram)
                st.subheader('Top n Frequently Occurring Unigrams')
                top_n_uni_col_2_tab5 = st.slider('Select n', 1, 10, 6,key='unigram_col_2_tab5')

                # st.write(train_df.columns)
                
                
                # create bar charts for 90th percentile (unigram)
                st.write(f'Top {top_n_uni_col_2_tab5} Frequently Occurring Unigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_uni_fig_col_2_tab5 = px.bar(top_n_grams(data_2_mod_top_sentences['sentence'],
                                   n=top_n_uni_col_2_tab5, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_uni_fig_col_2_tab5.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_uni_fig_col_2_tab5,use_container_width=True)

                # create bar charts for 10th percentile (unigram)

                st.write(f'Top {top_n_uni_col_2_tab5} Frequently Occurring Unigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_uni_fig_col_2_tab5 = px.bar(top_n_grams(data_2_mod_bottom_sentences['sentence'],
                                   n=top_n_uni_col_2_tab5, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_uni_fig_col_2_tab5.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_uni_fig_col_2_tab5,use_container_width=True)


                # slider (bigram)
                st.subheader('Top n Frequently Occurring Bigrams')
                top_n_bi_col_2_tab5 = st.slider('Select n', 1, 10, 6,key='bigram_col_2_tab5')

                # create bar charts for 90th percentile (Bigram)
                st.write(f'Top {top_n_bi_col_2_tab5} Frequently Occurring Bigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_bi_fig_col_2_tab5 = px.bar(top_n_grams(data_2_mod_top_sentences['sentence'],
                                   n=top_n_bi_col_2_tab5, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_bi_fig_col_2_tab5.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_bi_fig_col_2_tab5,use_container_width=True)

                # create bar charts for 10th percentile (bigram)

                st.write(f'Top {top_n_bi_col_2_tab5} Frequently Occurring Bigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_bi_fig_col_2_tab5 = px.bar(top_n_grams(data_2_mod_bottom_sentences['sentence'],
                                   n=top_n_bi_col_2_tab5, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_bi_fig_col_2_tab5.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_bi_fig_col_2_tab5,use_container_width=True)


                # slider (Top n Occuring Sentiment Words)
                st.subheader('Top n Occuring Sentiment Words')
                top_n_sentiment_col_2_tab5 = st.slider('Select n', 1, 15, 12,key='sentiment_col_2_tab5')

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_2_tab5} Occuring Positive Sentiment Words')
                    
                top_pos_sentiment_fig_col_2_tab5 = px.bar(data_2_mod_pos_sentiment_words_df.head(top_n_sentiment_col_2_tab5),x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Reds")
                top_pos_sentiment_fig_col_2_tab5.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Positive Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_pos_sentiment_fig_col_2_tab5,use_container_width=True)

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_2_tab5} Occuring Negative Sentiment Words')
                    
                top_neg_sentiment_fig_col_2_tab5 = px.bar(data_2_mod_neg_sentiment_words_df.head(top_n_sentiment_col_2_tab5),x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Blues")
                top_neg_sentiment_fig_col_2_tab5.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Negative Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_neg_sentiment_fig_col_2_tab5,use_container_width=True)
                
                
                # slider (NPS of Top n Occuring Tourist Attractions)
                st.subheader('NPS of Top n Occuring Tourist Attractions')
                top_n_entities_col_2_tab5 = st.slider('Select n', 1, 20, 8,key='entities_col_2_tab5')
                # sort by count
                data_2_mod_unique_entities_count_df = data_2_mod_unique_entities_count_df.sort_values(by=['count','nps','entity'],
                                                                                ascending=False)
                
                # plot top mentioned entities using 'unique_entities_count_df'
                st.write(f'NPS of Top {top_n_entities_col_2_tab5} Occuring Tourist Attractions - Entity Level')
                
                top_entities_fig_col_2_tab5 = px.bar(data_2_mod_unique_entities_count_df.head(top_n_entities_col_2_tab5),x='nps',y='entity', 
                            color='interest_1',
                                           color_discrete_map={'transport':'#F8A19F',
                                                               'art':'#3283FE',
                                                               'museum':'#85660D',
                                                               'nature':'#00A08B',
                                                               'architecture':'#565656',
                                                               'park':'#1C8356',
                                                               'culture':'#FF9616',
                                                               'food and drinks':'#F7E1A0',
                                                               'religion':'#E2E2E2',
                                                               'shopping':'#00B5F7',
                                                               'recreation':'#C9FBE5',
                                                               'wildlife park':'#DEA0FD',
                                                               'amusement park':'#D62728',
                                                               'historic':'#325A9B',
                                                               'heritage':'#FEAF16',
                                                               'payment':'#AA0DFE'})
                top_entities_fig_col_2_tab5.update_layout(
                    autosize=False,
                    height=500,
                    yaxis_title='Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_entities_fig_col_2_tab5,use_container_width=True)

                # plot top mentioned entities grouped by type of tourist attraction 
                st.write(f'NPS of Top {top_n_entities_col_2_tab5} Occuring Tourist Attractions - Grouped by Type')
                
                top_entities_type_fig_col_2_tab5 = px.bar(data_2_mod_unique_entities_count_df.filter(['interest_1','count','total_count','nps']).head(top_n_entities_col_2_tab5).groupby('interest_1').mean().reset_index(),
                                                    x='nps',y='interest_1',
                                                    color='interest_1',
                                           color_discrete_map={'transport':'#F8A19F',
                                                               'art':'#3283FE',
                                                               'museum':'#85660D',
                                                               'nature':'#00A08B',
                                                               'architecture':'#565656',
                                                               'park':'#1C8356',
                                                               'culture':'#FF9616',
                                                               'food and drinks':'#F7E1A0',
                                                               'religion':'#E2E2E2',
                                                               'shopping':'#00B5F7',
                                                               'recreation':'#C9FBE5',
                                                               'wildlife park':'#DEA0FD',
                                                               'amusement park':'#D62728',
                                                               'historic':'#325A9B',
                                                               'heritage':'#FEAF16',
                                                               'payment':'#AA0DFE'})
                top_entities_type_fig_col_2_tab5.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Type of Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'},
            showlegend=False)
                st.plotly_chart(top_entities_type_fig_col_2_tab5,use_container_width=True)





    with tab_6:
        if data_1 is None and data_2 is None:
            st.write('Please upload your own dataset in the section above (Data 1 and Data 2).')
        elif data_1 is None and data_2 is not None:
            st.write('Please upload your own dataset in the section above (Data 1).')
        elif data_1 is not None and data_2 is None:
            st.write('Please upload your own dataset in the section above (Data 2).')
        else:
            col_1_tab6, space_tab6, col_2_tab6 = st.columns([1,0.1,1])
            with col_1_tab6:
                # START PLOTTING ANALYSIS
                
                # compare count of selected entities
                st.write('')
                st.title('Compare Tourist Attractions')
                st.subheader('Uploaded Data 1')
                
                # change the color of the boxes that surrounds the selected tourist attraction
                st.markdown(
                    """
                <style>
                span[data-baseweb="tag"] {
                  background-color: #4f8b84 !important;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )
                # create a multiselect function
                options_col_1_tab6 = st.multiselect(
                'Select Tourist Attractions to Compare (Max. Selection: 5)',
                options=data_1_mod_unique_entities_count_df['entity'].unique(),
                default=[data_1_mod_unique_entities_count_df['entity'].loc[0],data_1_mod_unique_entities_count_df['entity'].loc[1]],
                max_selections=5,
                key='col_1_tab6'
                )
                # using the selected options, compare them in various ways
                # first, create dataframe based on selection
                selected_df_col_1_tab6 = data_1_mod_unique_entities_count_df[data_1_mod_unique_entities_count_df['entity'].isin(options_col_1_tab6)]
                # compare the frequency of occurrence
                st.subheader(f'Compare Frequency of Occurence')
                
                compare_freq_fig_col_1_tab6 = px.bar(selected_df_col_1_tab6,
                                     x='count',y='entity',
                                     color='count',
                                     color_continuous_scale="YlGn",
                       )
                compare_freq_fig_col_1_tab6.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_freq_fig_col_1_tab6,use_container_width=True)

                # compare the NPS
                st.subheader(f'Compare NPS')
                
                compare_nps_fig_col_1_tab6 = px.bar(selected_df_col_1_tab6,
                                     x='nps',y='entity',
                                     color='nps',
                                     color_continuous_scale="YlGn",
                       )
                compare_nps_fig_col_1_tab6.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_nps_fig_col_1_tab6,use_container_width=True)
                    
                
                
                st.title('General Analysis of the Entire Dataset')
                st.subheader('Uploaded Data 1')
                # slider (unigram)
                st.subheader('Top n Frequently Occurring Unigrams')
                top_n_uni_col_1_tab6 = st.slider('Select n', 1, 10, 6,key='unigram_col_1_tab6')

                # st.write(train_df.columns)
                
                
                # create bar charts for 90th percentile (unigram)
                st.write(f'Top {top_n_uni_col_1_tab6} Frequently Occurring Unigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_uni_fig_col_1_tab6 = px.bar(top_n_grams(data_1_mod_top_sentences['sentence'],
                                   n=top_n_uni_col_1_tab6, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_uni_fig_col_1_tab6.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_uni_fig_col_1_tab6,use_container_width=True)

                # create bar charts for 10th percentile (unigram)

                st.write(f'Top {top_n_uni_col_1_tab6} Frequently Occurring Unigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_uni_fig_col_1_tab6 = px.bar(top_n_grams(data_1_mod_bottom_sentences['sentence'],
                                   n=top_n_uni_col_1_tab6, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_uni_fig_col_1_tab6.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_uni_fig_col_1_tab6,use_container_width=True)


                # slider (bigram)
                st.subheader('Top n Frequently Occurring Bigrams')
                top_n_bi_col_1_tab6 = st.slider('Select n', 1, 10, 6,key='bigram_col_1_tab6')

                # create bar charts for 90th percentile (Bigram)
                st.write(f'Top {top_n_bi_col_1_tab6} Frequently Occurring Bigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_bi_fig_col_1_tab6 = px.bar(top_n_grams(data_1_mod_top_sentences['sentence'],
                                   n=top_n_bi_col_1_tab6, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_bi_fig_col_1_tab6.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_bi_fig_col_1_tab6,use_container_width=True)

                # create bar charts for 10th percentile (bigram)

                st.write(f'Top {top_n_bi_col_1_tab6} Frequently Occurring Bigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_bi_fig_col_1_tab6 = px.bar(top_n_grams(data_1_mod_bottom_sentences['sentence'],
                                   n=top_n_bi_col_1_tab6, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_bi_fig_col_1_tab6.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_bi_fig_col_1_tab6,use_container_width=True)


                # slider (Top n Occuring Sentiment Words)
                st.subheader('Top n Occuring Sentiment Words')
                top_n_sentiment_col_1_tab6 = st.slider('Select n', 1, 15, 12,key='sentiment_col_1_tab6')

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_1_tab6} Occuring Positive Sentiment Words')
                    
                top_pos_sentiment_fig_col_1_tab6 = px.bar(data_1_mod_pos_sentiment_words_df.head(top_n_sentiment_col_1_tab6),x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Reds")
                top_pos_sentiment_fig_col_1_tab6.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Positive Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_pos_sentiment_fig_col_1_tab6,use_container_width=True)

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_1_tab6} Occuring Negative Sentiment Words')
                    
                top_neg_sentiment_fig_col_1_tab6 = px.bar(data_1_mod_neg_sentiment_words_df.head(top_n_sentiment_col_1_tab6),x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Blues")
                top_neg_sentiment_fig_col_1_tab6.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Negative Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_neg_sentiment_fig_col_1_tab6,use_container_width=True)
                
                
                # slider (NPS of Top n Occuring Tourist Attractions)
                st.subheader('NPS of Top n Occuring Tourist Attractions')
                top_n_entities_col_1_tab6 = st.slider('Select n', 1, 20, 8,key='entities_col_1_tab6')
                # sort by count
                data_1_mod_unique_entities_count_df = data_1_mod_unique_entities_count_df.sort_values(by=['count','nps','entity'],
                                                                                ascending=False)
                
                # plot top mentioned entities using 'unique_entities_count_df'
                st.write(f'NPS of Top {top_n_entities_col_1_tab6} Occuring Tourist Attractions - Entity Level')
                
                top_entities_fig_col_1_tab6 = px.bar(data_1_mod_unique_entities_count_df.head(top_n_entities_col_1_tab6),x='nps',y='entity', 
                            color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
                top_entities_fig_col_1_tab6.update_layout(
                    autosize=False,
                    height=500,
                    yaxis_title='Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_entities_fig_col_1_tab6,use_container_width=True)

                # plot top mentioned entities grouped by type of tourist attraction 
                st.write(f'NPS of Top {top_n_entities_col_1_tab6} Occuring Tourist Attractions - Grouped by Type')
                
                top_entities_type_fig_col_1_tab6 = px.bar(data_1_mod_unique_entities_count_df.filter(['interest_1','count','total_count','nps']).head(top_n_entities_col_1_tab6).groupby('interest_1').mean().reset_index(),
                                                    x='nps',y='interest_1',
                                                    color='interest_1',
                                       color_discrete_map={'transport':'#F8A19F',
                                                           'art':'#3283FE',
                                                           'museum':'#85660D',
                                                           'nature':'#00A08B',
                                                           'architecture':'#565656',
                                                           'park':'#1C8356',
                                                           'culture':'#FF9616',
                                                           'food and drinks':'#F7E1A0',
                                                           'religion':'#E2E2E2',
                                                           'shopping':'#00B5F7',
                                                           'recreation':'#C9FBE5',
                                                           'wildlife park':'#DEA0FD',
                                                           'amusement park':'#D62728',
                                                           'historic':'#325A9B',
                                                           'heritage':'#FEAF16',
                                                           'payment':'#AA0DFE'})
                top_entities_type_fig_col_1_tab6.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Type of Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'},
            showlegend=False)
                st.plotly_chart(top_entities_type_fig_col_1_tab6,use_container_width=True)

                
        
            with col_2_tab6:
                # START PLOTTING ANALYSIS
                
                # compare count of selected entities
                st.write('')
                st.title('Compare Tourist Attractions')
                st.subheader('Uploaded Data 2')
                
                # change the color of the boxes that surrounds the selected tourist attraction
                st.markdown(
                    """
                <style>
                span[data-baseweb="tag"] {
                  background-color: #4f8b84 !important;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )
                # create a multiselect function
                options_col_2_tab6 = st.multiselect(
                'Select Tourist Attractions to Compare (Max. Selection: 5)',
                options=data_2_mod_unique_entities_count_df['entity'].unique(),
                default=[data_2_mod_unique_entities_count_df['entity'].loc[0],data_2_mod_unique_entities_count_df['entity'].loc[1]],
                max_selections=5,
                key='col_2_tab6'
                )
                # using the selected options, compare them in various ways
                # first, create dataframe based on selection
                selected_df_col_2_tab6 = data_2_mod_unique_entities_count_df[data_2_mod_unique_entities_count_df['entity'].isin(options_col_2_tab6)]
                # compare the frequency of occurrence
                st.subheader(f'Compare Frequency of Occurence')
                
                compare_freq_fig_col_2_tab6 = px.bar(selected_df_col_2_tab6,
                                     x='count',y='entity',
                                     color='count',
                                     color_continuous_scale="YlGn",
                       )
                compare_freq_fig_col_2_tab6.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_freq_fig_col_2_tab6,use_container_width=True)

                # compare the NPS
                st.subheader(f'Compare NPS')
                
                compare_nps_fig_col_2_tab6 = px.bar(selected_df_col_2_tab6,
                                     x='nps',y='entity',
                                     color='nps',
                                     color_continuous_scale="YlGn",
                       )
                compare_nps_fig_col_2_tab6.update_layout(
                    autosize=False,
                    height=250,
                    yaxis_title='Tourist Attractions',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(compare_nps_fig_col_2_tab6,use_container_width=True)
                    
                
                
                st.title('General Analysis of the Entire Dataset')
                st.subheader('Uploaded Data 2')
                # slider (unigram)
                st.subheader('Top n Frequently Occurring Unigrams')
                top_n_uni_col_2_tab6 = st.slider('Select n', 1, 10, 6,key='unigram_col_2_tab6')

                # st.write(train_df.columns)
                
                
                # create bar charts for 90th percentile (unigram)
                st.write(f'Top {top_n_uni_col_2_tab6} Frequently Occurring Unigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_uni_fig_col_2_tab6 = px.bar(top_n_grams(data_2_mod_top_sentences['sentence'],
                                   n=top_n_uni_col_2_tab6, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_uni_fig_col_2_tab6.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_uni_fig_col_2_tab6,use_container_width=True)

                # create bar charts for 10th percentile (unigram)

                st.write(f'Top {top_n_uni_col_2_tab6} Frequently Occurring Unigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_uni_fig_col_2_tab6 = px.bar(top_n_grams(data_2_mod_bottom_sentences['sentence'],
                                   n=top_n_uni_col_2_tab6, ngram=(1,1), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_uni_fig_col_2_tab6.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Unigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_uni_fig_col_2_tab6,use_container_width=True)


                # slider (bigram)
                st.subheader('Top n Frequently Occurring Bigrams')
                top_n_bi_col_2_tab6 = st.slider('Select n', 1, 10, 6,key='bigram_col_2_tab6')

                # create bar charts for 90th percentile (Bigram)
                st.write(f'Top {top_n_bi_col_2_tab6} Frequently Occurring Bigrams - Sentiment Score: 90th percentile (top 10%)')
                
                top_bi_fig_col_2_tab6 = px.bar(top_n_grams(data_2_mod_top_sentences['sentence'],
                                   n=top_n_bi_col_2_tab6, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Reds",
                       )
                top_bi_fig_col_2_tab6.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_bi_fig_col_2_tab6,use_container_width=True)

                # create bar charts for 10th percentile (bigram)

                st.write(f'Top {top_n_bi_col_2_tab6} Frequently Occurring Bigrams - Sentiment Score: 10th percentile (bottom 10%)')
                
                bottom_bi_fig_col_2_tab6 = px.bar(top_n_grams(data_2_mod_bottom_sentences['sentence'],
                                   n=top_n_bi_col_2_tab6, ngram=(2,2), stop=stopwords),
                                     x='count',y='text',
                                     color='count',
                                     color_continuous_scale="Blues",
                       )
                bottom_bi_fig_col_2_tab6.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Bigram',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(bottom_bi_fig_col_2_tab6,use_container_width=True)


                # slider (Top n Occuring Sentiment Words)
                st.subheader('Top n Occuring Sentiment Words')
                top_n_sentiment_col_2_tab6 = st.slider('Select n', 1, 15, 12,key='sentiment_col_2_tab6')

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_2_tab6} Occuring Positive Sentiment Words')
                    
                top_pos_sentiment_fig_col_2_tab6 = px.bar(data_2_mod_pos_sentiment_words_df.head(top_n_sentiment_col_2_tab6),x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Reds")
                top_pos_sentiment_fig_col_2_tab6.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Positive Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_pos_sentiment_fig_col_2_tab6,use_container_width=True)

                # plot top mentioned sentiment words
                st.write(f'Top {top_n_sentiment_col_2_tab6} Occuring Negative Sentiment Words')
                    
                top_neg_sentiment_fig_col_2_tab6 = px.bar(data_2_mod_neg_sentiment_words_df.head(top_n_sentiment_col_2_tab6),x='count',y='sentiment_word', 
                                color='count',
                                color_continuous_scale="Blues")
                top_neg_sentiment_fig_col_2_tab6.update_layout(
                        autosize=False,
                        height=380,
                        yaxis_title='Negative Sentiment Words',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_neg_sentiment_fig_col_2_tab6,use_container_width=True)
                
                
                # slider (NPS of Top n Occuring Tourist Attractions)
                st.subheader('NPS of Top n Occuring Tourist Attractions')
                top_n_entities_col_2_tab6 = st.slider('Select n', 1, 20, 8,key='entities_col_2_tab6')
                # sort by count
                data_2_mod_unique_entities_count_df = data_2_mod_unique_entities_count_df.sort_values(by=['count','nps','entity'],
                                                                                ascending=False)
                
                # plot top mentioned entities using 'unique_entities_count_df'
                st.write(f'NPS of Top {top_n_entities_col_2_tab6} Occuring Tourist Attractions - Entity Level')
                
                top_entities_fig_col_2_tab6 = px.bar(data_2_mod_unique_entities_count_df.head(top_n_entities_col_2_tab6),x='nps',y='entity', 
                            color='interest_1',
                                           color_discrete_map={'transport':'#F8A19F',
                                                               'art':'#3283FE',
                                                               'museum':'#85660D',
                                                               'nature':'#00A08B',
                                                               'architecture':'#565656',
                                                               'park':'#1C8356',
                                                               'culture':'#FF9616',
                                                               'food and drinks':'#F7E1A0',
                                                               'religion':'#E2E2E2',
                                                               'shopping':'#00B5F7',
                                                               'recreation':'#C9FBE5',
                                                               'wildlife park':'#DEA0FD',
                                                               'amusement park':'#D62728',
                                                               'historic':'#325A9B',
                                                               'heritage':'#FEAF16',
                                                               'payment':'#AA0DFE'})
                top_entities_fig_col_2_tab6.update_layout(
                    autosize=False,
                    height=500,
                    yaxis_title='Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_entities_fig_col_2_tab6,use_container_width=True)

                # plot top mentioned entities grouped by type of tourist attraction 
                st.write(f'NPS of Top {top_n_entities_col_2_tab6} Occuring Tourist Attractions - Grouped by Type')
                
                top_entities_type_fig_col_2_tab6 = px.bar(data_2_mod_unique_entities_count_df.filter(['interest_1','count','total_count','nps']).head(top_n_entities_col_2_tab6).groupby('interest_1').mean().reset_index(),
                                                    x='nps',y='interest_1',
                                                    color='interest_1',
                                           color_discrete_map={'transport':'#F8A19F',
                                                               'art':'#3283FE',
                                                               'museum':'#85660D',
                                                               'nature':'#00A08B',
                                                               'architecture':'#565656',
                                                               'park':'#1C8356',
                                                               'culture':'#FF9616',
                                                               'food and drinks':'#F7E1A0',
                                                               'religion':'#E2E2E2',
                                                               'shopping':'#00B5F7',
                                                               'recreation':'#C9FBE5',
                                                               'wildlife park':'#DEA0FD',
                                                               'amusement park':'#D62728',
                                                               'historic':'#325A9B',
                                                               'heritage':'#FEAF16',
                                                               'payment':'#AA0DFE'})
                top_entities_type_fig_col_2_tab6.update_layout(
                    autosize=False,
                    height=300,
                    yaxis_title='Type of Tourist Attraction',
                    legend_title_text='Type of Tourist Attraction',
                        yaxis={'categoryorder': 'total ascending'},
            showlegend=False)
                st.plotly_chart(top_entities_type_fig_col_2_tab6,use_container_width=True)
