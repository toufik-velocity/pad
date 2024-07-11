""" Helpers """

from urllib.parse import urlparse
import pandas as pd
import re
import pickle
import av
import shutil
import contractions
import matplotlib.pyplot as plt
from django.conf import settings
import time
import datetime
import instaloader
import facebook_scraper as fs
import torch
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import yt_dlp
import os
from bs4 import BeautifulSoup
from urllib.request import urlopen
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import Model , load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
from PIL import Image
import snscrape.modules.twitter as sntwitter
from transformers import AutoImageProcessor, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import pandas as pd
import urllib
import string


# Function to split types in MBTI dataset
def divide_mbti_types(df):
    df["EI"] = df['type'].astype(str).str[0]
    df["NS"] = df['type'].astype(str).str[1]
    df["FT"] = df['type'].astype(str).str[2]
    df["JP"] = df['type'].astype(str).str[3]
    return df

# Tokenizer
def tokenize_and_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Function to create and save a graph from datasets in a analysis
def save_graph(mbti_df, ocean_df):
    fig, ax = plt.subplots()
    mbti_df.plot(kind='bar', x='Character', y='Value', ax=ax)
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    plt.xlabel('Character')
    plt.ylabel('Values')
    plt.title('MBTI')
    app_static_dir = os.path.join(settings.BASE_DIR, 'social_media', 'static')
    plot_file_mbti = os.path.join(app_static_dir, 'mbti.png')
    plt.savefig(plot_file_mbti)
    fig, ax = plt.subplots()
    ocean_df.plot(kind='bar', x='Character', y='Value', ax=ax)
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    plt.xlabel('Character')
    plt.ylabel('Values')
    plt.title('OCEAN')
    plot_file_ocean = os.path.join(app_static_dir, 'ocean.png')
    plt.savefig(plot_file_ocean)
    return plot_file_mbti, plot_file_ocean

# Function to create and save a graph for the summary tab
def save_summary_graph(mbti, ocean, platform):
    # Given string
    ocean_string = str(ocean)

    # Regular expression pattern to extract dimension values
    pattern = r"(\w+)\s=\s(\d+)%"

    # Extract dimension values using regex
    dimension_values = re.findall(pattern, ocean_string)

    # Store dimension values in separate variables
    # OCEAN dimensions from the given string
    openness = 0
    conscientiousness = 0
    extraversion = 0
    agreeableness = 0
    neuroticism = 0

    # Iterate over extracted values and assign them to variables
    for dimension, value in dimension_values:
        if dimension == 'Openness':
            openness = int(value)
        elif dimension == 'Conscientiousness':
            conscientiousness = int(value)
        elif dimension == 'Extraversion':
            extraversion = int(value)
        elif dimension == 'Agreeableness':
            agreeableness = int(value)
        elif dimension == 'Neuroticism':
            neuroticism = int(value)

    # Bar plot
    dimensions = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    scores = [openness, conscientiousness, extraversion, agreeableness, neuroticism]

    data = {'Dimensions': dimensions,
            'Scores': scores}
    ocean_df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ocean_df.plot(kind='bar', x='Dimensions', y='Scores', ax=ax)
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    plt.xlabel('Character')
    plt.ylabel('Values')
    plt.title('OCEAN')

    # Save the bar plot as an image
    app_static_dir = os.path.join(settings.BASE_DIR, 'social_media', 'static')
    plot_file_ocean = os.path.join(app_static_dir, platform + 'ocean.png')
    plt.savefig(plot_file_ocean)

    # # Clear the current plot
    # plt.clf()

    # Given string
    mbti_string = str(mbti)

    # Regular expression pattern to extract dimension values
    pattern = r"([A-Z]) = (\d+)"

    # Extract dimension values using regex
    dimension_values = re.findall(pattern, mbti_string)

    # Store dimension values in separate variables
    introversion = 0
    extraversion = ""
    sensing = 0
    intuition = ""
    feeling = 0
    thinking = ""
    judging = 0
    perception = ""

    # Iterate over extracted values and assign them to variables
    for dimension, value in dimension_values:
        if dimension == 'I' or dimension == "E":
            extraversion = "Introversion (I)" if dimension == "I" else "Extraversion (E)"
            introversion = int(value)
        elif dimension == 'S' or dimension == "N":
            intuition = "Sensing (S)" if dimension == "S" else "Intuition (N)"
            sensing = int(value)
        elif dimension == 'T' or dimension == "F":
            feeling = "Thinking (T)" if dimension == "T" else "Feeling (F)"
            thinking = int(value)
        elif dimension == 'J' or dimension == "P":
            perception = "Judging (J)" if dimension == "J" else "Perception (P)"
            judging = int(value)

    # Bar plot
    dimensions = [extraversion, intuition, feeling, perception]
    scores = [introversion, sensing, thinking, judging]

    data = {'Dimensions': dimensions,
            'Scores': scores}
    mbti_df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    mbti_df.plot(kind='bar', x='Dimensions', y='Scores', ax=ax)
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    plt.xlabel('Character')
    plt.ylabel('Values')
    plt.title('MBTI')
    # Save the bar plot as an image
    app_static_dir = os.path.join(settings.BASE_DIR, 'social_media', 'static')
    plot_file_mbti = os.path.join(app_static_dir, platform + 'mbti.png')
    plt.savefig(plot_file_mbti)
    # Clear the current plot
    plt.clf()
    return plot_file_mbti, plot_file_ocean


def model_path(model):
    model_path = os.path.join(settings.MEDIA_ROOT, model)
    return model_path


# Function to remove contractions
def fix_contractions(df, column_name="posts"):
    df[column_name] = df[column_name].apply(lambda x: contractions.fix(x))
    return df


# Function to remove links and symbols
def clear_text(data):
    cleaned_text = []
    for sentence in data.posts:
        sentence = sentence.lower()

        #       removing links from text data
        sentence = re.sub(r'http[s]?://\S+', '', sentence)

        #       removing other symbols
        sentence = re.sub(r'@([a-zA-Z0-9_]{1,50})', '', sentence)
        sentence = re.sub(r'#([a-zA-Z0-9_]{1,50})', '', sentence)
        sentence = re.sub(r'[^A-Za-z]+', ' ', sentence)
        sentence = " ".join([word for word in sentence.split() if not len(word) < 3])

        cleaned_text.append(sentence)
    return cleaned_text


# Get predominant attribute (extraversion , introversion)
def extraversion_introversion(text):
    # LOADING MODELS, VECTORS & ENCODERS ------
    EI0 = pickle.load(open(model_path('EI_model0.sav'), 'rb'))
    EI1 = pickle.load(open(model_path('EI_model1.sav'), 'rb'))
    EI2 = pickle.load(open(model_path('EI_model2.sav'), 'rb'))
    EI3 = pickle.load(open(model_path('EI_model3.sav'), 'rb'))
    EI4 = pickle.load(open(model_path('EI_model4.sav'), 'rb'))

    EI_enc = pickle.load(open(model_path('EI_encoder.obj'), 'rb'))
    opn_vec = pickle.load(open(model_path('openness_vectors.pickle'), 'rb'))

    results = []
    for model in [EI0, EI1, EI2, EI3, EI4]:
        try:
            start_time = time.time()
            predictions = model.predict(opn_vec.transform(text).toarray())
            inverse_transform_time = time.time() - start_time

            # Add some debug information
            print(f"Inverse transform time: {inverse_transform_time}")

            attribute = EI_enc.inverse_transform(predictions)[0]
            results.append(attribute)
        except Exception as e:
            print(f"Error occurred: {e}")

        # Free memory by deleting the model object
        del model

    probability_e = results.count('E') / len(results)
    probability_i = results.count('I') / len(results)

    if probability_e > probability_i:
        probability = probability_e
        attribute = 'E'
    else:
        probability = probability_i
        attribute = 'I'

    # Free memory by deleting the remaining variables
    del EI0, EI1, EI2, EI3, EI4, EI_enc, opn_vec, predictions, results

    return attribute, probability


# Get predominant attribute (intuition sensing)
def intuition_sensing(text):
    opn_vec = pickle.load(open(model_path('openness_vectors.pickle'), 'rb'))

    NS0 = pickle.load(open(model_path('NS_model0.sav'), 'rb'))
    NS1 = pickle.load(open(model_path('NS_model1.sav'), 'rb'))
    NS2 = pickle.load(open(model_path('NS_model2.sav'), 'rb'))
    NS3 = pickle.load(open(model_path('NS_model3.sav'), 'rb'))
    NS4 = pickle.load(open(model_path('NS_model3.sav'), 'rb'))

    NS_enc = pickle.load(open(model_path('NS_encoder.obj'), 'rb'))

    results = []
    for model in [NS0, NS1, NS2, NS3, NS4]:
        try:
            start_time = time.time()
            predictions = model.predict(opn_vec.transform(text).toarray())
            inverse_transform_time = time.time() - start_time

            # Add some debug information
            print(f"Inverse transform time: {inverse_transform_time}")

            attribute = NS_enc.inverse_transform(predictions)[0]
            results.append(attribute)
        except Exception as e:
            print(f"Error occurred: {e}")

        # Free memory by deleting the model object
        del model

    probability_n = results.count('N') / len(results)
    probability_s = results.count('S') / len(results)

    if probability_n > probability_s:
        probability = probability_n
        attribute = 'N'
    else:
        probability = probability_s
        attribute = 'S'

    # Free memory by deleting the remaining variables
    del NS0, NS1, NS2, NS3, NS4, NS_enc, opn_vec, predictions, results

    return attribute, probability


# Get predominant attribute (feeling thinking)
def feeling_thinking(text):
    FT_enc = pickle.load(open(model_path('FT_encoder.obj'), 'rb'))
    opn_vec = pickle.load(open(model_path('openness_vectors.pickle'), 'rb'))

    FT0 = pickle.load(open(model_path('FT_model0.sav'), 'rb'))
    FT1 = pickle.load(open(model_path('FT_model1.sav'), 'rb'))
    FT2 = pickle.load(open(model_path('FT_model2.sav'), 'rb'))
    FT3 = pickle.load(open(model_path('FT_model3.sav'), 'rb'))
    FT4 = pickle.load(open(model_path('FT_model4.sav'), 'rb'))

    results = []
    for model in [FT0, FT1, FT2, FT3, FT4]:
        try:
            start_time = time.time()
            predictions = model.predict(opn_vec.transform(text).toarray())
            inverse_transform_time = time.time() - start_time

            # Add some debug information
            print(f"Inverse transform time: {inverse_transform_time}")

            attribute = FT_enc.inverse_transform(predictions)[0]
            results.append(attribute)
        except Exception as e:
            print(f"Error occurred: {e}")

        # Free memory by deleting the model object
        del model

    probability_f = results.count('F') / len(results)
    probability_t = results.count('T') / len(results)

    if probability_f > probability_t:
        probability = probability_f
        attribute = 'F'
    else:
        probability = probability_t

    # Free memory by deleting the remaining variables
    del FT0, FT1, FT2, FT3, FT4, FT_enc, opn_vec, predictions, results

    return attribute, probability


# Get predominant attribute (judging perception)
def judging_perception(text):
    JP0 = pickle.load(open(model_path('JP_model0.sav'), 'rb'))
    JP1 = pickle.load(open(model_path('JP_model1.sav'), 'rb'))
    JP2 = pickle.load(open(model_path('JP_model2.sav'), 'rb'))
    JP3 = pickle.load(open(model_path('JP_model3.sav'), 'rb'))
    JP4 = pickle.load(open(model_path('JP_model4.sav'), 'rb'))

    JP_enc = pickle.load(open(model_path('JP_encoder.obj'), 'rb'))
    opn_vec = pickle.load(open(model_path('openness_vectors.pickle'), 'rb'))

    results = []
    for model in [JP0, JP1, JP2, JP3, JP4]:
        try:
            start_time = time.time()
            predictions = model.predict(opn_vec.transform(text).toarray())
            inverse_transform_time = time.time() - start_time

            # Add some debug information
            print(f"Inverse transform time: {inverse_transform_time}")

            attribute = JP_enc.inverse_transform(predictions)[0]
            results.append(attribute)
        except Exception as e:
            print(f"Error occurred: {e}")

        # Free memory by deleting the model object
        del model

    probability_j = results.count('J') / len(results)
    probability_p = results.count('P') / len(results)

    if probability_j > probability_p:
        probability = probability_j
        attribute = 'J'
    else:
        probability = probability_p

    # Free memory by deleting the remaining variables
    del JP0, JP1, JP2, JP3, JP4, JP_enc, opn_vec, predictions, results

    return attribute, probability


# Get predominant attribute (openness)
def openness_probability(text):
    omod0 = pickle.load(open(model_path('openness_model0.sav'), 'rb'))
    omod1 = pickle.load(open(model_path('openness_model1.sav'), 'rb'))
    omod2 = pickle.load(open(model_path('openness_model2.sav'), 'rb'))
    omod3 = pickle.load(open(model_path('openness_model3.sav'), 'rb'))
    omod4 = pickle.load(open(model_path('openness_model4.sav'), 'rb'))

    opn_vec = pickle.load(open(model_path('openness_vectors.pickle'), 'rb'))
    opn_enc = pickle.load(open(model_path('openness_encoder.obj'), 'rb'))

    results = []
    for model in [omod0, omod1, omod2, omod3, omod4]:
        try:
            start_time = time.time()
            predictions = model.predict(opn_vec.transform(text).toarray())
            inverse_transform_time = time.time() - start_time

            # Add some debug information
            print(f"Inverse transform time: {inverse_transform_time}")

            attribute = opn_enc.inverse_transform(predictions)[0]
            results.append(attribute)
        except Exception as e:
            print(f"Error occurred: {e}")

        # Free memory by deleting the model object
        del model

    probability = results.count('y') / len(results)

    # Free memory by deleting the remaining variables
    del omod0, omod1, omod2, omod3, omod4, opn_enc, opn_vec, predictions, results

    return probability


# Get predominant attribute (conscientiousness)
def conscientiousness_probability(text):
    cmod0 = pickle.load(open(model_path('conscientiousness_model0.sav'), 'rb'))
    cmod1 = pickle.load(open(model_path('conscientiousness_model1.sav'), 'rb'))
    cmod2 = pickle.load(open(model_path('conscientiousness_model2.sav'), 'rb'))
    cmod3 = pickle.load(open(model_path('conscientiousness_model3.sav'), 'rb'))
    cmod4 = pickle.load(open(model_path('conscientiousness_model4.sav'), 'rb'))

    con_vec = pickle.load(open(model_path('conscientiousness_vectors.pickle'), 'rb'))
    con_enc = pickle.load(open(model_path('conscientiousness_encoder.obj'), 'rb'))

    results = []
    for model in [cmod0, cmod1, cmod2, cmod3, cmod4]:
        try:
            start_time = time.time()
            predictions = model.predict(con_vec.transform(text).toarray())
            inverse_transform_time = time.time() - start_time

            # Add some debug information
            print(f"Inverse transform time: {inverse_transform_time}")

            attribute = con_enc.inverse_transform(predictions)[0]
            results.append(attribute)
        except Exception as e:
            print(f"Error occurred: {e}")

        # Free memory by deleting the model object
        del model

    probability = results.count('y') / len(results)

    # Free memory by deleting the remaining variables
    del cmod0, cmod1, cmod2, cmod3, cmod4, con_enc, con_vec, predictions, results

    return probability


# Get predominant attribute (extraversion)
def extraversion_probability(text):
    emod0 = pickle.load(open(model_path('extraversion_model0.sav'), 'rb'))
    emod1 = pickle.load(open(model_path('extraversion_model1.sav'), 'rb'))
    emod2 = pickle.load(open(model_path('extraversion_model2.sav'), 'rb'))
    emod3 = pickle.load(open(model_path('extraversion_model3.sav'), 'rb'))
    emod4 = pickle.load(open(model_path('extraversion_model4.sav'), 'rb'))

    ext_vec = pickle.load(open(model_path('extraversion_vectors.pickle'), 'rb'))
    ext_enc = pickle.load(open(model_path('extraversion_encoder.obj'), 'rb'))

    results = []
    for model in [emod0, emod1, emod2, emod3, emod4]:
        try:
            start_time = time.time()
            predictions = model.predict(ext_vec.transform(text).toarray())
            inverse_transform_time = time.time() - start_time

            # Add some debug information
            print(f"Inverse transform time: {inverse_transform_time}")

            attribute = ext_enc.inverse_transform(predictions)[0]
            results.append(attribute)
        except Exception as e:
            print(f"Error occurred: {e}")

        # Free memory by deleting the model object
        del model

    probability = results.count('y') / len(results)

    # Free memory by deleting the remaining variables
    del emod0, emod1, emod2, emod3, emod4, ext_enc, ext_vec, predictions, results

    return probability


# Get predominant attribute (agreeableness)
def agreeableness_probability(text):
    amod0 = pickle.load(open(model_path('agreeableness_model0.sav'), 'rb'))
    amod1 = pickle.load(open(model_path('agreeableness_model1.sav'), 'rb'))
    amod2 = pickle.load(open(model_path('agreeableness_model2.sav'), 'rb'))
    amod3 = pickle.load(open(model_path('agreeableness_model3.sav'), 'rb'))
    amod4 = pickle.load(open(model_path('agreeableness_model4.sav'), 'rb'))

    agr_vec = pickle.load(open(model_path('agreeableness_vectors.pickle'), 'rb'))
    agr_enc = pickle.load(open(model_path('agreeableness_encoder.obj'), 'rb'))

    results = []
    for model in [amod0, amod1, amod2, amod3, amod4]:
        try:
            start_time = time.time()
            predictions = model.predict(agr_vec.transform(text).toarray())
            inverse_transform_time = time.time() - start_time

            # Add some debug information
            print(f"Inverse transform time: {inverse_transform_time}")

            attribute = agr_enc.inverse_transform(predictions)[0]
            results.append(attribute)
        except Exception as e:
            print(f"Error occurred: {e}")

        # Free memory by deleting the model object
        del model

    probability = results.count('y') / len(results)

    # Free memory by deleting the remaining variables
    del amod0, amod1, amod2, amod3, amod4, agr_enc, agr_vec, predictions, results

    return probability


# Get predominant attribute (neuroticism)
def neuroticism_probability(text):
    nmod0 = pickle.load(open(model_path('neuroticism_model0.sav'), 'rb'))
    nmod1 = pickle.load(open(model_path('neuroticism_model1.sav'), 'rb'))
    nmod2 = pickle.load(open(model_path('neuroticism_model2.sav'), 'rb'))
    nmod3 = pickle.load(open(model_path('neuroticism_model3.sav'), 'rb'))
    nmod4 = pickle.load(open(model_path('neuroticism_model4.sav'), 'rb'))

    neu_vec = pickle.load(open(model_path('neuroticism_vectors.pickle'), 'rb'))
    neu_enc = pickle.load(open(model_path('neuroticism_encoder.obj'), 'rb'))

    results = []
    for model in [nmod0, nmod1, nmod2, nmod3, nmod4]:
        try:
            start_time = time.time()
            predictions = model.predict(neu_vec.transform(text).toarray())
            inverse_transform_time = time.time() - start_time

            # Add some debug information
            print(f"Inverse transform time: {inverse_transform_time}")

            attribute = neu_enc.inverse_transform(predictions)[0]
            results.append(attribute)
        except Exception as e:
            print(f"Error occurred: {e}")

        # Free memory by deleting the model object
        del model

    probability = results.count('y') / len(results)

    # Free memory by deleting the remaining variables
    del nmod0, nmod1, nmod2, nmod3, nmod4, neu_enc, neu_vec, predictions, results

    return probability


# Function to equalize minority and majority classes
def undersample_majority(df, cls_col):
    """ Function to equalize minority and majority classes"""

    counter = Counter(df[cls_col])
    ratio = min(counter.values()) / max(counter.values())
    # inv = (max(counter.values()) / min(counter.values()))
    # print(f'Minority Class: Majority Class Ratio = 1:{inv:.2f}')
    # Convert columns to arrays
    if ratio < 0.7:
        X = np.array(df['posts']).reshape(-1, 1)
        y = np.array(df[cls_col])
        # Undersample majority class to the minority class size
        under = RandomUnderSampler()
        X, y = under.fit_resample(X, y)
        print(Counter(y))
        df = pd.DataFrame({'posts': list(X.flatten()), cls_col: y}, columns=['posts', cls_col])
    else:
        df = df
    return df

# Function to generate mbti results
def generate_mbti_results(data_list):
    ei, ei_ratio = extraversion_introversion(data_list)
    ns, ns_ratio = intuition_sensing(data_list)
    ft, ft_ratio = feeling_thinking(data_list)
    jp, jp_ratio = judging_perception(data_list)
    mbti = [ei, ns, ft, jp]
    mbti_pers = ''.join(mbti)
    result = (f'Likely MBTI personality is {mbti_pers}. {ei} = {ei_ratio:.0%}, {ns} = {ns_ratio:.0%}, '
              f'{ft} = {ft_ratio:.0%}, & {jp} = {jp_ratio:.0%}.')
    if ei == "I":
        intro = "Introversion"
    else:
        intro = "Extraversion"
    if ns == "N":
        intuition = "Intuition"
    else:
        intuition = "Sensation"
    if ft == "F":
        feeling = "Feeling"
    else:
        feeling = "Thinking"
    if jp == "J":
        judging = "Judging"
    else:
        judging = "Perception"

    data = {'Character': [intro, intuition, feeling, judging],
            'Value': [ei_ratio, ns_ratio, ft_ratio, jp_ratio]}
    df = pd.DataFrame(data)
    return result, df


# Function to generate big5 results
def generate_big5_results(data_list):
    opn_ratio = openness_probability(data_list)
    con_ratio = conscientiousness_probability(data_list)
    ext_ratio = extraversion_probability(data_list)
    agr_ratio = agreeableness_probability(data_list)
    neu_ratio = neuroticism_probability(data_list)
    result = (f'OCEAN dimensions: Openness = {opn_ratio:.0%}, Conscientiousness = {con_ratio:.0%}, Extraversion = '
              f'{ext_ratio:.0%}, Agreeableness = {agr_ratio:.0%}, & Neuroticism = {neu_ratio:.0%}.')
    data = {
        'Character': ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'],
        'Value': [opn_ratio, con_ratio, ext_ratio, agr_ratio, neu_ratio]
    }
    df = pd.DataFrame(data)
    return result, df


#processing text
def generate_data(info_list, df):
    raw_data = [''.join(str(word) for word in info_list)]
    df = pd.DataFrame(info_list, columns=["posts"])
    data = fix_contractions(df)
    df.posts = clear_text(data)
    texts = df.posts.to_list()
    cleaned_data = [''.join(str(text) for text in texts)]
    return raw_data, cleaned_data


# video captionning
def hugging_face_extract_text_from_video(filepath):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load pretrained processor, tokenizer, and model
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

    # load video
    video_path = filepath
    container = av.open(video_path)

    # extract evenly spaced frames from video
    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))

    # generate caption
    gen_kwargs = {
        "min_length": 10,
        "max_length": 20,
        "num_beams": 8,
    }
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return caption


#helper for download video from url
def download_videos(url, directory):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filename = "video.mp4"  # Specify the filename for the downloaded video
        file_path = os.path.join(directory, filename)
        # Download the image
        urllib.request.urlretrieve(url, file_path)


# image captionning
def hugging_face_extract_text_from_image(filepath):
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def predict_step(image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)
        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

    return predict_step([filepath])[0]


# Generate captions for images
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# predict caption from and image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'sttseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text


# image captionning 2
# def local_extract_text_from_image(filepath):
#     # Test with images in the wild
#     vgg_model = VGG16()  # 'video_caption_model.h5'
#     # restructure the model
#     vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
#     image_path = filepath
#     # load image
#     image = load_img(image_path, target_size=(224, 224))
#     # convert image pixels to numpy array
#     image = img_to_array(image)
#     # reshape data for model
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     # preprocess image for vgg
#     image = preprocess_input(image)
#     # extract features
#     feature = vgg_model.predict(image, verbose=0)
#     # Load model from pickle
#     model = load_model(os.path.join(settings.MEDIA_ROOT, "image_caption_model.h5"))
#     # Load tokenizer from pickle
#     with open(os.path.join(settings.MEDIA_ROOT, 'tokenizer.pickle'), 'rb') as f:
#         tokenizer = pickle.load(f)
#     # predict from the trained model
#     result = predict_caption(model, feature, tokenizer, 35)
#     caption = result.replace('sttseq', '').replace('endseq', '')
#     return caption

# helper to download image from a url
def download_images(url, directory):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filename = "image.jpg"  # Specify the filename for the downloaded image
        file_path = os.path.join(directory, filename)
        # Download the image
        urllib.request.urlretrieve(url, file_path)



def get_website_text(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            return text
        else:
            print(f"Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


# get content text from a website by url
def get_text_from_website_url(url):
    return str(get_website_text(str(url)))


# extract a text from a youtube post url
def get_text_from_youtube_url(url, analysis):
    link = url
    script_directory = settings.MEDIA_ROOT
    now = datetime.datetime.now().timestamp()
    output_path = os.path.join(script_directory, analysis, f"{str(now)}.mp4")

    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
    final_text = "" + hugging_face_extract_text_from_video(output_path)

    # Retry the deletion with a delay between attempts
    folder_path = os.path.join(script_directory, analysis)
    max_retries = 5
    retry_delay = 1  # Adjust the delay as needed

    for attempt in range(max_retries):
        try:
            shutil.rmtree(folder_path)
            break  # Folder deleted successfully, exit the loop
        except PermissionError:
            print(f'Permission denied - folder in use by another process (attempt {attempt + 1}/{max_retries})')
            time.sleep(retry_delay)

    # Check if the folder was deleted
    if os.path.exists(folder_path):
        print('Failed to delete the folder.')

    return final_text


# extract a text from a tiktok post url
def get_text_from_tiktok_url(url, analysis):
    cookies = {
        #         '__cflb': '02DiuEcwseaiqqyPC5qrDCss6XuWcthYMtwDyGzoyRy7h',
    }

    headers = {
        'authority': 'ssstik.io',
        'accept': '*/*',
        'accept-language': 'fr,fr-FR;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        # 'cookie': '__cflb=02DiuEcwseaiqqyPC5qrDCss6XuWcthYMtwDyGzoyRy7h',
        'dnt': '1',
        'hx-current-url': 'https://ssstik.io/en',
        'hx-request': 'true',
        'hx-target': 'target',
        'hx-trigger': '_gcaptcha_pt',
        'origin': 'https://ssstik.io',
        'referer': 'https://ssstik.io/en',
        #         'sec-ch-ua': '"Microsoft Edge";v="111", "Not(A:Brand";v="8", "Chromium";v="111"',
        #         'sec-ch-ua-mobile': '?0',
        #         'sec-ch-ua-platform': '"Windows"',
        #         'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.54',
    }

    params = {
        'url': 'dl',
    }

    data = {
        'id': url,
        'locale': 'en',
        'tt': '',
    }

    #     print("STEP 4: Getting the download link")
    #     print("If this step fails, PLEASE read the steps above")
    response = requests.post('https://ssstik.io/abc', params=params, cookies=cookies, headers=headers, data=data)
    #     print(response)
    downloadSoup = BeautifulSoup(response.text, "html.parser")
    #     print(response.text)
    #     print(downloadSoup)
    downloadLink = downloadSoup.a["href"]
    # videoTitle = downloadSoup.p.getText().strip()
    script_directory = settings.MEDIA_ROOT
    os.makedirs(os.path.join(script_directory, analysis) , exist_ok=True)
    now = datetime.datetime.now().timestamp()
    location = os.path.join(script_directory, analysis, f"{str(now)}.mp4")

    #     print("STEP 5: Saving the video :)")
    mp4File = urlopen(downloadLink)
    # Feel free to change the download directory
    with open(f"" + location, "wb") as output:
        while True:
            data = mp4File.read(4096)
            if data:
                output.write(data)
            else:
                print("Done")
                break
    text = hugging_face_extract_text_from_video(location)
    # Retry the deletion with a delay between attempts
    folder_path = os.path.join(script_directory, analysis)
    max_retries = 5
    retry_delay = 1  # Adjust the delay as needed

    for attempt in range(max_retries):
        try:
            shutil.rmtree(folder_path)
            break  # Folder deleted successfully, exit the loop
        except PermissionError:
            print(f'Permission denied - folder in use by another process (attempt {attempt + 1}/{max_retries})')
            time.sleep(retry_delay)

    # Check if the folder was deleted
    if os.path.exists(folder_path):
        print('Failed to delete the folder.')
    return str(text)


# extract a text from a instagram post url
def get_text_from_instagram_url(url, analysis):
    L = instaloader.Instaloader()

    # Get the current date and time
    script_directory = settings.MEDIA_ROOT
    now = datetime.datetime.now().timestamp()

    # Download the post
    post = instaloader.Post.from_shortcode(L.context, url.split('/')[-2])
    L.download_post(post, target=str(now))

    final_text = ""
    for filename in os.listdir(str(now)):
        path = os.path.join(str(now), filename)
        if os.path.isfile(path):
            if '.jpg' in path:
                final_text += " " + hugging_face_extract_text_from_image(path)
            if '.mp4' in path:
                fileout = hugging_face_extract_text_from_video(path)
                final_text += " " + fileout
            if '.txt' in path:
                with open(path, errors="ignore") as mytxt:
                    for line in mytxt:
                        final_text += " " + line
    max_retries = 5
    retry_delay = 1  # Adjust the delay as needed

    for attempt in range(max_retries):
        try:
            shutil.rmtree(str(now))
            break  # Folder deleted successfully, exit the loop
        except PermissionError:
            print(f'Permission denied - folder in use by another process (attempt {attempt + 1}/{max_retries})')
            time.sleep(retry_delay)

    # Check if the folder was deleted
    if os.path.exists(str(now)):
        print('Failed to delete the folder.')
    return final_text


# extract a text from a facebook post url
def get_text_from_facebook_url(url , analysis):
    if re.match(r'^https?://(web\.)?facebook\.com', url):
        url = url.replace("web", "www", 1)
    # Scrape the post
    post = fs.get_posts(post_urls=[url], options={"comments": 5})

    # Create the output directory
    # Define the directories to store the raw messages, images, and videos
    script_directory = settings.MEDIA_ROOT
    image_directory = os.path.join(script_directory, str(analysis), 'facebook_images')
    os.makedirs(image_directory, exist_ok=True)

    extract_text = ""
    print(post)
    # Download and save the images
    for p in post:
        #     print(p)
        extract_text += " " + p["text"]
        # Check if the post has images
        if "image" in p:
            image_url = p["image_lowquality"]

    # Create a unique filename for the image
    image_filename = f"image_{p['post_id']}.jpeg"
    image_path = os.path.join(image_directory, image_filename)

    # Download the image
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(image_path, "wb") as f:
            f.write(response.content)
        print("Image saved:", image_path)
        extract_text += " " + hugging_face_extract_text_from_image(image_path)
        # Retry the deletion with a delay between attempts
        folder_path = os.path.join(script_directory, analysis)
        max_retries = 5
        retry_delay = 1  # Adjust the delay as needed

        for attempt in range(max_retries):
            try:
                shutil.rmtree(folder_path)
                break  # Folder deleted successfully, exit the loop
            except PermissionError:
                print(f'Permission denied - folder in use by another process (attempt {attempt + 1}/{max_retries})')
                time.sleep(retry_delay)

        # Check if the folder was deleted
        if os.path.exists(folder_path):
            print('Failed to delete the folder.')

    else:
        print("Failed to download image:", image_url)
    return extract_text


# extract a text from a twitter post url
def get_text_from_twitter_url(url, analysis):
    # Define the directories to store the raw messages, images, and videos
    script_directory = settings.MEDIA_ROOT
    # raw_message_directory = os.path.join(script_directory, str(analysis), 'twitter_raw_messages')
    image_directory = os.path.join(script_directory, str(analysis), 'twitter_images')
    video_directory = os.path.join(script_directory, str(analysis), 'twitter_videos')

    # Create the directories if they don't exist
    # os.makedirs(raw_message_directory, exist_ok=True)
    os.makedirs(image_directory, exist_ok=True)
    os.makedirs(video_directory, exist_ok=True)

    link = str(url)
    parsed_url = urlparse(link)
    path_segments = parsed_url.path.split("/")
    extract_text = ""
    # Extract the username from the URL
    username = path_segments[1]
    for tweet in sntwitter.TwitterProfileScraper(username).get_items():
        if tweet.url == link:
            extract_text += " " + tweet.rawContent
            media = tweet.media[0] if tweet.media is not None else None
            if isinstance(media, sntwitter.Photo):
                mediaUrl = media.fullUrl
                # Download the image
                download_images(mediaUrl, image_directory)
                extract_text += " " + str(
                    hugging_face_extract_text_from_image(os.path.join(image_directory, "image.jpg")))

            elif isinstance(media, sntwitter.Video):
                if len(media.variants) > 0:
                    video_variant = media.variants[0]
                    mediaUrl = video_variant.url
                    # Download the video
                    download_videos(mediaUrl, video_directory)
                    extract_text += " " + str(
                        hugging_face_extract_text_from_video(os.path.join(video_directory, "video.mp4")))

            break
    # Retry the deletion with a delay between attempts
    folder_path = os.path.join(script_directory, analysis)
    max_retries = 5
    retry_delay = 1  # Adjust the delay as needed

    for attempt in range(max_retries):
        try:
            shutil.rmtree(folder_path)
            break  # Folder deleted successfully, exit the loop
        except PermissionError:
            print(f'Permission denied - folder in use by another process (attempt {attempt + 1}/{max_retries})')
            time.sleep(retry_delay)

    # Check if the folder was deleted
    if os.path.exists(folder_path):
        print('Failed to delete the folder.')
    return extract_text


# Function to recognize the platform by the post url
def get_social_media_platform(url):
    if re.match(r'^https?://(www\.)?facebook\.com', url) or re.match(r'^https?://(web\.)?facebook\.com', url):
        return 'Facebook'
    elif re.match(r'^https?://(www\.)?instagram\.com', url):
        return 'Instagram'
    elif re.match(r'^https?://(www\.)?linkedin\.com', url):
        return 'LinkedIn'
    elif re.match(r'^https?://(www\.)?youtube\.com', url):
        return 'YouTube'
    elif re.match(r'^https?://(www\.)?twitter\.com', url):
        return 'Twitter'
    elif re.match(r'^https?://(www\.)?tiktok\.com', url):
        return 'TikTok'
    else:
        return 'Website'


# get results
def get_results(text):
    tweet = str(text)
    twitter_df = pd.DataFrame()
    source = [tweet]
    raw_data, new_data = generate_data(source, twitter_df)
    mbti_results, dfmbti = generate_mbti_results(new_data)
    big5_results, dfbig5 = generate_big5_results(new_data)

    return mbti_results, dfmbti, big5_results, dfbig5
