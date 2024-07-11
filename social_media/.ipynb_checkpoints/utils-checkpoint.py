from urllib.parse import urlparse
import pandas as pd
import re
import pickle
import contractions
from django.conf import settings
import os
# import builtins
from .text_processing import tokenize_and_lemmatize
# from nltk import WordNetLemmatizer


# def tokenize_and_lemmatize(text):
#     lemmatizer = WordNetLemmatizer()
#     tokens = text.split()
#     lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
#     return lemmatized_tokens


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


def extraversion_introversion(text):
    ## LOADING MODELS, VECTORS & ENCODERS ------

    EI0 = pickle.load(open(model_path('EI_model0.sav'), 'rb'))
    EI1 = pickle.load(open(model_path('EI_model1.sav'), 'rb'))
    EI2 = pickle.load(open(model_path('EI_model2.sav'), 'rb'))
    EI3 = pickle.load(open(model_path('EI_model3.sav'), 'rb'))
    EI4 = pickle.load(open(model_path('EI_model4.sav'), 'rb'))

    EI_enc = pickle.load(open(model_path('EI_encoder.obj'), 'rb'))
    # EI_vec = pickle.load(open(model_path('EI_vectors.pickle'), 'rb'))
    opn_vec = pickle.load(open(model_path('openness_vectors.pickle'), 'rb'))
    tokenizer = tokenize_and_lemmatize

    results = [EI_enc.inverse_transform(EI0.predict(opn_vec.transform(text).toarray()))[0],
               EI_enc.inverse_transform(EI1.predict(opn_vec.transform(text).toarray()))[0],
               EI_enc.inverse_transform(EI2.predict(opn_vec.transform(text).toarray()))[0],
               EI_enc.inverse_transform(EI3.predict(opn_vec.transform(text).toarray()))[0],
               EI_enc.inverse_transform(EI4.predict(opn_vec.transform(text).toarray()))[0]
               ]

    probability_e = results.count('E') / len(results)
    probability_i = results.count('I') / len(results)

    if probability_e > probability_i:
        probability = probability_e
        attribute = 'E'
    else:
        probability = probability_i
        attribute = 'I'
    return attribute, probability


def intuition_sensing(text):
    opn_vec = pickle.load(open(model_path('openness_vectors.pickle'), 'rb'))

    NS0 = pickle.load(open(model_path('NS_model0.sav'), 'rb'))
    NS1 = pickle.load(open(model_path('NS_model1.sav'), 'rb'))
    NS2 = pickle.load(open(model_path('NS_model2.sav'), 'rb'))
    NS3 = pickle.load(open(model_path('NS_model3.sav'), 'rb'))
    NS4 = pickle.load(open(model_path('NS_model3.sav'), 'rb'))

    # NS_vec = pickle.load(open(model_path('NS_vectors.pickle'), 'rb'))
    NS_enc = pickle.load(open(model_path('NS_encoder.obj'), 'rb'))

    results = [NS_enc.inverse_transform(NS0.predict(opn_vec.transform(text).toarray()))[0],
               NS_enc.inverse_transform(NS1.predict(opn_vec.transform(text).toarray()))[0],
               NS_enc.inverse_transform(NS2.predict(opn_vec.transform(text).toarray()))[0],
               NS_enc.inverse_transform(NS3.predict(opn_vec.transform(text).toarray()))[0],
               NS_enc.inverse_transform(NS4.predict(opn_vec.transform(text).toarray()))[0]
               ]

    probability_n = results.count('N') / len(results)
    probability_s = results.count('S') / len(results)

    if probability_n > probability_s:
        probability = probability_n
        attribute = 'N'
    else:
        probability = probability_s
        attribute = 'S'
    return attribute, probability


def feeling_thinking(text):
    FT_enc = pickle.load(open(model_path('FT_encoder.obj'), 'rb'))
    opn_vec = pickle.load(open(model_path('openness_vectors.pickle'), 'rb'))

    FT0 = pickle.load(open(model_path('FT_model0.sav'), 'rb'))
    FT1 = pickle.load(open(model_path('FT_model1.sav'), 'rb'))
    FT2 = pickle.load(open(model_path('FT_model2.sav'), 'rb'))
    FT3 = pickle.load(open(model_path('FT_model3.sav'), 'rb'))
    FT4 = pickle.load(open(model_path('FT_model4.sav'), 'rb'))
    # FT_vec = pickle.load(open(model_path('FT_vectors.pickle'), 'rb'))

    results = [FT_enc.inverse_transform(FT0.predict(opn_vec.transform(text).toarray()))[0],
               FT_enc.inverse_transform(FT1.predict(opn_vec.transform(text).toarray()))[0],
               FT_enc.inverse_transform(FT2.predict(opn_vec.transform(text).toarray()))[0],
               FT_enc.inverse_transform(FT3.predict(opn_vec.transform(text).toarray()))[0],
               FT_enc.inverse_transform(FT4.predict(opn_vec.transform(text).toarray()))[0]
               ]

    probability_f = results.count('F') / len(results)
    probability_t = results.count('T') / len(results)

    if probability_f > probability_t:
        probability = probability_f
        attribute = 'F'
    else:
        probability = probability_t
        attribute = 'T'
    return attribute, probability


def judging_perception(text):
    JP0 = pickle.load(open(model_path('JP_model0.sav'), 'rb'))
    JP1 = pickle.load(open(model_path('JP_model1.sav'), 'rb'))
    JP2 = pickle.load(open(model_path('JP_model2.sav'), 'rb'))
    JP3 = pickle.load(open(model_path('JP_model3.sav'), 'rb'))
    JP4 = pickle.load(open(model_path('JP_model4.sav'), 'rb'))

    JP_enc = pickle.load(open(model_path('JP_encoder.obj'), 'rb'))
    # JP_vec = pickle.load(open(model_path('JP_vectors.pickle'), 'rb'))

    opn_vec = pickle.load(open(model_path('openness_vectors.pickle'), 'rb'))

    results = [JP_enc.inverse_transform(JP0.predict(opn_vec.transform(text).toarray()))[0],
               JP_enc.inverse_transform(JP1.predict(opn_vec.transform(text).toarray()))[0],
               JP_enc.inverse_transform(JP2.predict(opn_vec.transform(text).toarray()))[0],
               JP_enc.inverse_transform(JP3.predict(opn_vec.transform(text).toarray()))[0],
               JP_enc.inverse_transform(JP4.predict(opn_vec.transform(text).toarray()))[0]
               ]

    probability_j = results.count('J') / len(results)
    probability_p = results.count('P') / len(results)

    if probability_j > probability_p:
        probability = probability_j
        attribute = 'J'
    else:
        probability = probability_p
        attribute = 'P'
    return attribute,


def openness_probability(text):
    omod0 = pickle.load(open(model_path('openness_model0.sav'), 'rb'))
    omod1 = pickle.load(open(model_path('openness_model1.sav'), 'rb'))
    omod2 = pickle.load(open(model_path('openness_model2.sav'), 'rb'))
    omod3 = pickle.load(open(model_path('openness_model3.sav'), 'rb'))
    omod4 = pickle.load(open(model_path('openness_model4.sav'), 'rb'))

    opn_vec = pickle.load(open(model_path('openness_vectors.pickle'), 'rb'))
    opn_enc = pickle.load(open(model_path('openness_encoder.obj'), 'rb'))

    results = [opn_enc.inverse_transform(omod0.predict(opn_vec.transform(text).toarray()))[0],
               opn_enc.inverse_transform(omod1.predict(opn_vec.transform(text).toarray()))[0],
               opn_enc.inverse_transform(omod2.predict(opn_vec.transform(text).toarray()))[0],
               opn_enc.inverse_transform(omod3.predict(opn_vec.transform(text).toarray()))[0],
               opn_enc.inverse_transform(omod4.predict(opn_vec.transform(text).toarray()))[0]
               ]

    probability = results.count('y') / len(results)
    return probability


def conscientiousness_probability(text):
    cmod0 = pickle.load(open(model_path('conscientiousness_model0.sav'), 'rb'))
    cmod1 = pickle.load(open(model_path('conscientiousness_model1.sav'), 'rb'))
    cmod2 = pickle.load(open(model_path('conscientiousness_model2.sav'), 'rb'))
    cmod3 = pickle.load(open(model_path('conscientiousness_model3.sav'), 'rb'))
    cmod4 = pickle.load(open(model_path('conscientiousness_model4.sav'), 'rb'))

    con_vec = pickle.load(open(model_path('conscientiousness_vectors.pickle'), 'rb'))
    con_enc = pickle.load(open(model_path('conscientiousness_encoder.obj'), 'rb'))

    results = [con_enc.inverse_transform(cmod0.predict(con_vec.transform(text).toarray()))[0],
               con_enc.inverse_transform(cmod1.predict(con_vec.transform(text).toarray()))[0],
               con_enc.inverse_transform(cmod2.predict(con_vec.transform(text).toarray()))[0],
               con_enc.inverse_transform(cmod3.predict(con_vec.transform(text).toarray()))[0],
               con_enc.inverse_transform(cmod4.predict(con_vec.transform(text).toarray()))[0]
               ]

    probability = results.count('y') / len(results)
    return probability


def extraversion_probability(text):
    emod0 = pickle.load(open(model_path('extraversion_model0.sav'), 'rb'))
    emod1 = pickle.load(open(model_path('extraversion_model1.sav'), 'rb'))
    emod2 = pickle.load(open(model_path('extraversion_model2.sav'), 'rb'))
    emod3 = pickle.load(open(model_path('extraversion_model3.sav'), 'rb'))
    emod4 = pickle.load(open(model_path('extraversion_model4.sav'), 'rb'))

    ext_vec = pickle.load(open(model_path('extraversion_vectors.pickle'), 'rb'))
    ext_enc = pickle.load(open(model_path('extraversion_encoder.obj'), 'rb'))

    results = [ext_enc.inverse_transform(emod0.predict(ext_vec.transform(text).toarray()))[0],
               ext_enc.inverse_transform(emod1.predict(ext_vec.transform(text).toarray()))[0],
               ext_enc.inverse_transform(emod2.predict(ext_vec.transform(text).toarray()))[0],
               ext_enc.inverse_transform(emod3.predict(ext_vec.transform(text).toarray()))[0],
               ext_enc.inverse_transform(emod4.predict(ext_vec.transform(text).toarray()))[0]
               ]

    probability = results.count('y') / len(results)
    return probability


def agreeableness_probability(text):
    amod0 = pickle.load(open(model_path('agreeableness_model0.sav'), 'rb'))
    amod1 = pickle.load(open(model_path('agreeableness_model1.sav'), 'rb'))
    amod2 = pickle.load(open(model_path('agreeableness_model2.sav'), 'rb'))
    amod3 = pickle.load(open(model_path('agreeableness_model3.sav'), 'rb'))
    amod4 = pickle.load(open(model_path('agreeableness_model4.sav'), 'rb'))

    agr_vec = pickle.load(open(model_path('agreeableness_vectors.pickle'), 'rb'))
    agr_enc = pickle.load(open(model_path('agreeableness_encoder.obj'), 'rb'))

    results = [agr_enc.inverse_transform(amod0.predict(agr_vec.transform(text).toarray()))[0],
               agr_enc.inverse_transform(amod1.predict(agr_vec.transform(text).toarray()))[0],
               agr_enc.inverse_transform(amod2.predict(agr_vec.transform(text).toarray()))[0],
               agr_enc.inverse_transform(amod3.predict(agr_vec.transform(text).toarray()))[0],
               agr_enc.inverse_transform(amod4.predict(agr_vec.transform(text).toarray()))[0]
               ]

    probability = results.count('y') / len(results)
    return probability


def neuroticism_probability(text):
    nmod0 = pickle.load(open(model_path('neuroticism_model0.sav'), 'rb'))
    nmod1 = pickle.load(open(model_path('neuroticism_model1.sav'), 'rb'))
    nmod2 = pickle.load(open(model_path('neuroticism_model2.sav'), 'rb'))
    nmod3 = pickle.load(open(model_path('neuroticism_model3.sav'), 'rb'))
    nmod4 = pickle.load(open(model_path('neuroticism_model4.sav'), 'rb'))

    neu_vec = pickle.load(open(model_path('neuroticism_vectors.pickle'), 'rb'))
    neu_enc = pickle.load(open(model_path('neuroticism_encoder.obj'), 'rb'))

    results = [neu_enc.inverse_transform(nmod0.predict(neu_vec.transform(text).toarray()))[0],
               neu_enc.inverse_transform(nmod1.predict(neu_vec.transform(text).toarray()))[0],
               neu_enc.inverse_transform(nmod2.predict(neu_vec.transform(text).toarray()))[0],
               neu_enc.inverse_transform(nmod3.predict(neu_vec.transform(text).toarray()))[0],
               neu_enc.inverse_transform(nmod4.predict(neu_vec.transform(text).toarray()))[0]
               ]

    probability = results.count('y') / len(results)
    return probability


def generate_mbti_results(data_list):
    ei, ei_ratio = extraversion_introversion(data_list)
    ns, ns_ratio = intuition_sensing(data_list)
    ft, ft_ratio = feeling_thinking(data_list)
    jp, jp_ratio = judging_perception(data_list)
    mbti = [ei, ns, ft, jp]
    mbti_pers = ''.join(mbti)
    result = (f'Likely MBTI personality is {mbti_pers}. {ei} = {ei_ratio:.0%}, {ns} = {ns_ratio:.0%}, '
              f'{ft} = {ft_ratio:.0%}, & {jp} = {jp_ratio:.0%}.')
    data = {'Character': ['Introversion', 'Sensation', 'Thinking', 'Perception'],
            'Value': [ei_ratio, ns_ratio, ft_ratio, jp_ratio]}
    df = pd.DataFrame(data)
    return result, df


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


def generate_data(info_list, df):
    raw_data = [''.join(str(word) for word in info_list)]
    df = pd.DataFrame(info_list, columns=["posts"])
    data = fix_contractions(df)
    df.posts = clear_text(data)
    texts = df.posts.to_list()
    cleaned_data = [''.join(str(text) for text in texts)]
    return raw_data, cleaned_data


def get_tweet_from_url(url):
    tweet = "Leonel Messi meilleur joueur de tous les temps"
    twitter_df = pd.DataFrame()
    source = [tweet]
    raw_data, new_data = generate_data(source, twitter_df)
    mbti_results, dfmbti = generate_mbti_results(new_data)
    big5_results, dfbig5 = generate_big5_results(new_data)

    return mbti_results
