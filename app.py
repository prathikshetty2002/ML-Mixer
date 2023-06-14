import pandas as pd
import numpy as np
import streamlit as st 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from goose3 import Goose
import plotly.express as px 
import matplotlib.pyplot as plt
import dotenv
from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv('HUGGINGFACE_API'))


st.title('Sentiment Analysis Tool')

st.subheader('Vader Sentiment')

text = st.text_input('Enter a comment')
click=st.button('Compute')

print(text)

def senti(text):
    obj=SentimentIntensityAnalyzer()
    senti_dict=obj.polarity_scores(text)
    print(senti_dict)
    if senti_dict['compound']>=0.05:
        st.write("😁 Positive")
    elif senti_dict['compound']<=-0.05:
        st.write("😥 Negative")
    else:
        st.write("🙂 Neutral")
        
if click:
    senti(text)

st.header("Hate speech classification")

import requests

API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
headers = {"Authorization": "Bearer " + os.getenv('HUGGINGFACE_API')}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

sentence = st.text_input('Enter your text')

def senti_class(sentence):
    print(sentence)
    output=query({"inputs":str(sentence)})
    result={}
    if sentence:
        for data in output:
            print(data)
    return data
output=senti_class(sentence)

st.write(output)
# output = query({
# 	"inputs": "I hate you. I dont love you",
# })


