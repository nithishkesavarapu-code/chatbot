import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import requests
import urllib.parse
import random
import json
import pickle
import numpy as np
import rich
from datetime import datetime
from datetime import date

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

faqs = json.loads(open('FAQ.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)

    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w:
                bag[i]=1
    
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list

def get_response(intents_list,intents_json):

    if len(intents_list) == 0:
        return "Sorry, I didn’t understand that."

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['faq']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result

    return "Sorry, I didn’t understand that."
    

console = Console()

welcome_panel = Panel(
    "[bold cyan]Welcome to the Chatbot CLI![/bold cyan]\nType 'quit' or 'exit' to end the session.",
    title="[bold green]Chatbot Interface[/bold green]",
    border_style="green"
)
console.print(welcome_panel)

def get_location():
    try:
        response = requests.get("http://ip-api.com/json/").json()
        return response.get("city"), response.get("country")
    except:
        return None, None

def get_weather():
    city, country = get_location()
    if not city:
        return "Sorry, I couldn't detect your location."

    api_key = "6944ac1cd0d9985811a7e8d8af0ec026"
    city_encoded = urllib.parse.quote(city)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_encoded}&appid={api_key}&units=metric"
    response = requests.get(url).json()

    if response.get("main"):
        temp = response["main"]["temp"]
        desc = response["weather"][0]["description"]
        return f"The current weather in {city}, {country} is {temp}°C with {desc}."
    return "Sorry, I couldn't fetch the weather right now."


while True:
    message = (Prompt.ask("[bold yellow]You[/bold yellow]")).lower()
    if(message=="exit" or message=="quit"):
        break
    ints = predict_class(message)
    
    res = get_response(ints,faqs)

    if res == "__TIME__":
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        res = f"The current time is {current_time}."
    if res == "__DATE__":
        today = date.today()
        res = f"Today's date is {today}."
    if res == "__WEATHER__":
        res = get_weather()
    
    console.print(f"[bold blue]Bot:[/bold blue] {res}")
    if(res=="Goodbye! Have a nice day." or res=="See you later!"):
        break