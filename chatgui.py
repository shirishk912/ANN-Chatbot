import nltk
import os   
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

from tkHyperLinkManager import HyperlinkManager
import webbrowser
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Retrieving hyperlinks

def getResponseLink(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            link = random.choice(i['link'])
            break
    return link

def chatbot_response_link(msg):
    ints = predict_class(msg, model)
    link = getResponseLink(ints, intents)
    return link


# GRAPHICAL USER INTERFACE (GUI) USING TKINTER LIBRARY

import tkinter
from tkinter import *


# Define a callback function

def callback(url):
   webbrowser.open_new_tab(url)
 

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "STUDENT: " + msg + '\n\n')
        ChatLog.config(foreground="black", font=("Arial", 8 ))
    
        res = chatbot_response(msg)
        link= chatbot_response_link(msg)
        ChatLog.insert(END, "CHATBOT: " + res)
        ChatLog.insert(END, "here" + '\n\n', hyperlink.add(partial(webbrowser.open,link)))
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

# CREATING CHAT WINDOW BASE STRUCTURE

base = Tk()
base.title("Swansea University Chatbot")
base.geometry("700x900")
base.resizable(width=TRUE, height=TRUE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", fg="black", height="800", width="150", font="Arial",)

ChatLog.config(state=DISABLED)

hyperlink= HyperlinkManager(ChatLog)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set


#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="38", height="2", font="Arial")
#EntryBox.bind("<Return>", send)


#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="10", height="2",
                    bd=0, bg="black", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=265, y=401, height=90)

base.mainloop()
