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
    
        res = chatbot_response(msg)
        link= chatbot_response_link(msg)
        ChatLog.insert(END, "CHATBOT: " + res)
        ChatLog.insert(END, "here" + '\n\n', hyperlink.add(partial(webbrowser.open,link)))
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

# CREATING CHAT WINDOW BASE STRUCTURE

base = Tk()
base.title("Swansea University Chatbot")
base.configure(width=770, height=550, bg=BG_COLOR)
base.resizable(width=False, height=False)

head_label = Label(base, bg=BG_COLOR, fg=TEXT_COLOR, text="Welcome", font=FONT_BOLD, pady=10)

line = Label(base, width=750, bg=BG_GRAY)

#Create Chat window
ChatLog = Text(base, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
ChatLog.config(cursor="arrow", state=DISABLED)

hyperlink= HyperlinkManager(ChatLog)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(ChatLog)
scrollbar.configure(command=ChatLog.yview)

# bottom label
bottom_label = Label(base, bg=BG_GRAY, height=80)

#Create the box to enter message
EntryBox = Text(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
EntryBox.focus()

#Create Button to send message
SendButton = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                    command= send )


#Place all components on the screen
head_label.place(relwidth=1)
line.place(relwidth=1, rely=0.07, relheight=0.012)
ChatLog.place(relheight=0.745, relwidth=1, rely=0.08)
scrollbar.place(relheight=1, relx=0.974)
bottom_label.place(relwidth=1, rely=0.825)
EntryBox.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
SendButton.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

base.mainloop()
