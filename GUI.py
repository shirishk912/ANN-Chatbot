# CHATBOT GUI COMPONENT

# IMPORTING THE NECESSARY LIBRARIES FOR THE TASK

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random
from tkHyperLinkManager import HyperlinkManager
import webbrowser
from functools import partial
from keras.models import load_model
import tkinter
from tkinter import *

# TO DISABLE TENSORFLOW FROM DISPLAYING ERROR MESSAGES RELATING TO GPU ALLOCATION
# **Not related to context of chatbot model implementation**
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BORDERS = "#4F4F4F"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
SEND_BTN = "#90EE90"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

# LOADING THE SAVED MODEL, INTENTS FILE, DESERIALIZING WORDS AND CLASSES FILE
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# PRE-PROCESSING (TOKENIZING AND LEMMATIZING) USER INPUT TEXT
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# PERFORMING BAG OF WORDS TECHNIQUE ON USER INPUT PATTERN AND WORDS LIST

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# PREDICTING THE CLASS(TAG) TO WHICH THE USER INPUT BELONGS TO
def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    # FILTERING OUT PREDICTIONS BELOW A THRESHOLD
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # SORTING BY STRENGTH OF PROBABILITY OF THE PREDICTION
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# GETTING THE PREDICTED RESPONSE
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# PREDICTING AND GETTING THE RESPONSE AND SENT THE RESPONSE TO DISPLAY
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# RETRIEVING HYPERLINKS FOR THE PREDICTED RESPONSE

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



# DEFINING A CALLBACK FUNCTION TO OPEN A NEW BROWSER TAB
def callback(url):
   webbrowser.open_new_tab(url)
 
# GETTING USER INPUT
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "STUDENT: \n\n" + msg + '\n\n')
        
        # RETRIEVING AND DISPLAYING THE RESPONSE
        res = chatbot_response(msg)
        link= chatbot_response_link(msg)
        ChatLog.insert(END, "CHATBOT: \n\n" + res)
        ChatLog.insert(END, "here" + '\n\n', hyperlink.add(partial(webbrowser.open,link)))     
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

# CREATING CHAT WINDOW BASE GUI STRUCTURE

base = Tk()
base.title("Swansea University Chatbot")
base.configure(width=770, height=550, bg=BG_COLOR)
base.resizable(width=False, height=False)
# HEAD LABEL
head_label = Label(base, bg="#FDF5E6", fg="#33A1C9", text="Welcome", font=FONT_BOLD, pady=10)
# A DIVIDER SEPERATING HEAD LABEL AND CHATLOG
line = Label(base, width=750, bg=BORDERS)
# CHATLOG FRAME
ChatLog = Text(base, width=20, height=2, bg="#292421", fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
ChatLog.config(cursor="arrow", state=DISABLED)
hyperlink= HyperlinkManager(ChatLog)
# BINDING SCROLLBAR TO THE CHATLOG FRAME
scrollbar = Scrollbar(ChatLog)
scrollbar.configure(command=ChatLog.yview)
# A DIVIDER SEPERATING CHATLOG AND INPUT AREA
bottom_label = Label(base, bg=BORDERS, height=80)
# INPUT ENTRY BOX
EntryBox = Text(bottom_label, bg="#FFF5EE", fg="#1E1E1E", font=FONT)
EntryBox.focus()
# SEND BUTTON
SendButton = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=SEND_BTN,
                    command= send )
# PLACING ALL THE COMPONENTS IN THE BASE STRUCTURE
head_label.place(relwidth=1)
line.place(relwidth=1, rely=0.07, relheight=0.012)
ChatLog.place(relheight=0.745, relwidth=1, rely=0.08)
scrollbar.place(relheight=1, relx=0.974)
bottom_label.place(relwidth=1, rely=0.825)
EntryBox.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
SendButton.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
base.mainloop()
