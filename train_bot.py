# CHATBOT ANN MODEL COMPONENT

# IMPORTING THE NECESSARY LIBRARIES FOR THE TASK

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

# TO DISABLE TENSORFLOW FROM DISPLAYING ERROR MESSAGES RELATING TO GPU ALLOCATION
# **Not related to context of chatbot model implementation**

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# VARIABLES INITIALIZATION AND LOADING INTENTS DATASET
 
words=[]
classes = []
documents = []
ignore_words = ['?','!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

# PRE-PROCESSING THE INTENTS TRAINING DATASET

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # TOKENIZING EACH WORD OF THE PATTERN
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # PAIRING EACH WORD OF THE PATTERN WITH ITS RESPECTIVE TAG 
        documents.append((w, intent['tag']))
        # STORING EACH UNIQUE TAG
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# LEMMATIZING, IGNORING SPECIAL CHARACTERS AND CONVERTING EACH WORD INTO LOWERCASE
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# SORTING ELEMENTS OF THE CLASS IN ASCENDING ORDER
classes = sorted(list(set(classes)))
# DISPLAYING THE DOCUMENTS(PAIR BETWEEN PATTERN AND ITS TAG) OF THE DATA
print (len(documents), "documents", documents)
# DISPLAYING THE TAGS OF THE DATA
print (len(classes), "classes", classes)
# DISPLAYING ALL THE UNIQUE WORDS OF ALL THE PATTERNS IN THE DATA
print (len(words), "unique lemmatized words", words)

# SERIALIZING THE WORDS AND CLASSES OBJECTS WHICH IS LATER USED BY GUI.PY
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# CREATING TRAINING DATA
training = []
output_empty = [0] * len(classes)
# CREATING BAG OF WORDS FOR EACH SENTENCE OF THE TRAINING SET
for doc in documents:
    bag = []
    # LIST OF TOKENIZED WORDS FOR A PATTERN
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # UPDATE BAG OF WORDS WITH 1 IF MATCH FOUND IN CURRENT POSITION, IF NOT APPEND 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # FOR EACH PATTERN, APPEND 1 TO OUTPUT FOR CURRENT TAG, 0 FOR OTHER TAGS  
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
# SHUFFLING THE DATA ALONG BAG OF WORDS AXIS AND CONVERTING INTO NUMPY ARRAY
random.shuffle(training)
training = np.array(training, dtype=object)
# CREATING TRAINING AND TESTING LISTS
# TRAIN_X GIVES THE INPUT SIZE OF THE MODEL i.e. NUMBER OF PATTERNS
# TRAIN_Y GIVES THE OUTPUT SIZE OF THE MODEL i.e. NUMBER OF TAGS
train_x = list(training[:,0])
train_y = list(training[:,1])
print(len(train_x[0]))
print(len(train_y[0]))
print("\nTraining data is created")

# DEFINING THE MODEL
# 1 INPUT LAYER, 3 HIDDEN LAYERS AND 1 OUTPUT LAYER
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# COMPILING THE MODEL
# OPTIMIZER - STOCHASTIC GRADIENT DESCENT WITH NESTEROV ACCELERATED GRADIENT
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# FITTING THE TRAINING DATA
hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=4, verbose=1)

# METRICS
score = model.evaluate(np.array(train_x), np.array(train_y))
print(f"\nTest accuracy: {score[1]*100:0.2f}%")

# SAVING THE MODEL
model.save('model.h5', hist)

print("\n Model is Created")
