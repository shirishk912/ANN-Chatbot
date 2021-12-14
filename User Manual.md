# APPLICATION - CHATBOT TO AUTOMATE STUDENT SUPPORT
 
## SUMMARY
The main objective of the project is to automate the process of student support system at the Swansea University to an extent using a conversational agent, also commonly referred to as chatbot. The existing student support system at the university is basically handled by human agents. The response time for student queries ranges from immediate (in case of live chat with an agent) to days (in case of e-mail). The live chat agents provide quick response but are only operational during specified period of the university working days. This waiting time makes sense for a personalized/official query involving response from staff. But some basic queries could be handled without the help of a human agent. The chatbot we implement serves this purpose.
A chatbot is a computer program that simulates and processes human conversation (either written or spoken), allowing humans to interact with digital devices as if they were communicating with a real person [1]. These chatbots are broadly classified into two types based on their method of implementation, which would be discussed in further sections. The chatbot we implement is a retrieval-based chatbot which uses Natural Language Processing (NLP) and deep learning techniques to simulate human conversations through the medium of text. This project uses NLP and Deep Learning and hence can be classified as the sub domain of Artificial Intelligence. The field of Artificial Intelligence (AI) is evolving rapidly and AI chatbots are becoming increasingly valuable to organizations for automating business processes such as customer service, sales, and human resources [2]. The retrieval-based chatbot we implement provides instant response to most first-level queries and thus addresses the issue of waiting times and eliminates the need of human intervention for most basic queries. This chatbot model is trained with a set of predefined queries and responses using NLP and DL techniques. Although the training dataset is limited, since it’s custom built (using university website as a source) to answer most basic first level queries, it fairly serves the purpose and if required, can be enhanced into fully deployable software in the future by expanding the training dataset consisting of university’s live chat data to make it fully operational to partially replace the existing student support system.

## USER MANUAL

This document contains all the required information from setting up the system to running the chatbot application. The following are the pre-requisites for the system to run the chatbot application:
•	Python should have been installed previously, if not the installation file and guide for python is found here, https://www.python.org/downloads/.
•	Install the following packages required to run the application using any python package manager: nltk, json, pickle, numpy, keras, tensorflow, random, os, webbrowser, functools, and tkinter.
Now that all the prerequisites are installed, we are ready to run our application. To run the application, follow the steps:
•	Open command line (on Windows) or terminal (Mac or Linux) and navigate to the chatbot application folder.
•	The model is already trained, if we wish to see the training process or training statistics, the model can be trained again. To train the model (optional), run the command shown below,

python train_bot.py (On Windows)

python3 train_bot.py (On Linux or Mac)

•	To run the chatbot application, type the command shown below,

python GUI.py (On Windows)

python3 GUI.py (On Linux or Mac)

•	The above command runs the chatbot application interface and now we can interact with the chatbot.


