import random
import json
import pickle 
import numpy as np 
import nltk 
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lameetizer=WordNetLemmatizer()
intents = json.loads(open('intens.json').read())


#we need to load all classes and model:
words = pickle.load(open('words.pkl', 'rb')) #reding binary mode


#we need to load the :
words = pickle.load(open('words.pkl', 'rb'))
classes=pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')
#Till here we get numerical data but we want to end up with words 
#we need 4 function :function for cleaning up the sentences
#function for getting the bag of words
#function for predicting the class  based on the sentences  
#function for getting a responce 


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
 #To lammetize the word
    sentence_words =  [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
    
    
#bag of words:to convert a sentce into a bog of words :into a list full of zeros and 1
#that indicates the words is there or not using flax  
def bag_of_words(sentence):
    sentence_word = clean_up_sentence(sentence)
    bag = [0] * len(words) # initial bag for zeros  as many zeors as there are 
    for w in sentence_words:
        for i , word in enumerate(words):
            if word == w :
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):  # sourcery skip: list-comprehension
    bow = bag_of_words(sentence)
    res = model.predict(np.array(([bow])))[0]
    ERROR_THRESHOLD = 0.25 
    resutl = [[i, r] for i , r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort( key =lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability':str(r[1])})










   

    
    
    
    
    
    
    
    
    
    












