# install intense.jdon file :contains certain categories of things that can happen:
#Example:the user can greet the chatbot , ask question about Price, goodbye
#they are provided with text we have have tag called greeting 
#And for each tag we also specify patterns like: hi, hello, hows it going? 
#chat bot takes as trainig data & see okay this is what greeting me looks like?!
#And I am going to adapet to the exact things 
# in json we re gonna have tag :of the particular category ,  patterns:are the exaples what a greeting look like :hi,hey,hello :
#And responses:  are hard coded so :They are going to be the exact responses:here we should take care of uppercase,lowercase, grammar 

# sourcery skip: for-index-underscore, hoist-statement-from-loop
import random
import json
import pickle #For serialization
import numpy as np  #pip install numpy !
import nltk #natural language toolkit ! pip install nltk
from nltk.stem import WordNetLemmatizer #To reduce the word to its stem (we dont lose any performace cuz its looking for the exact  word)
#Ex: work  works working  ..are the same words (this is what the lemmetizer does )
# from tensorflow.keras.models import sequential
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation , Dropout
from tensorflow.keras.optimizers import SGD #Stochastic Gradient descent 


#Lametize the individual words:
lameetizer=WordNetLemmatizer()
intents = json.loads(open('intens.json').read()) #read content of json file as text (load, we get the json object which is a dictionary )
words= []
classes= []
documents =[]
ignore_letters= ['?','!',',','.'] #the letters were not going to take care of 

for intent in intents['intents']:
    for pattern in intent['patterns']:#as a subkey , bleow subvalue!So sub dic
        word_list=nltk.word_tokenize(pattern) #Tokenize :means you get the text ,split it into individual hey how are you :hey,how, are, you !
        words.extend(word_list) #add that word to collection words (extend meanins taking the content and append it to a list )
        documents.append((word_list, intent['tag']))#append docs word_list and also classes of the particular intent
          #word_list belongs to this category to this class to this tag 
          
#To check if this class is already in the classes list :
        if intent['tag'] not in classes:
            classes.append((intent['tag']))

# print(documents) # all our intents written in Terminal 
words= [lameetizer.lemmatize(word for word in words if word not in ignore_letters)]
#To laminade the duplicates:
#set :eliminates the duplicates,sorted turns it back into a list & sorts it 

words = sorted(set(words))
#we will get a list of  lemitize:
# print(words)

classes = sorted(set(classes))# we should not have any duplicates

#To save them in files:
pickle.dump(words, open ('words.pkl', 'wb')) #as writing binary  

#the same things for the classes:
pickle.dump(classes, open('classes.pkl','wb' ))

#To get into the MACHINE LEARINING :(We need to represent these words As  numerical)
#To do that we use sth called:bag of words


training = []
output_empty =[0]* len(classes) #a template of zeros (we need as many zeros as there are classes so *len... )


for document in documents:
    bag = [] #for those of combincation we make an empty bag
    word_patterns = document[0]
      #wordpatterns equals list comprehension
    word_patterns = [lemmazier.lemmatize(word.lower()) for word in word_patterns]
    #for each word we want to know if it occurs in the pattern so :
    for word in words:
        if word in word_patterns:
            bag.append(1) if word in word_patterns else bag.append(0)


    output_row = list(output_empty) #to copy the list (were really not type casting it ,were copying )
    output_row[classes.index(document[1])] = 1 #wanna know the class which is at index one wanna know the index were gonna add to set this index in the output row 
    training.append([bag, output_row]) #append the whole thing here 


random.shuffle(training)
training = np.array(training)

#The features and lables
#To split it into x and y values :
train_x = list(training[:,0]) #everything and zero dimension
train_y = list(training[:,0])

#The features and lables were using to train our neural network :
#To start our neural Network :
model = Sequential()
#Add some layers :input layer:add(Dense),  length of training data for x ,activation='relu':Specify the activation function to be rectified linear unit 
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) #128 neuran ,input_shape=size of trainig data for x 
model.add(Dropout(0.5))#to prevent overfeeding 
model.add(Dense(64, activation= 'relu'))    #we add another dense layer with 63 neurons and activation function 
#another dropout layer:
model.add(Dropout(0.5))
#another dense layer but :we have as many neurons an there are classes 
model.add(Denselen(train_y), activation='softmax') #activation here is like a softmax function cuz :thats the function gonna allow us to add up the results by 
#softmax:the function that sums up or scales the in the output layer so they all add up to one so that we have sort of precentages of how likeli it is to have that output 

#To define a stochastic gradient decent optimizer by  :

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#0.01 is the learning rate ,decay=1e-6 another way of writing decimal places 

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
 
#epocs :were going to fee the same data 200 times into the neural network in a batch size of 5:
hist= model.fit(np.array(train_x), np.array(train_y),epochs=200, batch_size=5,verbose=1  ) #verbose=1 :we get a medium amount of information 
model.save('chatbotmodel.h5',hist)
print("Done")
 
#Now we need to create a chatbot application !
   