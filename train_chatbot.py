import json
import random
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import pickle


lemmatizer = WordNetLemmatizer()
words=[]
classes = []
documents = []
ignore = ['?', '!', '.', ',', "'"]

# Load the prompts and answers dataset
with open("intents.json", "r") as json_file:
    intents = json.load(json_file)



for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        # Tokenize each word
        w = word_tokenize(pattern)  
        words.extend(w)

        # Add documents in the corpus
        documents.append((w, intent['tag']))

        # Add unique intents into classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lower and lemmatize each word
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]

# Remove duplicates and sort the list
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# documents = list of patterns & intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = meaningful words
print(len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


train = []
output_empty = np.zeros(len(classes), dtype="uint8")

for doc in documents:
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is 0 for every tag and 1 for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    train.append([bag, output_row])

# Shuffle features
random.shuffle(train)
train = np.array(train)


X_train = list(train[:,0])
y_train = list(train[:,1])

print("Training data created")
print(len(y_train[0]))

# Create model with 3 layers
# First layer contains 128 neurons, second layer has 64 neurons. 
# 3rd layer is the output layer that contains number of neurons
# equal to number of intents (80)
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model 
hist = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created and saved")
