import json
import random
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pickle

# Download necessary NLTK models
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',', "'"]

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = word_tokenize(pattern)
        # Add words to the words list
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Initialize training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [0]*len(words)
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        if w in pattern_words:
            bag[words.index(w)] = 1
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists. X - patterns, Y - intents
X_train = np.array(list(training[:, 0]), dtype='float32')
y_train = np.array(list(training[:, 1]), dtype='float32')

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("Model is created and saved")
