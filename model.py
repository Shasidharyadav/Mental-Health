import numpy as np
import json
import random
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from tensorflow.keras.models import load_model

model = load_model('chatbot_model.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = tokenize(sentence)
    bag = np.zeros(len(words), dtype="uint8") 

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                # Assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s"%w)
    return bag

def predict_class(sentence, model):
    # Filter out predictions below a threshold
    bag = bow(sentence, words, show_details=False)
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for item in results:
        return_list.append({"intent": classes[item[0]], "probability": str(item[1])})
    return return_list

def get_response(ints, intents):
    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    ERROR_MSG = "I am sorry, I cannot interpret what you are trying to say..."
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            return result
    return ERROR_MSG

def chat(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res
