from flask import Flask, request, jsonify, render_template
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import tflearn
import numpy as np
import tensorflow as tf


# Download nltk data
nltk.download('punkt')

# Initialize stemmer
stemmer = LancasterStemmer()

# Load intents data
with open('data.json') as file:
    data = json.load(file)

# Load words, labels, training data, and output from the data
words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in '?']
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Reset tensorflow graph
tf.compat.v1.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.embedding(net, input_dim=len(training[0]), output_dim=128)  
net = tflearn.lstm(net, 128, dropout=0.8)  
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

# Initialize model
model = tflearn.DNN(net)

# Load the trained model
model.load("model.h5")
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)
app = Flask(__name__)
@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict_diseases/<input>',methods=["POST","GET"])
def predict_diseases(input):
    
    if input.lower() == "quit":
        return jsonify({"response": "Prediction stopped."})
    else:
        results = model.predict([bag_of_words(input, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                response_list = nltk.sent_tokenize(str(responses[0]))
                return jsonify({"response": response_list})


if __name__ == '__main__':
    app.run(debug=True)
