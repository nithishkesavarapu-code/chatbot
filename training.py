import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

import random
import json
import pickle #serialization
import numpy as np

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

faqs = json.loads(open('FAQ.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for faq in faqs["faq"]:
    for pattern in faq["patterns"]:
        word_list = nltk.word_tokenize(pattern) #divide the sentence into words
        words.extend(word_list)
        documents.append((word_list,faq["tag"]))
        if faq['tag'] not in classes:
            classes.append(faq["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row])

random.shuffle(training)

train_x = []
train_y = []

for bag, output_row in training:
    train_x.append(bag)
    train_y.append(output_row)

train_x = np.array(train_x)
train_y = np.array(train_y)

model = Sequential()
model.add(Dense(128,input_shape= (len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5',hist)
print("Done")

