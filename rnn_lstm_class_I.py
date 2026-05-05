df = open('/content/story.txt','r').read()

df

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts([df])

len(tokenizer.word_index)

input_sequences = []  #creating empty list
for sentence in df.split('\n'):   #iterator wherever \n is there, splits the file
  tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0] #first sentence

  for i in range(1,len(tokenized_sentence)):  #Example 11 sentences: (1,10) converted numbers
    input_sequences.append(tokenized_sentence[:i+1]) #indexing and slicing [1+1=2]:0,1;[2+1=3]:0,1,2;[3+1=4]:0,1,2,3 positioned numbers

input_sequences

max_len =max([len(x) for x in input_sequences])
max_len

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre') #adding 0s before the words/tokenised nums

padded_input_sequences

X = padded_input_sequences[:,:-1]

y=padded_input_sequences[:,-1]

X.shape

y.shape

from tensorflow.keras.utils import to_categorical
y = to_categorical(y,num_classes=374)

y.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Dropout

##from tensorflow.keras.callbacks import EarlyStopping

##early_stop = EarlyStopping(
 #   monitor='loss',
 #   patience=5,
 #   restore_best_weights=True
#)

model =Sequential()
model.add(Embedding(374,100))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Bidirectional(LSTM(150, return_sequences=False)))
model.add(Dense(374,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

model.summary()

model.fit(X,y,epochs=100, batch_size = 32, validation_split=0.2)

import numpy as np
def predict_next_word(model, tokenizer, text, max_len):
    # Convert text → sequence
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Pad sequence
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

    # Predict
    predicted = np.argmax(model.predict(token_list), axis=-1)[0]

    # Convert index → word
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word

predict_next_word(model, tokenizer, "In the heart of a", max_len)