
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


# In[3]:

text = (open("Investors.txt")).read()


# In[13]:

len(text)


# In[5]:

text.lower()


# In[6]:

unique_char = sorted(list(set(text)))


# In[8]:

len(unique_char)


# In[11]:

char_int = {}
int_char = {}

for i, c in enumerate (unique_char):
    char_int.update({c: i})
    int_char.update({i: c})


# In[40]:

import numpy as np


# In[32]:

#start_index = np.random.randint(0, 5)


# In[33]:

#start_index


# In[38]:

X = []
Y = []
length = len(text)
seq_length = 100
for i in range(0, length-seq_length, 1):
     sequence = text[i:i + seq_length]
     label =text[i + seq_length]
     X.append([char_int[char] for char in sequence])
     Y.append(char_int[label])


# In[42]:

X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(unique_char))
Y_modified = np_utils.to_categorical(Y)


# In[43]:

Y_modified


# In[44]:

X_modified


# In[46]:

model = Sequential()
model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:

print("Training Start")
# fitting the model
model.fit(X_modified, Y_modified, epochs=5, batch_size=50)
print("Training end")
# picking a random seed
start_index = np.random.randint(0, len(X)-1)
new_string = X[start_index]

# generating characters
for i in range(100):
    x = np.reshape(new_string, (1, len(new_string), 1))
    x = x / float(len(unique_char))

    #predicting
    pred_index = np.argmax(model.predict(x, verbose=0))
    char_out = int_char[pred_index]
    seq_in = [int_char[value] for value in new_string]
    new_string.append(pred_index)
    new_string = new_string[1:len(new_string)]


# In[ ]:
#combining text
txt=""
for char in full_string:
    txt = txt+char
txt



