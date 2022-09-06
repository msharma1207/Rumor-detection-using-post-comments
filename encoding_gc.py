import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import logging
plt.style.use('fivethirtyeight')

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D , Dropout , Conv1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding , Concatenate
from tensorflow.keras.models import Model
import pickle
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
import os
import json
import pickle as pkl
import preprocess
import nltk


from keras import initializers

"""### Loading glove vectors"""

embeddings_index = dict()     # LOADING GLOVE TO MEMORY !
f = open('glove.6B.200d.txt',encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

"""### Reading and preprocess"""

Max_cmt = maxpost_length =m = maxcmt_length = dim = 200
l=80
mdash = 80

#make rumor for gc
#read news/reply

fakes = 5325
real = 16815

label = []
for i in range(fakes):
    label.append(0)  #5323 fakes , then 16817 real
for i in range(real):
    label.append(1)

post = []
dirr = 'prep_gc'
news = os.path.join(dirr,'news.txt')

with open(news,encoding="utf-8") as f:
    f1 = f.readlines()
    for line in f1:
        post.append(line)

dirr = 'prep_gc'
replies = os.path.join(dirr,'reply')
replyy= []

for i in range(22140):
    temp = 'reply'
    temp += str(i) 
    temp += '.txt'
    
    repp =  os.path.join(replies,temp)
    with open(repp,encoding="utf-8") as f:
        f1 = f.readlines()
        reptemp = []
        for line in f1:
            reptemp.append(line)

    replyy.append(reptemp)

import random
temp = list(zip(post, replyy, label))
random.shuffle(temp)
t1, t2, t3 = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
post, replyy, label = list(t1), list(t2),list(t3)

"""
## Encoding Post and comments
"""

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token = True ) 
all_text = []
all_text.extend(replyy)
all_text.extend(post)
tokenizer.fit_on_texts(all_text)
vocab_size = len(tokenizer.word_index) + 1

embedding_matrix = np.random.random((vocab_size, dim))   #100 is dim
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

train_post = post[:17000]
test_post = post[17001:]

train_cmt = replyy[:17000]
test_cmt = replyy[17001:]

y_train = label[:17000]
y_test = label[17001:]

#y_train = tf.keras.utils.to_categorical(y_train)   DO keras.to_catogorical after reading
#y_test = tf.keras.utils.to_categorical(y_test)

from tqdm import tqdm

def _encode_comments(comments):  ## call b4 model fit 
    encoded_texts = np.zeros((len(comments), Max_cmt, maxcmt_length), dtype='int32')
    for i,text in tqdm(enumerate(comments)):
        encoded_text = np.array(pad_sequences(
            tokenizer.texts_to_sequences(text),
            maxlen=maxcmt_length, padding='post', truncating='post', value=0))[:Max_cmt]
        encoded_texts[i][:len(encoded_text)] = encoded_text

    return encoded_texts


def _encode_post(post):  ## call b4 model fit 
    encoded_texts = np.zeros((len(post),maxpost_length, maxpost_length), dtype='int32')
    
    for i,text in tqdm(enumerate(post)):
        encoded_text = np.array(pad_sequences(tokenizer.texts_to_sequences(post),
            maxlen=maxpost_length, padding='post', truncating='post', value=0))[:maxpost_length]
        
        encoded_texts[i][:len(encoded_text)] = encoded_text

    return encoded_texts

######################################

with open('unencoded_train_cmt_gc','wb') as f3: pickle.dump(train_cmt, f3)
with open('unencoded_test_cmt_gc','wb') as f4: pickle.dump(test_cmt, f4)


label = np.array(label)
with open('label_gc_final','wb') as f2: pickle.dump(label, f2)  # DO keras.to_catogorical after reading

encoded_train_x = _encode_post(train_post)
trpost_reshaped1 = encoded_train_x.reshape(encoded_train_x.shape[0], -1)

with open('encoded_train_post_gc','wb') as f: pickle.dump(trpost_reshaped1, f)

encoded_val_x = _encode_post(test_post)
tepost_reshaped2 = encoded_val_x.reshape(encoded_val_x.shape[0], -1)

with open('encoded_test_post_gc','wb') as f1: pickle.dump(tepost_reshaped2, f1)

#################################
#with open('filename','rb') as f: arrayname1 = pickle.load(f)   # to read 
