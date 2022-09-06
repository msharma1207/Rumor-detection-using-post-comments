# -*- coding: utf-8 -*-
"""t15+t16.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1pUpWF6kHxCeb0ylSSiw4X5GtrVdRUNMK
"""

# -*- coding: utf-8 -*-
"""workingt15.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1-oGID4Cs1qaRh7hUAxfIb5HMCki8hDyU
"""

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

from keras import initializers

"""### Loading glove vectors"""

embeddings_index = dict()     # LOADING GLOVE TO MEMORY !
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

"""### Reading and preprocess"""

Max_cmt = 100
maxpost_length = 100
maxcmt_length = 100
l = 80
mdash = 80

import json
import nltk
import preprocess

labels = []
tweetid = []

ch = input("Which data to run ? 1 : T15 , 2 : T16 ")

if ch ==1 :
    label_path = 'label15.txt'
    mainpath = 'reply_15/'
else:
    label_path = 'label16.txt'
    mainpath = 'reply_16/'

with open(label_path) as f:
    for line in f:
        line = line.split(':') #change
        labels.append(line[0]) 
        x = int(line[1])
        tweetid.append(str(x))
        
rumor = []        #class 
for label in labels:
    if label =='true':
        rumor.append(1)
    else:
        rumor.append(0)


post = []
comments = []
todel = []
iddx = 0


count= 0

for ids in tweetid:           #accessing tweets from tweetid of labels
    path = mainpath
    path += ids
    path += '.json'
    f = open(path)
    data = json.load(f)
    prep_post = preprocess.preprocess_text(data[ids])
    del data[ids]                 # deleting post from data
    if(prep_post == "failed"):
        del rumor[iddx]
        continue
    iddx +=1
    post.append(prep_post)
    commentsperpost = []
    for tid in data:       # iterating comments for a post now
        count += 1
        if count>Max_cmt:
            count = 0
            continue
        prep_cmt = preprocess.preprocess_text(data[tid])
        commentsperpost.append(prep_cmt)
    comments.append(commentsperpost)

for allcmt in comments:
    for cmt in allcmt:  
        if cmt == "failed":
            allcmt.remove(cmt)
        if cmt == "":
            allcmt.remove(cmt)



"""## Encoding Post and comments
"""

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token = True ) 
all_text = []
all_text.extend(allcmt)
all_text.extend(post)
tokenizer.fit_on_texts(all_text)
vocab_size = len(tokenizer.word_index) + 1

embedding_matrix = np.random.random((vocab_size, 100))   #100 is dim
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

if ch == 1:
    trsize = 500
else:
    trsize = 300

train_post = post[:trsize]
test_post = post[trsize+1:]

train_cmt = comments[:trsize]
test_cmt = comments[trsize+1:]

y_train = rumor[:trsize]
y_test = rumor[trsize+1:]

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

y_train.shape

def _encode_comments(comments):  ## call b4 model fit 
    encoded_texts = np.zeros((len(comments), Max_cmt, maxcmt_length), dtype='int32')
    for i,text in enumerate(comments):
        encoded_text = np.array(pad_sequences(
            tokenizer.texts_to_sequences(text),
            maxlen=maxcmt_length, padding='post', truncating='post', value=0))[:Max_cmt]
        encoded_texts[i][:len(encoded_text)] = encoded_text

    return encoded_texts

def _encode_post(post):  ## call b4 model fit 
    encoded_texts = np.zeros((len(post),maxpost_length, maxpost_length), dtype='int32')
    for i,text in enumerate(post):
        encoded_text = np.array(pad_sequences(tokenizer.texts_to_sequences(post),
            maxlen=maxpost_length, padding='post', truncating='post', value=0))[:100]
        
        encoded_texts[i][:len(encoded_text)] = encoded_text

    return encoded_texts

#xxx= _encode_post(train_post)

##_encode_post(train_post).shape

"""### Creating Graph -> GCN -> Self Attention"""

class graph(Layer):
    def __init__(self, **kwargs):
        super(graph, self).__init__(**kwargs)
        self.init = initializers.get('normal')
        self.mdash = mdash  #80
        self.m = maxpost_length #200
        self.max_cmt = Max_cmt      #200               
        self.l = l          #80

    def build(self, input_shape, mask=None):
        self.W0 = K.variable(self.init((self.l, self.mdash)))

        self.W1 = K.variable(self.init((self.mdash, self.mdash)))
        
        self._trainable_weights = [self.W0, self.W1]
        

    def call(self,cmts,post):
        post = tf.expand_dims(post, axis=1)
        
        rows, cols = (self.max_cmt+1, self.max_cmt+1)
        A = [[0.0]*cols]*rows      #A matrix - (n+1)x(n+1)
        
        for i in range(rows):
            for j in range(cols):
                if i==0 or j==0:
                    A[i][j] = 1.0
                
        A[0][0] = 0
 
        concat = Concatenate(axis=1)
        X = concat([post,cmts])    # X matrix - (n+1)x100  - Nonex101x100
        A = tf.stack(A)            # 101 x 101
        A = tf.expand_dims(A, axis=0)
        
        h1 = tf.matmul(A,X)
       
        H1 = K.tanh(K.dot(h1,self.W0))
        #H1 = K.tanh(tf.einsum('btd,dD,bDn->btn', A, X, self.W0))
        
        h2 = tf.matmul(A,H1)
        H2 = K.tanh(K.dot(h2,self.W1))
       
        return H2

#H2 = graph(name = "GCN")(all_comment_encoder, lstm_post)
#H2

"""### Self attention of comments"""

class selfAtt(Layer):
    def __init__(self, **kwargs):
        super(selfAtt, self).__init__(**kwargs)
        self.init = initializers.get('normal')
        self.mdash = mdash  #80
        self.m = maxpost_length #200
        self.max_cmt = Max_cmt      #200               
        self.l = l          #80
        

    def call(self,H2):
        Xc = H2[:,1:,:]
        q,k,v = Xc,Xc,Xc
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) 

        Xc_att = tf.matmul(attention_weights, v) 
        
        return Xc_att





"""### Co-attention Between comments and post"""

from tensorflow.keras.layers import Layer, InputSpec
from keras import initializers
from keras import backend as K
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *

class Coattn(Layer):            #change max_cmt later
    """
    Co-attention layer which accept post and comment states and computes co-attention between them and returns the
     weighted sum of the content and the comment states
    """
   
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.mdash = mdash
        self.m = maxpost_length
        self.max_cmt = Max_cmt                      # Change when needed 
        self.l = l
        super(Coattn, self).__init__(**kwargs)

    def build(self, input_shape, mask=None):
        self.W_cw = K.variable(self.init((self.mdash, self.l)))

        self.Wc = K.variable(self.init((self.max_cmt, self.mdash)))
        self.Ww = K.variable(self.init((self.m, self.l)))

        self.whw = K.variable(self.init((1,self.m)))
        self.whc = K.variable(self.init((1,self.m)))
        self._trainable_weights = [self.W_cw, self.Wc, self.Ww, self.whw, self.whc]

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):  #x is input - > [Xc_att , , post_lstm] - > dim : nonex20x80 , nonex100x80 ( will permute in funct) 
        comment_rep = x[0]  # 20x100
  
        sentence_rep = x[1] # 100x100
   
        sentence_rep_trans = K.permute_dimensions(sentence_rep, (0, 2, 1)) #100x100 - W'
        comment_rep_trans = K.permute_dimensions(comment_rep, (0, 2, 1)) #100x20 - C
        
        F = K.tanh(tf.einsum('btd,dD,bDn->btn', comment_rep, self.W_cw, sentence_rep_trans)) #20,100

        #F = K.batch_dot(comment_rep*self.W_cw,sentence_rep_trans)
        #K.eval(F)
        #K.eval(F)
        F_trans = K.permute_dimensions(F, (0, 2, 1)) #100,20
       
        
        Hc = K.tanh(tf.einsum('kd,bdn->bkn', self.Ww, sentence_rep_trans) + tf.einsum('kd,bdt,btn->bkn', self.Wc,comment_rep_trans, F))
    
        #Hc = K.dot(self.Ww,sentence_rep_trans ) + K.batch_dot(comment_rep_trans*self.Wc,F_trans)
        
        Hw = K.tanh(tf.einsum('kd,bdt->bkt', self.Wc, comment_rep_trans) + tf.einsum('kd,bdn,bnt->bkt', self.Ww,
                                                                                     sentence_rep_trans, F_trans))
        
        Aw = K.softmax(tf.einsum('yk,bkn->bn', self.whw, Hw)) 
        Ac = K.softmax(tf.einsum('yk,bkt->bt', self.whc, Hc))
        co_w = tf.einsum('bdn,bn->bd', sentence_rep_trans, Aw)
        co_c = tf.einsum('bdt,bt->bd', comment_rep_trans, Ac)
        co_sc = K.concatenate([co_w, co_c], axis=1)  # output shape has to be 1x160 . check

        return co_sc

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.l + self.l)







"""## Model Build -  Adding layers"""

import random  as rn

      
post_input = Input(shape=(maxpost_length,) , name = 'post_input')
post_emb = Embedding(vocab_size , 100, weights=[embedding_matrix] , input_length= maxpost_length,  trainable=True , )(post_input)
post_lstm = tf.keras.layers.LSTM(l, dropout = 0.5)(post_emb)

post_encoder = Model(post_input  ,post_lstm)

fin_post_input = Input(shape=(maxpost_length, maxpost_length), dtype='float32',name = "Post-Input" )
fin_post_encoder = TimeDistributed(post_encoder)(fin_post_input)

lstm_post = tf.keras.layers.LSTM(l, dropout = 0.5)(fin_post_encoder)

sentence_level_encoder = Model(fin_post_input, lstm_post)

#sentence_level_encoder.summary()

comment_input = Input(shape=(maxcmt_length,), dtype='int32',name = 'com_input')
com_emb = Embedding(vocab_size, 100, weights=[embedding_matrix],input_length=maxcmt_length, trainable=True,
                                 mask_zero=True)(comment_input)
cmt_lstm = tf.keras.layers.LSTM(l, dropout = 0.5)(com_emb)
#c_att = AttLayer(name='comment_word_attention')(cmt_lstm)
com_encoder = Model(comment_input, cmt_lstm, )

all_comment_input = Input(shape=(Max_cmt, maxcmt_length ),dtype= 'float32', name = "Comments-Input")
all_comment_encoder = TimeDistributed(com_encoder, name='comment_sequence_encoder')(all_comment_input)

lstmC =  tf.keras.layers.LSTM(l, dropout = 0.5)(all_comment_encoder)

comment_sequence_encoder = Model(all_comment_input, lstmC)
#comment_sequence_encoder.summary()
##ADD graph and GCN HERE



#Convolution layer
conv1 = Conv1D(filters = l, kernel_size = (3,), activation='relu', padding = 'same', name = 'conv1')(all_comment_encoder)
h_loc1 = MaxPooling1D(pool_size=Max_cmt, name = 'hlocal' , padding='valid')(conv1)
dropconv = Dropout(0.5)(h_loc1)
h_loc = Flatten()(dropconv)


#Creating graph and applying Convolution 
H2 = graph(name = "GCN")(all_comment_encoder, lstm_post)

#Self attention layer
Xc_att = selfAtt(name = "self-Attention")(H2)


#Coattention layer
h_global = Coattn(name="h_global")([Xc_att, fin_post_encoder])

from keras.backend import cast
concat = Concatenate(axis=1)
final = concat([h_global,h_loc ])  # Merging the global and local features

from tensorflow.keras.utils import plot_model
Drop = Dropout(0.5)(final)
preds = Dense(2, activation='softmax' , name = 'softmax1')(final)
model = Model(inputs=[all_comment_input, fin_post_input], outputs=preds)
model.summary()

plot_model(model, to_file='final.png', show_shapes=True)

"""## Calling training"""

# Custom functions
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

from tensorflow.keras import callbacks
def train(train_x, train_y, train_c, val_c, val_x, val_y,
              batch_size=8, epochs=50,
              embeddings_path=False, saved_model_filename=None, ):
        # Fit the vocabulary set on the content and comments
        # fit_on_texts_and_comments(train_x, train_c, val_x, val_c)
        tf.keras.backend.clear_session()
        encoded_train_x = _encode_post(train_x)
        encoded_val_x = _encode_post(val_x)
        encoded_train_c = _encode_comments(train_c)
        encoded_val_c = _encode_comments(val_c)
        
        optimizer = 'adam'
        loss = 'binary_crossentropy'
        loss2 = 'categorical_crossentropy'
        
        checkpoint_filepath = './'
        chkpt = ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor='val_accuracy',mode='max', save_best_only=True)
        
        early = EarlyStopping(patience=40,monitor="val_accuracy")
        
        model.compile(optimizer, loss2 , metrics=['accuracy',f1_m,precision_m, recall_m])
        history = model.fit([encoded_train_c, encoded_train_x], y=train_y,
                  validation_data=([encoded_val_c, encoded_val_x], val_y),
                       batch_size=batch_size, epochs=epochs, verbose=1 , callbacks=[early])
        
        from matplotlib import pyplot as plt
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
  
        return history



model = train(train_post ,y_train ,train_cmt , test_cmt , test_post , y_test)

model.load_model("my_model.h5")

model.summary()