#!/usr/bin/env python
# coding: utf-8

#Installing tensorflow-datasets
#pip install tensorflow-datasets
#pip install beautifulsoup4

#importing the Dependency

import numpy as np
import re
import pandas as pd
from bs4 import BeautifulSoup
import tensorflow as tf
import tensorflow.keras as layers
import tensorflow_datasets as tfds

#Specifing the columns
col = ["Setimentals", "id", "date","query", "user", "text"]
#Importing the columns
X_data = pd.read_csv("trainingandtestdata/training.1600000.processed.noemoticon.csv",header=None, names = col,engine='python',encoding= 'latin1')
#Dropping the unwanted datas
X_data.drop(["id", "date","query", "user"], axis = 1, inplace = True)


def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Removing the @
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Removing the URL links
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    # Keeping only letters
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Removing additional whitespaces
    tweet = re.sub(r" +", ' ', tweet)
    return tweet

#cleaning up of data
data = [clean_tweet(tweets) for tweets in X_data.text]
data_lables = X_data.Setimentals.values
data_lables[data_lables==4]=1


#assigning up of numbers to the sentances
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(data, target_vocab_size=2**16) 
data_input = [tokenizer.encode(sentence) for sentence in data]

#Get the max lenght of the sentence
MAX_LEN = max([len(sentance) for sentance in data_input])
#Assigning Zero at the end of each sentance to make all the sentence equal length
data_input = tf.keras.preprocessing.sequence.pad_sequences(data_input, value=0, padding='post', maxlen=MAX_LEN)

#create ramdom test indexes

test_idx = np.random.randint(0, 800000, 8000)
test_idx = np.concatenate(test_idx,test_idx+800000)

#Creating test and train with the random generated string 

test_input = data_input[test_idx]
test_labels = data_lables[test_idx]
train_input = np.delete(data_input, test_idx, axis=0)
train_labels = np.delete(data_lables, test_idx)
 
#Modeling of datamodel
class DCNN(tf.keras.Model):
    
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=512,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="dcnn"):
        super(DCNN, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim)
        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",
                                    activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,
                                     padding="valid",
                                     activation="relu")
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                      kernel_size=4,
                                      padding="valid",
                                      activation="relu")
        self.pool = layers.GlobalMaxPool1D() # no training variable so we can
                                             # use the same layer for each
                                             # pooling step
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)
        
        merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3 * nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)
        
        return output

#Configuration Parameters
VOCAB_SIZE = tokenizer.vocab_size

EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = len(set(train_labels))

DROPOUT_RATE = 0.2

BATCH_SIZE = 32
NB_EPOCHS = 5

#Assigning 
Dcnn = DCNN(vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, nb_filters=NB_FILTERS, FFN_units=FFN_UNITS, nb_classes=NB_CLASSES, dropout_rate=DROPOUT_RATE)



checkpoint_path = "trainingandtestdata/"

ckpt = tf.train.Checkpoint(Dcnn=Dcnn)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")
    
Dcnn.fit(train_inputs,
         train_labels,
         batch_size=BATCH_SIZE,
         epochs=NB_EPOCHS)
ckpt_manager.save()

results = Dcnn.evaluate(test_inputs, test_labels, batch_size=BATCH_SIZE)
print(results)

Dcnn(np.array([tokenizer.encode("bad teacher")]), training=False).numpy()

