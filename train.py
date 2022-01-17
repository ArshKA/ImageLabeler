import numpy as np
import pandas as pd
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import random
from copy import copy
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import adam_v2
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

LABEL_PATH = '/content/flickr30k_images/results.csv'
EMBEDDING_PATH = '/content/glove.6B.200d.txt'
IMAGE_PATH = '/content/flickr30k_images/flickr30k_images'
max_len = 34

def get_descriptions(LABEL_PATH):
  with open(LABEL_PATH, 'r') as file:
    doc = file.readlines()

  descriptions = dict()
  for line in doc[1:]:

    # split line by white space
    line = line.split('| ')
    
    # take the first token as image id, the rest as description
    try: image_id, image_desc = line[0], line[2]
    except: continue
    
    if len(image_desc.split()) > max_len:
      continue
    # extract filename from image id
    image_id = image_id.split('.')[0]
    
    # convert description tokens back to string
    if image_id not in descriptions.keys():
        descriptions[image_id] = [image_desc]
    else:
      descriptions[image_id].append(image_desc)

  return descriptions

descriptions = get_descriptions(LABEL_PATH)

table = str.maketrans('', '', string.punctuation)
for key in descriptions.keys():
    desc_list = copy(descriptions[key])
    for i in range(len(desc_list)):
        desc = desc_list[i]
        desc = 'startseq ' + desc + ' endseq'
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each token
        desc = [w.translate(table) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word)>1]
        # remove tokens with numbers in them
        desc = [word for word in desc if word.isalpha()]

        desc = [word for word in desc if word != 'the']
        # store as string

        desc_list[i] =  ' '.join(desc)
    descriptions[key] = copy(desc_list)

print('Creating Vocabulary...')
vocabulary = {'Not_all_zeros': 20}
for val in descriptions.values():
  for sentence in val:
    for word in sentence.split():
      if word not in vocabulary:
        vocabulary[word] = 1
      else: vocabulary[word] += 1
vocabulary = [x for x in vocabulary.keys() if vocabulary[x] >= 5]

vocab = {}
with open(EMBEDDING_PATH, 'r') as file:
  for row in file:
    row = row.split()
    word = row[0]
    vector = np.array(row[1:])
    if (word in vocabulary):
      vocab[word] = vector

for x in vocabulary:
  if x not in vocab.keys():
    if (x == 'startseq'):
      vocab[x] = np.ones((200))
    elif (x == 'endseq'):
      vocab[x] = np.zeros((200))
    else:
      vocab[x] = np.zeros((200))

wordtoix = {}
ixtoword = {}
for i, voc in enumerate(vocabulary):
  wordtoix[voc] = i
  ixtoword[i] = voc

print('Processing Images...')
def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

# Function to encode a given image into a vector of size (2048, )
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

# Load the inception v3 model
model = InceptionV3(weights='imagenet')
# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)

tuple_dataset = []

i = 0
for img, des in descriptions.items():
  i += 1
  if i%(len(descriptions)//10) == 0:
    print("Encoding Image {}/{}".format(i, len(descriptions)))
  img = '{}/{}.jpg'.format(IMAGE_PATH, img)
  try: img = encode(img)
  except: continue
  for x in des:
    tuple_dataset.append((img, x))

def data_generator(tuple_dataset, wordtoix, max_length, batch_size):
    X1 = np.zeros((batch_size, 2048))
    X2 = np.zeros((batch_size, max_length))
    Y = np.zeros((batch_size, len(wordtoix)))
    n=0
    # loop for ever over images
    while True:
      img, sentence = tuple_dataset[random.randrange(0, len(tuple_dataset))]
      sentence = sentence.split()
      
      embedding_sen = np.zeros((max_length)).tolist()
      sentence = [wordtoix[x] for x in sentence if x in wordtoix.keys()]
      if (len(sentence) <= 2):
        continue
      i = random.randrange(1, len(sentence))
      embedding_sen[:i] = sentence[:i]
      next_word = np.zeros((len(wordtoix)))
      next_word[sentence[i]] = 1
      X1[n] = img
      X2[n] = copy(embedding_sen)
      Y[n] = next_word
      n += 1
      if n==batch_size:
        yield [X1, X2], Y
        X1 = np.zeros((batch_size, 2048))
        X2 = np.zeros((batch_size, max_length))
        Y = np.zeros((batch_size, len(wordtoix)))
        n=0


embedding_dim = 200
vocab_size = len(vocabulary)
# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    #if i < max_words:
    embedding_vector = vocab.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

def create_model(max_len, summary=False):
  inputs1 = Input(shape=(2048,))
  fe1 = Dense(512, activation='relu')(inputs1)
  fe2 = Dense(256, activation='relu')(fe1)
  inputs2 = Input(shape=(max_len,))
  se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
  se2 = Dropout(0.2)(se1)
  se3 = LSTM(512, return_sequences=True)(se2)
  se4 = LSTM(256)(se3)

  decoder1 = add([fe2, se4])
  decoder2 = Dense(256, activation='relu')(decoder1)
  decoder3 = Dense(256, activation='relu')(decoder2)
  decoder4 = Dense(512, activation='relu')(decoder3)

  outputs = Dense(vocab_size, activation='softmax')(decoder4)
  model = Model(inputs=[inputs1, inputs2], outputs=outputs)

  if summary:
    model.summary()

  model.layers[1].set_weights([embedding_matrix])
  model.layers[1].trainable = False
  return model

model = create_model(max_len, summary=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

generator = data_generator(tuple_dataset[len(tuple_dataset)//5:], wordtoix, max_len, 128)
val_generator = data_generator(tuple_dataset[:len(tuple_dataset)//5], wordtoix, max_len, 64)
model.fit_generator(generator, epochs=1, verbose=1, steps_per_epoch=200, validation_data = val_generator, validation_steps=27, epochs=100)
