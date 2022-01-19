import numpy as np
import pandas as pd
import string
import random
from copy import copy
import csv
import os
from PIL import Image
from keras.preprocessing import sequence
from keras.models import Sequential, load_model, Model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input


IMG_PATH = '' #Image path
LABEL_PATH = '/content/flickr30k_images/results.csv'
EMBEDDING_PATH = '/content/glove.6B.200d.txt'
MODEL_PATH = '/content/label.keras'
max_len = 34

model = load_model(MODEL_PATH)

with open('{}/{}'.format(os.path.dirname(EMBEDDING_PATH), 'vocabIndex.csv')) as csv_file:
    reader = csv.reader(csv_file)
    ixtoword = dict(reader)
ixtoword = {int(k): v for k, v in ixtoword.items()}
wordtoix = {v: int(k) for k, v in ixtoword.items()}
  
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
inception_model = InceptionV3(weights='imagenet')
# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(inception_model.input, inception_model.layers[-2].output)
  
def encode_sentence(sentence):
  for i in range(34):
    if (len(sentence) > i):
      if (sentence[i] in wordtoix.keys()):
        sentence[i] = wordtoix[sentence[i]]
    else: sentence.append(0)
  sentence = np.reshape(sentence, (1, len(sentence)))
  return sentence
sen = ['startseq']
img = encode(IMG_PATH)
img = np.reshape(img, (1, 2048))

for i in range(max_len):
  encoded_sen = encode_sentence(copy(sen))
  pred = model.predict([img, encoded_sen])[0]
  pred = np.argmax(pred)
  if pred == 17:
    break
  sen.append(ixtoword[pred])

sen = ' '.join(sen)
print(sen)
