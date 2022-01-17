import numpy as np
import pandas as pd
import string
import random
from copy import copy
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
