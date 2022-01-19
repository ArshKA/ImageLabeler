import csv
import numpy as np
from numpy import array
import string
from copy import copy
import os

def get_descriptions(LABEL_PATH, max_len):
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

def create_vocabulary(LABEL_PATH, EMBEDDING_PATH, max_len):
  descriptions = get_descriptions(LABEL_PATH, max_len)

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
    
  with open('{}/{}'.format(os.path.dirname(EMBEDDING_PATH), 'vocabIndex.csv'), 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in ixtoword.items():
       writer.writerow([key, value])
  return descriptions
