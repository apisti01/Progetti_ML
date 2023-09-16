import random
import numpy as np

vocabulary_file='word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)

# fujnction to calculate distance from a word
def distance(coo, input_list):
    first = [999, 'lol']
    second = [999, 'lol']
    
    for word, v in vectors.items():
        tmp = np.linalg.norm(coo - v)
        if tmp < first[0] and word not in input_list:
            second = first
            first = [tmp, word]
        elif tmp < second[0] and word not in input_list:
            second = [tmp, word]
    
    return first, second

# Main loop for analogy
while True:
    input_term = input("\n Enter three words(EXIT to break): ").split(' ')
    if input_term[0] == 'EXIT':
        break
    else:
        coordinates = [0, 0, 0]
        for x in range(3):
            coordinates[x] = W[vocab[input_term[x]]]
        
        target = coordinates[2] + (coordinates[1] - coordinates[0])

        distances = distance(target, input_term)


        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for x in range(2):
            print("%35s\t\t%f\n" % (distances[x][1], distances[x][0]))