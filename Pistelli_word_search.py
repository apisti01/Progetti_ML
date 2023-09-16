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
def distance(word):
    first = [999, 'lol']
    second = [999, 'lol']
    third = [999, 'lol']
    
    coo = W[vocab[word]]

    for word, v in vectors.items():
        tmp = np.linalg.norm(coo - v)
        if tmp < first[0]:
            third = second
            second = first
            first = [tmp, word]
        elif tmp < second[0]:
            third = second
            second = [tmp, word]
        elif tmp < third[0]:
            third = [tmp, word]
    
    return first, second, third


# Main loop for analogy
while True:
    input_term = input("\n Enter a word (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        distances = distance(input_term)

        
        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for x in range(3):
            print("%35s\t\t%f\n" % (distances[x][1], distances[x][0]))