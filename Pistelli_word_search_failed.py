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
    coo = W[vocab[word]]
    #create dictionary with the words and the euclidean distance between the given word and the words in the dictionary
    tmp = []
    for word, v in vectors.items():
        tmp = np.append(tmp, np.linalg.norm(coo - v))
    return tmp

# Main loop for analogy
while True:
    input_term = input("\nEnter three words (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        distances = distance(input_term)

        print(distances)
        sol = np.argsort(distances)

        #sol = sorted(sol.items(), key=lambda x: x[1])
        #sort the sol dictionary by the values

        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for x in range(3):
            print("%35s\t\t%f\n" % (ivocab[sol[x]], distances[sol[x]]))