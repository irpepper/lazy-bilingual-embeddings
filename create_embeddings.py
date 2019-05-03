import gensim
from pprint import pprint
import random
import argparse
import time

parser = argparse.ArgumentParser(description='Generate bilingual embeddings')
parser.add_argument("--lang1", type=str, default="data/Books.en-es.en", help="Language 1")
parser.add_argument("--lang2", type=str, default="data/Books.en-es.es", help="Language 2")
parser.add_argument("--l1", type=str, default="en", help="Language 1 Shorthand")
parser.add_argument("--l2", type=str, default="es", help="Language 2 Shorthand")
parser.add_argument("--method", type=int, default=2, help="Mixing method to use")
parser.add_argument("--window", type=int, default=3, help="Context window for word2vec training")
parser.add_argument("--dims", type=int, default=300, help="Embedding dimension size")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
params = parser.parse_args()


#Load in corpora
print("\n\n\nLoading Corpora")
t1 = time.time()
with open(params.lang1, encoding="utf-8") as f:
    lang1 = f.readlines()

with open(params.lang2, encoding="utf-8") as f:
    lang2 = f.readlines()

#Tokenize each corpus
lang1 = [gensim.utils.simple_preprocess(line) for line in lang1]
lang2 = [gensim.utils.simple_preprocess(line) for line in lang2]

#Get all word types in each corpus
lang1_words = list(set([word for sent in lang1 for word in sent]))
lang2_words = list(set([word for sent in lang2 for word in sent]))
print("-- {} -- seconds\n".format(time.time()-t1))

#Method 1 for mixing languages via word position
def mix1(l1, l2):
    output = []
    len_l1 = len(l1)
    len_l2 = len(l2)
    if not len_l1 or not len_l2:
        return []

    if len_l2 >  len_l1:
        for i in range(len_l1):
            new_line = []
            for j in range(len_l1):
                if j != i:
                    new_line.append(l1[j])
                else:
                    new_line.append(l2[j])
            output.append(new_line)
        output[-1] += l2[j+1:]

    if len_l1 >= len_l2:
        for i in range(len_l1):
            new_line = []
            for j in range(len_l1):
                if j < len_l2:
                    if j != i:
                        new_line.append(l1[j])
                    else:
                        new_line.append(l2[j])
                else:
                    if j != i:
                        new_line.append(l1[j])
                    else:
                        possible_words = l2[len_l2-j-1:]
                        if len(possible_words)>0:
                            x = random.randint(0,len(possible_words)-1)
                        else:
                            x = 0
                        try:
                            new_line.append(possible_words[x])
                        except:
                            print(possible_words)
                            print(l1)
                            print(l2)
            output.append(new_line)

    return output


#Method 2
def mix2(l1,l2):
    output = []
    if len(l1) > len(l2):
        new_line = []
        for i in range(len(l1)):
            if random.randint(0,1):
                new_line.append(l1[i])
            elif i < len(l2):
                new_line.append(l2[i])
            else:
                new_line.append(l1[i])
    else:
        new_line = []
        for i in range(len(l2)):
            if random.randint(0,1):
                new_line.append(l2[i])
            elif i < len(l1):
                new_line.append(l1[i])
            else:
                new_line.append(l2[i])
    return new_line


#Use one of the methods
print("Creating bilingual data set")
t1 = time.time()
if params.method == 1:
    data = []
    for i,j in zip(lang1,lang2):
        data += mix1(i,j)

elif params.method == 2:
    data = []
    for i,j in zip(lang1[2:],lang2[2:]):
        i_n = len(i)
        j_n = len(j)
        if i_n > 4 and j_n > 4:
            for k in range(int(max(i_n,j_n)**(0.5))*2):
                data.append(mix2(i,j))
print("-- {} -- seconds\n".format(time.time()-t1))

#Load word2vec model
print("Loading gensim.Word2Vec")
t1 = time.time()
model = gensim.models.Word2Vec(
        data,
        size=params.dims,
        window=params.window,
        min_count=2,
        workers=4)
print("-- {} -- seconds\n".format(time.time()-t1))


#Train the model
print("Training Model")
t1 = time.time()
model.train(data, total_examples=len(data), epochs=params.epochs)
print("-- {} -- seconds\n".format(time.time()-t1))


#Write to vector files
print("Writing Language 1")
t1 = time.time()
with open("{}{}-{}.{}.vec".format(str(params.method),params.l1,params.l2,params.l1), "w", encoding="utf-8") as f:
    f.write("{} {}\n".format(len(lang1_words),params.dims))
    for word in lang1_words:
        try:
            f.write(word + " " + str(model[word])[1:-1].replace(",","").replace("\n","") + "\n")
        except:
            pass
    f.close()
print("-- {} -- seconds\n".format(time.time()-t1))

print("Writing Language 2")
t1 = time.time()
with open("{}{}-{}.{}.vec".format(str(params.method),params.l1,params.l2,params.l2), "w", encoding="utf-8") as f:
    f.write("{} {}\n".format(len(lang2_words),params.dims))
    for word in lang2_words:
        try:
            f.write(word + " " + str(model[word])[1:-1].replace(",","").replace("\n","") + "\n")
        except:
            pass
    f.close()
print("-- {} -- seconds\n".format(time.time()-t1))
