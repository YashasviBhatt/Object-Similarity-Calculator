# Importing the Libraries
import numpy as np
import pandas as pd

def cosine_similarity(u, v):
    '''
    :param u: First Vector - Numpy Array
    :param v: Second Vector - Numpy Array
    :return: Cosine Similarity between 2 Vectors
    '''

    # Computing Dot Product between the 2 Vectors
    dot = np.dot(u, v)

    # Computing Norm of u
    norm_u = np.sqrt(np.sum(u ** 2))

    # Computing Norm of v
    norm_v = np.sqrt(np.sum(v ** 2))

    # Computing Dot Product of both the Norms
    norm_dot = np.dot(norm_u, norm_v)

    # Calculating Cosine Similarity
    cosine_similarity = dot / norm_dot

    return '{}'.format(round(cosine_similarity * 100, 2))


# Importing the Dataset
df = pd.read_csv('data.csv')

# Encoding the Variables
enc = {col : {'No' : 0, 'Yes' : 1} for col in df.columns}
df = df.replace(enc)

# Converting the Dataframe into Dictionary
df = df.set_index('Class').T.to_dict('list')
for key in df:
    df[key] = np.array(df[key])

# Setting up Constants
FATHER = df['Father']
MOTHER = df['Mother']
BOY = df['Boy']
GIRL = df['Girl']
BALL = df['Ball']
CROCODILE = df['Crocodile']
FRANCE = df['France']
PARIS = df['Paris']
ROME = df['Rome']
ITALY = df['Italy']

# Finding Cosine Similarities between Objects
print('Similarity % Between Father and Mother :', cosine_similarity(FATHER, MOTHER))
print('Similarity % Between Father and Boy :', cosine_similarity(FATHER, BOY))
print('Similarity % Between Mother and Girl :', cosine_similarity(MOTHER, GIRL))
print('Similarity % Between Mother and Boy :', cosine_similarity(MOTHER, BOY))
print('Similarity % Between Ball and Crocodile :', cosine_similarity(BALL, CROCODILE))
print('Similarity % Between Boy and Crocodile :', cosine_similarity(BOY, CROCODILE))
print('Similarity % Between France and Italy :', cosine_similarity(FRANCE, ITALY))
print('Similarity % Between Paris and Italy :', cosine_similarity(PARIS, ITALY))
print('Similarity % Between Ball and France :', cosine_similarity(BALL, FRANCE))