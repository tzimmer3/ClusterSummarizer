# Import Packages
#import numpy as np
#import pandas as pd
from joblib import load
#from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity



# ================== #
#  Get Embeddings
# ================== #

def get_embeddings(text=None, model=None):
    """
    Generate embeddings on a string of text.
    """
    if model==None:
        model = load('./model/SentBERTmodel.pkl')

    return model.encode(text)