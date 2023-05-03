import tomotopy as tp
from tomotopy.utils import Corpus
from gensim.utils import simple_preprocess
import pickle
import os
import re
import gensim.corpora as corpora

import numpy as np
import pandas as pd
from pprint import pprint

# spacy for lemmatization
import spacy
import nltk
from nltk.corpus import stopwords

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

class Classical_Model():
    def __init__(self, model_path, data_path, dataset_dir):
         with open(data_path, 'rb') as inp:
            self.loaded_data = pickle.load(inp)

        