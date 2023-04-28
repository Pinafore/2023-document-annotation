import random
from scipy.sparse import vstack
import numpy as np
import json
import pandas as pd
import os
import pickle
import argparse
from alto_session import NAITM
from sklearn.feature_extraction.text import TfidfVectorizer

class Neural_Model():
    def __init__(self, model_path, data_path):
        with open(model_path, 'rb') as inp:
            self.loaded_data = pickle.load(inp)

        self.model = self.loaded_data['model']
        self.document_probas = self.loaded_data['document_probas']
        self.doc_topic_probas = self.loaded_data['doc_topic_probas']


    