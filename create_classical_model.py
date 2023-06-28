import tomotopy as tp
from tomotopy.utils import Corpus
from gensim.utils import simple_preprocess
import pickle
import os
import re
import argparse
from topic_model import Topic_Model

import numpy as np
import pandas as pd
from pprint import pprint


    
def main():
    # __init__(self, num_topics, model_type, load_data_path, train_len)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_topics", help="number of topics",
                       type=int, default=20, required=False)
    argparser.add_argument("--num_iters", help="number of iterations",
                       type=int, default=1000, required=False)
    argparser.add_argument("--model_type", help="type of the model",
                       type=str, default='LDA', required=False)
    argparser.add_argument("--load_data_path", help="Whether we LOAD the data",
                       type=str, default='./Data/congressional_bill_processed.pkl', required=False)
    argparser.add_argument("--train_len", help="number of training samples",
                       type=int, default=500, required=False)

    args = argparser.parse_args()
    
    Model = Topic_Model(args.num_topics, args.num_iters, args.model_type, args.load_data_path, args.train_len, {}, False, None)
    save_path = './Model/{}_{}.pkl'.format(args.model_type, args.num_topics)
    Model.train(save_path)
    

if __name__ == "__main__":
    main()