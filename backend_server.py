from spacy_topic_model import TopicModel
import json
import time
import pandas as pd
import argparse
import random
import sys
import pandas as pd
import pickle
from alto_session import NAITM
from sklearn.feature_extraction.text import TfidfVectorizer
from Neural_Topic_Model import Neural_Model

doc_dir = './Data/newsgroup_sub_500.json'
model_types_map = {1: 'LDA', 2: 'SLDA', 3: 'ETM'}
num_iter = 600
load_data = True
save_model = False
load_model_path = './Model/{}_model_data.pkl'
num_topics = 20
inference_alg = 'logreg'

class Session():
    def __init__(self, mode):
        self.mode = mode
        self.df = pd.read_json(doc_dir)
        self.raw_texts = self.df.text.values.tolist()
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
        self.vectorizer_idf = vectorizer.fit_transform(self.df.text.values.tolist())
        self.initial = True

        if mode != 0:
            if mode == 1 or mode == 2:
                self.model = TopicModel(corpus_path=doc_dir, model_type=model_types_map[mode], min_num_topics= 5, num_iters= num_iter, load_model=load_data, save_model= save_model, load_path=load_model_path.format(model_types_map[mode]), hypers = None)
                self.model.preprocess(5, 100)

                self.model.train(num_topics)
                self.topics = self.model.print_topics(verbose=False)
                self.document_probas, self.doc_topic_probas = self.model.group_docs_to_topics()
                # self.word_topic_distributions = self.model.get_word_topic_distribution()
                
                self.alto = NAITM(self.raw_texts, self.document_probas,  self.doc_topic_probas, self.df, inference_alg, self.vectorizer_idf, 500, 1)
            elif mode == 3:
                self.model = Neural_Model('./Model/ETM_{}.pkl'.format(num_topics), doc_dir)
                self.document_probas, self.doc_topic_probas = self.model.document_probas, self.model.doc_topic_probas
                self.alto = NAITM(self.raw_texts, self.document_probas,  self.doc_topic_probas, self.df, inference_alg, self.vectorizer_idf, 500, 1)
        else:
            self.alto = NAITM(self.raw_texts, None,  None, self.df, inference_alg, self.vectorizer_idf, 500, 0)

    def round_trip1(self, label, doc_id, response_time):
        result = dict()
        if self.mode == 1 or self.mode == 2 or self.mode == 3:
            # if label:
            if not self.initial and isinstance(label, str):
                self.alto.label(int(doc_id), label)
                
            self.initial = False
            # print(self.topics)
            random_document, _ = self.alto.recommend_document()
            result['raw_text'] = self.raw_texts[random_document]
            result['document_id'] = str(random_document)
            topic_distibution, topic_res_num = self.model.predict_doc_with_probs(int(doc_id), self.topics)
            result['topic_order'] = topic_distibution
            # result['topic_keywords'] = topic_keywords
            result['topic'] = self.model.get_word_span_prob(random_document, topic_res_num, 0.001)
            result['prediction'] = 'sport'
            
            # print(result)
        
            return result
        elif self.mode == 0:
            if not self.initial and isinstance(label, str):
                self.alto.label(int(doc_id), label)
                
            self.initial = False

            random_document, _ = self.alto.recommend_document()
            result['raw_text'] = self.raw_texts[random_document]
            result['document_id'] = str(random_document)
            result['prediction'] = 'sport'

            topics = {"1": {"spans": [], "keywords": []}}
            result['topic'] = topics

            return result
