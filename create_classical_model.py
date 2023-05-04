import tomotopy as tp
from tomotopy.utils import Corpus
from gensim.utils import simple_preprocess
import pickle
import os
import re
import argparse


import numpy as np
import pandas as pd
from pprint import pprint



class Create_Model():
    def __init__(self, num_topics, num_iters, model_type, load_data_path, train_len):
        self.num_topics = num_topics
        self.model_type = model_type
        self.load_data_path = load_data_path
        self.train_length = train_len
        self.num_iters = num_iters

    def train(self, save_data_path):

        # print('training...')
        print('num topics:', self.num_topics)
        with open(self.load_data_path, 'rb') as inp:
            saved_data = pickle.load(inp)

    
        datawords_nonstop = saved_data['datawords_nonstop']
        spans = saved_data['spans']

        corpus = Corpus()
        labels = saved_data['labels']
        label_set = list(set(labels))
        if np.nan in label_set:
            label_set.remove(np.nan)
        if None in label_set:
            label_set.remove(None)
        if 'None' in label_set:
            label_set.remove('None')

        label_dict = dict()
        for i,label in enumerate(label_set):
            label_dict[label] = i
        if self.model_type == 'LLDA':
            print('Enumerating grams...')
            for i, ngrams in enumerate(self.datawords_nonstop):
                if self.labels and not self.labels[i] == 'None':
                    label = labels[i]
                    corpus.add_doc(ngrams, labels=[label])
                else:
                    corpus.add_doc(ngrams)
        elif self.model_type == 'LDA':
            '''
            Change something here
            '''
            for i, ngrams in enumerate(datawords_nonstop):
                corpus.add_doc(ngrams)
        elif self.model_type == 'SLDA':
            for i, ngrams in enumerate(datawords_nonstop):
                y = [0 for _ in range(len(label_set))]
                null_y = [np.nan for _ in range(len(label_set))]
                        
                if labels and not labels[i] == 'None':
                    label = labels[i]
                    y[label_dict[label]] = 1
                    corpus.add_doc(ngrams, y=y)
                    # print(y)
                else:
                    corpus.add_doc(ngrams, y=null_y)
                            
        else:
            raise Exception("unsupported model type!")

        if self.model_type == 'LLDA':
            print('Created LLDA model')
            mdl = tp.LLDAModel(k=self.num_topics)
        elif self.model_type == 'SLDA':
            print('Created SLDA model')
            # print('Getting into SLDA...')
            mdl = tp.SLDAModel(k=self.num_topics, vars=['b' for _ in range(len(label_set))], glm_param= [1.1 for i in range(len(label_set))], nu_sq = [5 for i in range(len(label_set))])
            # mdl = tp.SLDAModel(k=num_topics, vars=self.label_set)
                    
        elif self.model_type == 'LDA':
            print('Created LDA model')
            mdl = tp.LDAModel(k=self.num_topics, alpha =0.05, eta=0.1)

        mdl.add_corpus(corpus)

        print('starting training...')

        # print('total # topics {}'.format(mdl.k))
        for i in range(0, self.num_iters, 10):
            # print('training iter {}'.format(i))
            mdl.train(10)
            print(f'Iteration: {i}, Log-likelihood: {mdl.ll_per_word}, Perplexity: {mdl.perplexity}')
        # mdl.train(self.num_iters)
        self.lda_model = mdl
        assert len(corpus) == len(saved_data['texts'])
       

        # Instantiate a coherence model with the topic-word distribution, the corpus, and the dictionary
        coherence_model = tp.coherence.Coherence(
            corpus=mdl, coherence="c_v", top_n=10
        )

        print(coherence_model.get_score())
        '''
        Make documents for normal LDA
        '''
        self.maked_docs = []
        for doc in datawords_nonstop:
            curr_doc = mdl.make_doc(doc)
            self.maked_docs.append(curr_doc)


        # print(len(self.maked_docs))
        document_probas, doc_topic_probas = self.group_docs_to_topics()
        result = {}
        # print(document_probas)

        mdl.save(save_data_path.replace('pkl', 'bin'))
        result['document_probas'] = document_probas
        result['doc_topic_probas'] = doc_topic_probas
        result['spans'] = spans
        result['datawords_nonstop'] = saved_data['datawords_nonstop']
        result['texts'] = saved_data['texts']
        # result['get_document_topic_dist'] = doc_topic
        with open(save_data_path, 'wb+') as outp:
            pickle.dump(result, outp)


    def group_docs_to_topics(self):
        doc_prob_topic = []
        doc_to_topics, topics_probs = {}, {}
        for doc_id, doc in enumerate(self.maked_docs[0:self.train_length]):
            if self.model_type == 'SLDA':
                inferred = self.lda_model.estimate(doc)
            else:
                inferred, _ = self.lda_model.infer(doc)

            doc_topics = list(enumerate(inferred))
            doc_prob_topic.append(inferred)
                    

            # Infer the top three topics of the document
            doc_topics.sort(key = lambda a: a[1], reverse= True)
                    
            # print('doc topics {}'.format(doc_topics))
            doc_to_topics[doc_id] = doc_topics

            '''
            doc_topics[0][0] is the topic id. doc_topics[0][1] is doc id probability.
            '''
            if doc_topics[0][0] in topics_probs:
                topics_probs[doc_topics[0][0]].append((doc_id, doc_topics[0][1]))
            else:
                topics_probs[doc_topics[0][0]] = [(doc_id, doc_topics[0][1])]


        '''
        Sort the documents by topics based on their probability in descending order
        '''
        for k, v in topics_probs.items():
            topics_probs[k].sort(key = lambda a: a[1], reverse= True)

        return topics_probs, doc_prob_topic
    
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
                       type=str, default='./Data/newsgroup_sub_500_processed.pkl', required=False)
    argparser.add_argument("--train_len", help="number of training samples",
                       type=int, default=500, required=False)

    args = argparser.parse_args()
    
    Model = Create_Model(args.num_topics, args.num_iters, args.model_type, args.load_data_path, args.train_len)
    save_path = './Model/{}_{}.pkl'.format(args.model_type, args.num_topics)
    Model.train(save_path)
    

if __name__ == "__main__":
    main()