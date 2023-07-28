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


# sys.path.append('../home/ec2-user/AlexaSimbotActionInferenceModelWrapper/TopicModelBack')
doc_dir = './Data/newsgroup_sub_500.json'
# doc_dir = './Data/newsgroup_sub_2000.json'
model_type = 'SLDA'
num_iter = 50
load_data = True
save_model = False
load_model_path = './Model/LDA_model_data.pkl'
num_topics = 20
inference_alg = 'logreg'



# import os.path
# doc_dir = './Data/Nist_all_labeled.json'
# check_file = os.path.isfile(doc_dir)
# print(os.getcwd())  
# print(check_file)

class Session():
    def __init__(self):
        self.df = pd.read_json(doc_dir)
        self.raw_texts = self.df.text.values.tolist()
        self.model = TopicModel(corpus_path=doc_dir, model_type=model_type, min_num_topics= 5, num_iters= num_iter, load_model=load_data, save_model= save_model, load_path=load_model_path, hypers = None)
        self.model.preprocess(5, 100)

        self.model.train(num_topics)
        self.topics = self.model.print_topics(verbose=False)
        self.document_probas, self.doc_topic_probas = self.model.group_docs_to_topics()
        self.word_topic_distributions = self.model.get_word_topic_distribution()
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
        self.vectorizer_idf = vectorizer.fit_transform(self.df.text.values.tolist())
        self.alto = NAITM(self.model.get_texts(), self.document_probas,  self.doc_topic_probas, self.df, inference_alg, self.vectorizer_idf, len(self.topics), 500)
        

    def get_all_doc(self):
        result = dict()
        result['Documents'] = dict()
        # result['Documents'] = self.df.text.values.tolist()
        result['Topics'] = dict()
        text_lst = self.df.text.values.tolist()
        for i in range(len(text_lst)):
            result['Documents'][str(i)] = text_lst[i]
        
        if model_type == 'LDA':
            result['Topics'] = self.topics
        else:
            for k, v in self.topics.items():
                result['Topics'][str(k)] = v[0]
        
        # result['Probable_Topic'] = self.document_probas
        result['Probable_Topic'] = dict()
        for k, v in self.document_probas.items():
            result['Probable_Topic'][str(k)] = [ele[0] for ele in v]


        return result

    def highlight_doc(self):
        random_document, _ = self.alto.recommend_document()
        # inferred_topics = self.model.predict_doc(random_document, 10)

        result = dict()
        result['Document_id'] = random_document
        # result['Topics'] = dict()
        # for num, prob in inferred_topics:
        #     result['Topics'][str(num)] = self.topics[num][0]

        # result['highlights'] = [(0, 5)]
        return result
    
    # def label_and_recommend(self, new_label, doc_id):


    def label_doc(self, user_input, new_label, doc_id):
        if user_input >= 0 and user_input < len(self.topics):
            self.alto.label(doc_id, user_input)
            # doc_count += 1
        else:
            self.alto.label(doc_id, user_input)


    def get_topics(self, doc_id):
        inferred_topics = self.model.predict_doc(doc_id, num_topics)
        result = dict()

        if model_type == 'LDA':
            result['topics'] = [str(num) for num, prob in inferred_topics]
        else:
            result['topics'] = [str(num) for num, prob in inferred_topics]
        
        result['highlights'] = [(0, 5)]
        
        return result
    
    def round_trip1(self, label, new_label, doc_id, response_time, recommend_val):
        result = dict()
        # print(new_label == 'None')
        # print(label)
        # print(self.topics[19])
        if new_label == 'None' and label:
            if recommend_val == -1:
                self.alto.label(int(doc_id), label)
                
            random_document, _ = self.alto.recommend_document()
            result['raw_text'] = self.raw_texts[random_document]
            result['document_id'] = random_document
            topic_distibution, topic_keywords, topic_res_num = self.model.predict_doc_with_probs(int(doc_id), self.topics)
            result['topic_order'] = topic_distibution
            # result['topic_keywords'] = topic_keywords
            result['topic'] = self.model.get_word_span_prob(random_document, topic_res_num, 0.001)
            


        return result

    

# new_sess = Session()
# # new_sess.__init__()
# new_sess.get_topics(10)
# res = new_sess.highlight_doc()
# new_sess.label(468, 7)

# res = new_sess.get_all_doc()
# print(res['Probable_Topic'])

# # print(res)
# with open("./Data/random_shoe.json", "w") as outfile:
#     json.dump(res, outfile)
# print(res['Topics'].keys())
