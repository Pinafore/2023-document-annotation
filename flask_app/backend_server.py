# from spacy_topic_model import TopicModel
import sys
sys.path.append('../')
from Topic_Models.topic_model import Topic_Model
from .classifier import Active_Learning
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Topic_Models.Neural_Topic_Model import Neural_Model
import pickle
from multiprocessing import Process, Manager
import copy
import random
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
topic_models_dir = os.path.dirname(current_dir)


'''
Replace it with the path of your dataset
'''
# doc_dir = os.path.join(topic_models_dir,'Topic_Models/Data/congressional_bill_train.json')
# processed_doc_dir = os.path.join(topic_models_dir,'Topic_Models/Data/congressional_bill_train_processed.pkl')
doc_dir = os.path.join(topic_models_dir,'Topic_Models/Data/congressional_bill_train.json')
processed_doc_dir = os.path.join(topic_models_dir,'Topic_Models/Data/congressional_bill_train_processed.pkl')
model_types_map = {0: 'LA' , 1: 'LDA', 2: 'SLDA', 3: 'CTM'}
num_iter = 1500
load_data = True
save_model = False
num_topics =30
inference_alg = 'logreg'
test_dataset_name = os.path.join(topic_models_dir,'Topic_Models/Data/congressional_bill_train_test_test.json')
USE_TEST_DATA = False
table = pd.read_json(doc_dir)
training_length = len(table)



class User():
    '''
    Initialize a user session with needed elements
    '''
    def __init__(self, mode, user_id):
        self.mode = mode
        self.user_id = user_id
        self.df = pd.read_json(doc_dir)
        self.raw_texts = self.df.text.values.tolist()[0:training_length]
        self.test_df = None
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,1))
        self.updated_model = False
        self.click_tracks = {}
        self.slda_update_freq = 0
        self.purity = -1
        self.RI = -1
        self.NMI = -1
        
        if USE_TEST_DATA:
            self.test_df = pd.read_json(test_dataset_name)
            test_texts = self.test_df.text.values.tolist()
            self.raw_texts.extend(test_texts)
            self.vectorizer_idf = self.vectorizer.fit_transform(self.raw_texts)
        else:
            self.vectorizer_idf = self.vectorizer.fit_transform(self.df.text.values.tolist())

        


            
        
        self.user_labels = set()

        if mode != 0:
            if mode == 1 or mode == 2:
                self.update_process = None
                self.model = Topic_Model(num_topics, 0, model_types_map[mode], processed_doc_dir, training_length, {}, True, os.path.join(topic_models_dir,'Topic_Models/Model/{}_{}.pkl'.format(model_types_map[mode], num_topics)))
                self.topics = self.model.print_topics(verbose=False)
               

                concatenated_features = self.model.concatenate_features(self.model.doc_topic_probas, self.vectorizer_idf)
                self.concatenated_features = concatenated_features

                self.string_topics = {str(k): v for k, v in self.topics.items()}
                
            
                self.document_probas, self.doc_topic_probas = self.model.group_docs_to_topics()
                

                
                # self.word_topic_distributions = self.model.word_topic_distribution
                
                print('Mode {}'.format(model_types_map[mode]))
                # print(self.document_probas)

                self.alto = Active_Learning(self.raw_texts, copy.deepcopy(self.document_probas), self.doc_topic_probas, self.df, inference_alg, self.concatenated_features, training_length, 1, self.test_df, None)
            elif mode == 3:
                self.model = Neural_Model(os.path.join(topic_models_dir,'Topic_Models/Model/{}_{}.pkl'.format(model_types_map[mode], num_topics)), processed_doc_dir, doc_dir)
                self.topics = self.model.print_topics(verbose=False)

              

                concatenated_features = self.model.concatenate_features(self.model.doc_topic_probas, self.vectorizer_idf)
                self.concatenated_features = concatenated_features


                self.string_topics = {str(k): v for k, v in self.topics.items()}
                # print(self.string_topics)
                self.document_probas, self.doc_topic_probas = self.model.document_probas, self.model.doc_topic_probas
                # self.word_topic_distributions = self.model.word_topic_distribution

                self.alto = Active_Learning(self.raw_texts, copy.deepcopy(self.document_probas),  self.doc_topic_probas, self.df, inference_alg, self.concatenated_features, training_length, 1, self.test_df, None)
        else:
            # self.len_list = list(range(len(self.df)))
            len_list = list(range(len(self.df)))
            self.len_list = random.sample(len_list, len(len_list))
            self.alto = Active_Learning(self.raw_texts, None,  None, self.df, inference_alg, self.vectorizer_idf, training_length, 0, self.test_df, None)


    '''
    Fetch the document information given a document id. 
    '''
    def get_doc_information(self, doc_id):
        result = dict()
        
        self.click_tracks[str(doc_id)] = 'click'

        if self.mode == 1 or self.mode == 2 or self.mode == 3:
            topic_distibution, topic_res_num = self.model.predict_doc_with_probs(int(doc_id), self.topics)
            result['topic_order'] = topic_distibution
            # result['topic_keywords'] = topic_keywords
            result['topic'] = self.model.get_word_span_prob(int(doc_id), topic_res_num, 0.001)

            if len(self.user_labels) >= 2:
                preds, dropdown = self.alto.predict_label(int(doc_id))
                result['prediction'] = preds
                # result['dropdown'] = dropdown
                # result['dropdowns'] = 
            else:
                result['prediction'] = 'No Prediction'
                # result['dropdown'] = []
            # print('result', result)

            return result
        elif self.mode ==0:
            preds, dropdown = self.alto.predict_label(int(doc_id))
            result['prediction'] = preds
            # result['dropdown'] = dropdown

            topics = {"1": {"spans": [], "keywords": []}}
            result['topic'] = topics
            result['topic_order'] = {}

            return result


    '''
    Skip the current recommended document and return the next recommended document
    '''
    def skip_doc(self):
        doc_id, _ = self.alto.recommend_document(True)
        
        return doc_id


    '''
    When the user enters a label for a document, this method automatically returns
    the next recommended document
    '''
    def sub_roundtrip(self, label, doc_id, response_time):
        result = dict()
        self.click_tracks[str(doc_id)] = 'label, recommended {}'.format(self.alto.last_recommended_doc_id)
        if self.mode == 2:
            if isinstance(label, str):
                print('calling self.label...')
                self.user_labels.add(label)
                self.alto.label(int(doc_id), label)
                        
            
            # print(self.topics)
            random_document, _ = self.alto.recommend_document(True)
             
            result['document_id'] = str(random_document)
                    
            # print(result)
            print('unique user labels length is {}'.format(len(self.user_labels)))
            if len(self.user_labels) >= 2:
                local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI = self.alto.eval_classifier()
                self.purity = purity
                self.RI = RI
                self.NMI = NMI
                return local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI, result
            else:
                return -1, -1, -1, -1, -1, -1, -1, -1, result
        elif self.mode == 1 or self.mode == 3:
            if isinstance(label, str):
                print('calling self.label...')
                self.user_labels.add(label)
                self.alto.label(int(doc_id), label)
                    
            
            # print(self.topics)
            random_document, _ = self.alto.recommend_document(True)
            
            # result['raw_text'] = str(random_document)
            result['document_id'] = str(random_document)
                    
                
            # print(result)
            print('unique user labels length is {}'.format(len(self.user_labels)))
            if len(self.user_labels) >= 2:
                local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI = self.alto.eval_classifier()
                self.purity = purity
                self.RI = RI
                self.NMI = NMI
                return local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI, result
            else:
                return -1, -1, -1, -1, -1, -1, -1, -1, result
        elif self.mode == 0:
            if isinstance(label, str):
                self.user_labels.add(label)
                self.alto.label(int(doc_id), label)
                    
        

            random_document, _ = self.alto.recommend_document(True)
            
            # result['raw_text'] = str(random_document)
            result['document_id'] = str(random_document)

            print('unique user labels length is {}'.format(len(self.user_labels)))
            if len(self.user_labels) >= 2:
                local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI = self.alto.eval_classifier()
                self.purity = purity
                self.RI = RI
                self.NMI = NMI
                return local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI, result
            else:
                return -1, -1, -1, -1, -1, -1, -1, -1, result

    '''
    Retrive the document informaton-topic orders, predictions from the classifier
    to save to the database
    '''      
    def get_doc_information_to_save(self, doc_id):
        result = dict()
        print('getting document information to save...')
        print('mode is ', self.mode)
        if self.mode == 1 or self.mode == 2 or self.mode == 3:
            
            topic_distibution, topic_res_num = self.model.predict_doc_with_probs(int(doc_id), self.topics)
            result['topic_order'] = topic_distibution
            # result['topic_keywords'] = topic_keywords
            result['topics'] = topic_res_num

            if len(self.user_labels) >= 2:
                preds, dropdown = self.alto.predict_label(int(doc_id))
                # result['dropdown'] = dropdown
                result['prediction'] = preds
            else:
                result['prediction'] =['No prediction']
                # result['dropdown'] = []

            return result
        elif self.mode ==0:
            if len(self.user_labels) >= 2:
                preds, dropdown = self.alto.predict_label(int(doc_id))
                result['prediction'] = preds
                # result['dropdown'] = dropdown
            else:
                result['prediction'] =['No prediction']
                # result['dropdown'] = dropdown
            
            result['topics'] = {}
            result['topic_order'] = {}
       
            return result

    '''
    Reinitialize a slda topic model and train it and save it.
    '''
    def update_slda(self):
        model = Topic_Model(num_topics, num_iter, model_types_map[self.mode], processed_doc_dir, training_length, self.alto.user_labels, False, None)
        model.train(os.path.join(topic_models_dir,'Topic_Models/Model/SLDA_user{}.pkl'.format(self.user_id)))
        # self.update_process.join()

    '''
    Load the updated SLDA model to the current process
    '''
    def load_updated_model(self):
        try:
            self.model = Topic_Model(num_topics, 0, model_types_map[self.mode], processed_doc_dir, training_length, {}, True, os.path.join(topic_models_dir,'Topic_Models/Model/SLDA_user{}.pkl'.format(self.user_id)))
            self.topics = self.model.print_topics(verbose=False)

            self.string_topics = {str(k): v for k, v in self.topics.items()}
            # print(self.string_topics)
                            
            self.document_probas, self.doc_topic_probas = self.model.group_docs_to_topics()
            
            # self.word_topic_distributions = self.model.word_topic_distribution

            concatenated_features = self.model.concatenate_features(self.model.doc_topic_probas, self.vectorizer_idf)
            self.concatenated_features = concatenated_features

            self.alto.update_text_vectorizer(self.concatenated_features)

            self.alto.update_doc_probs(copy.deepcopy(self.document_probas), self.doc_topic_probas)

            print('updated new SLDA model')
        except Exception as e:
            print(f"An error occurred loading: {e}")
            pass

        with Manager() as manager:
            self.update_process = Process(target=self.update_slda)
            self.update_process.start()

    '''
    When the user enters a label for a document, this method automatically returns
    the next recommended document. Also loads or train a new SLDA model
    '''
    def round_trip1(self, label, doc_id, response_time):
        # print('calling round trip')
        print('alto num docs labeld are', self.alto.num_docs_labeled)
        if model_types_map[self.mode] == 'SLDA':
            print('SLDA mode')
            if self.alto.num_docs_labeled >= 4:
                if self.update_process is None:
                    self.slda_update_freq += 1
                    with Manager() as manager:
                        self.update_process = Process(target=self.update_slda)
                        self.update_process.start()
                elif not self.update_process.is_alive():
                    '''
                    After SLDA model finishes updating in the backend, load it from the
                    saved directory. Then updates the sLDA again
                    '''
                    self.load_updated_model()
                    self.slda_update_freq += 1
                    with Manager() as manager:
                        self.update_process = Process(target=self.update_slda)
                        self.update_process.start()
            
            result = self.sub_roundtrip(label, doc_id, response_time)
            return result
        else:
            return self.sub_roundtrip(label, doc_id, response_time)


    '''
    Return a dictionary, where the keys represent the topics from the topic
    model. and the values are the documents associated with the topic number
    '''
    def get_document_topic_list(self, recommend_action):
        print('calling get document topic list')
        if self.mode == 1 or self.mode == 2 or self.mode == 3:
            document_probas = self.document_probas
            result = {}
            cluster = {}
            for k, v in document_probas.items():
                cluster[str(k)] = [ele[0] for ele in v if not self.alto.is_labeled(int(ele[0]))]

            result['cluster'] = cluster
            # self.doc_topic_distribution = cluster


            # print(recommend_result)
            if recommend_action:
                random_document, _ = self.alto.recommend_document(False)
            else:
                random_document = -1

            result['document_id'] = random_document
            result['keywords'] = self.string_topics
            
        else:
            result = {}
            cluster = {}
            cluster["1"] = self.len_list
            # print(cluster)
            result['cluster'] = cluster
            if recommend_action:
                random_document, _ = self.alto.recommend_document(False)
            else:
                random_document = -1
            
            result['document_id'] = random_document
            result['keywords'] = {}
           

        return result
    

    '''
    check whether the mode is just active learning or with topic model
    '''
    def check_active_list(self):
        print('calling check active list')
        if self.mode == 1 or self.mode == 2 or self.mode == 3:
            document_probas = self.document_probas
            result = {}
            cluster = {}
            for k, v in document_probas.items():
                cluster[str(k)] = [ele[0] for ele in v]

            result['cluster'] = cluster
            # self.doc_topic_distribution = cluster



            result['keywords'] = self.string_topics
            
        else:
            result = {}
            cluster = {}
            cluster["1"] = list(range(len(self.df)))
            # print(cluster)
            result['cluster'] = cluster
     
            result['keywords'] = {}
            

        return result


    '''
    Calculate the metrics and return them. Accuracy, purity, rand index, NMI
    '''
    def get_metrics_to_save(self):
        try:
            local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI = self.alto.eval_classifier()
            self.purity = purity
            self.RI = RI
            self.NMI = NMI
            return purity, RI, NMI, user_purity, user_RI, user_NMI
        except:
            return -1, -1, -1, -1, -1, -1