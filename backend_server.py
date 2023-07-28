# from spacy_topic_model import TopicModel
from topic_model import Topic_Model
import pandas as pd
from alto_session import NAITM
from sklearn.feature_extraction.text import TfidfVectorizer
from Neural_Topic_Model import Neural_Model
import pickle
from multiprocessing import Process, Manager
import copy
import random

# doc_dir = './Data/newsgroup_sub_500.json'
# processed_doc_dir = './Data/newsgroup_sub_500_processed.pkl'
# doc_dir = './Data/CongressionalBill/congressional_bills.json'
# processed_doc_dir = './Data/congressional_bill_processed.pkl'
# doc_dir = './Data/newsgroup_train.json'
# processed_doc_dir = './Data/newsgroup_train_processed.pkl'
doc_dir = './Data/congressional_bill_train.json'
processed_doc_dir = './Data/congressional_bill_train_processed.pkl'
model_types_map = {0: 'LA' , 1: 'LDA', 2: 'SLDA', 3: 'ETM'}
num_iter = 3000
load_data = True
save_model = False
load_model_path = './Model/{}_model_data.pkl'
num_topics =30
inference_alg = 'logreg'
test_dataset_name = './Data/congressional_bill_test.json'
USE_TEST_DATA = True
USE_PROCESSED_TEXT = False
CONCATENATE_KEYWORDS = True
table = pd.read_json(doc_dir)
training_length = len(table)



class User():
    def __init__(self, mode, user_id):
        self.mode = mode
        self.user_id = user_id
        self.df = pd.read_json(doc_dir)
        self.raw_texts = self.df.text.values.tolist()[0:training_length]
        self.test_df = None
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
        self.updated_model = False
        self.click_tracks = {}

        # self.vectorizer_idf = 
        if USE_TEST_DATA and not USE_PROCESSED_TEXT:
            self.test_df = pd.read_json(test_dataset_name)
            test_texts = self.test_df.text.values.tolist()
            self.raw_texts.extend(test_texts)
            self.vectorizer_idf = self.vectorizer.fit_transform(self.raw_texts)
        elif not USE_PROCESSED_TEXT and not USE_TEST_DATA:
            self.vectorizer_idf = self.vectorizer.fit_transform(self.df.text.values.tolist())

        # print('rws text length is', len(self.raw_texts))

        if USE_PROCESSED_TEXT:
            res_dataset_name = test_dataset_name.replace('.json', '_processed.json')
            self.test_df = pd.read_json(res_dataset_name)
            pickle_test = res_dataset_name.replace('json', 'pkl')
            with open(pickle_test, 'rb') as inp:
                loaded_test_data = pickle.load(inp)
                self.processed_test_data = loaded_test_data['datawords_nonstop']

        
        self.user_labels = set()

        if mode != 0:
            if mode == 1 or mode == 2:
                self.update_process = None
                self.model = Topic_Model(num_topics, 0, model_types_map[mode], processed_doc_dir, training_length, {}, True, './Model/{}_{}.pkl'.format(model_types_map[mode], num_topics))
                self.topics = self.model.print_topics(verbose=False)
                if USE_PROCESSED_TEXT:
                    self.vectorizer_idf = self.vectorizer.fit_transform(self.model.concatenate_keywords(self.topics, self.processed_test_data))

                if CONCATENATE_KEYWORDS:
                    concatenated_texts = self.model.concatenate_keywords_raw(self.topics, self.raw_texts[0:training_length])
                    concatenated_texts.extend(self.test_df.text.values.tolist())
                    self.vectorizer_idf = self.vectorizer.fit_transform(concatenated_texts)

                self.string_topics = {str(k): v for k, v in self.topics.items()}
                # print(self.string_topics)
            
                self.document_probas, self.doc_topic_probas = self.model.group_docs_to_topics()
                

                # self.word_topic_distributions = self.model.get_word_topic_distribution()
                self.word_topic_distributions = self.model.word_topic_distribution
                
                print('Mode {}'.format(model_types_map[mode]))
                # print(self.document_probas)

                self.alto = NAITM(self.raw_texts, copy.deepcopy(self.document_probas), self.doc_topic_probas, self.df, inference_alg, self.vectorizer_idf, training_length, 1, self.test_df)
            elif mode == 3:
                self.model = Neural_Model('./Model/ETM_{}.pkl'.format(num_topics), processed_doc_dir, doc_dir)
                self.topics = self.model.print_topics(verbose=False)

                if USE_PROCESSED_TEXT:
                    self.vectorizer_idf = self.vectorizer.fit_transform(self.model.concatenate_keywords(self.topics, self.processed_test_data))

                if CONCATENATE_KEYWORDS:
                    concatenated_texts = self.model.concatenate_keywords_raw(self.topics, self.raw_texts[0:training_length])
                    concatenated_texts.extend(self.test_df.text.values.tolist())
                    self.vectorizer_idf = self.vectorizer.fit_transform(concatenated_texts)


                self.string_topics = {str(k): v for k, v in self.topics.items()}
                # print(self.string_topics)
                self.document_probas, self.doc_topic_probas = self.model.document_probas, self.model.doc_topic_probas
                # self.model.get_topic_word_dist()

                self.alto = NAITM(self.raw_texts, copy.deepcopy(self.document_probas),  self.doc_topic_probas, self.df, inference_alg, self.vectorizer_idf, training_length, 1, self.test_df)
        else:
            if USE_PROCESSED_TEXT:
                with open(processed_doc_dir, 'rb') as inp:
                    self.loaded_data = pickle.load(inp)

                datawords_nonstop = self.loaded_data['datawords_nonstop']
                data_to_transform = [' '.join(doc) for doc in datawords_nonstop]
                self.vectorizer_idf = self.vectorizer.fit_transform(data_to_transform)

            # self.len_list = list(range(len(self.df)))
            len_list = list(range(len(self.df)))
            self.len_list = random.sample(len_list, len(len_list))
            self.alto = NAITM(self.raw_texts, None,  None, self.df, inference_alg, self.vectorizer_idf, training_length, 0, self.test_df)

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
                result['dropdown'] = dropdown
                # result['dropdowns'] = 
            else:
                result['prediction'] = ['']
                result['dropdown'] = []
            # print('result', result)

            return result
        elif self.mode ==0:
            preds, dropdown = self.alto.predict_label(int(doc_id))
            result['prediction'] = preds
            result['dropdown'] = dropdown

            topics = {"1": {"spans": [], "keywords": []}}
            result['topic'] = topics
            result['topic_order'] = {}

            return result

    def skip_doc(self):
        doc_id, _ = self.alto.recommend_document(True)
        
        return doc_id

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
            if len(self.user_labels) >= 2 and REGRESSOR_PREDICT:
                local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI = self.alto.eval_classifier()
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
            if len(self.user_labels) >= 2 and REGRESSOR_PREDICT:
                local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI = self.alto.eval_classifier()
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
            if len(self.user_labels) >= 2 and REGRESSOR_PREDICT:
                local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI = self.alto.eval_classifier()
                return local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI, result
            else:
                return -1, -1, -1, -1, -1, -1, -1, -1, result
            
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

    def update_slda(self):
        model = Topic_Model(num_topics, num_iter, model_types_map[self.mode], processed_doc_dir, training_length, self.alto.user_labels, False, None)
        model.train('./Model/SLDA_user{}.pkl'.format(self.user_id))
        # self.update_process.join()

    def round_trip1(self, label, doc_id, response_time):
        print('calling round trip')
        print('alto num docs labeld are', self.alto.num_docs_labeled)
        if model_types_map[self.mode] == 'SLDA':
            print('SLDA mode')
            
            if self.alto.num_docs_labeled >= 4:
                if self.update_process is None:
                    with Manager() as manager:
                        self.update_process = Process(target=self.update_slda)
                        self.update_process.start()
                elif not self.update_process.is_alive():
                    print('SLDA model is updated')
                    try:
                        self.model = Topic_Model(num_topics, 0, model_types_map[self.mode], processed_doc_dir, training_length, {}, True, './Model/SLDA_user{}.pkl'.format(self.user_id))
                        self.topics = self.model.print_topics(verbose=False)
                        if USE_PROCESSED_TEXT:
                            self.vectorizer_idf = self.vectorizer.fit_transform(self.model.concatenate_keywords(self.topics, self.processed_test_data))

                        self.string_topics = {str(k): v for k, v in self.topics.items()}
                        # print(self.string_topics)
                            
                        self.document_probas, self.doc_topic_probas = self.model.group_docs_to_topics()
                        # self.word_topic_distributions = self.model.get_word_topic_distribution()
                        self.word_topic_distributions = self.model.word_topic_distribution

                        if CONCATENATE_KEYWORDS:
                            concatenated_texts = self.model.concatenate_keywords_raw(self.topics, self.raw_texts[0:training_length])
                            concatenated_texts.extend(self.test_df.text.values.tolist())
                            self.vectorizer_idf = self.vectorizer.fit_transform(concatenated_texts)
                            self.alto.update_text_vectorizer(self.vectorizer_idf)

                        self.alto.update_doc_probs(copy.deepcopy(self.document_probas), self.doc_topic_probas)
                    except:
                        pass
                    with Manager() as manager:
                        self.update_process = Process(target=self.update_slda)
                        self.update_process.start()
            # elif self.alto.num_docs_labeled >= 5:
                
            

            # print('self topics are')
            # print(self.topics)
            result = self.sub_roundtrip(label, doc_id, response_time)
            return result
        else:
            return self.sub_roundtrip(label, doc_id, response_time)

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
            # random_document, _ = self.alto.recommend_document()
            # result['document_id'] = random_document
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
            # print('recommend document')
            # print(recommend_result)

            # print('recommended result')
            # print(recommend_result)
            result['document_id'] = random_document
            result['keywords'] = {}
            # random_document, _ = self.alto.recommend_document()
            # result['document_id'] = random_document
            # print('result')
            # print(result)

        return result
    
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
            # random_document, _ = self.alto.recommend_document()
            # result['document_id'] = random_document
        else:
            result = {}
            cluster = {}
            cluster["1"] = list(range(len(self.df)))
            # print(cluster)
            result['cluster'] = cluster
     
            result['keywords'] = {}
            # random_document, _ = self.alto.recommend_document()
            # result['document_id'] = random_document
            # print('result')
            # print(result)

        return result

    def get_metrics_to_save(self):
        try:
            local_training_acc, local_testing_preds, purity, RI, NMI, user_purity, user_RI, user_NMI = self.alto.eval_classifier()
            return purity, RI, NMI, user_purity, user_RI, user_NMI
        except:
            return -1, -1, -1, -1, -1, -1