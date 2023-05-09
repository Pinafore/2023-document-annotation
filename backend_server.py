from spacy_topic_model import TopicModel
import pandas as pd
import pandas as pd
from alto_session import NAITM
from sklearn.feature_extraction.text import TfidfVectorizer
from Neural_Topic_Model import Neural_Model
import pickle

doc_dir = './Data/newsgroup_sub_500.json'
processed_doc_dir = './Data/newsgroup_sub_500_processed.pkl'
model_types_map = {1: 'LDA', 2: 'SLDA', 3: 'ETM'}
num_iter = 1200
load_data = True
save_model = False
load_model_path = './Model/{}_model_data.pkl'
num_topics = 20
inference_alg = 'logreg'
test_dataset_name = './Data/newsgroup_sub_1000.json'
USE_TEST_DATA = True
USE_PROCESSED_TEXT = False
training_length = 500
REGRESSOR_PREDICT = False


class User():
    def __init__(self, mode):
        self.mode = mode
        self.df = pd.read_json(doc_dir)
        self.raw_texts = self.df.text.values.tolist()
        self.test_df = None
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
        
        if USE_PROCESSED_TEXT:
            with open(processed_doc_dir, 'rb') as inp:
                self.loaded_data = pickle.load(inp)

            self.vectorizer_idf = vectorizer.fit_transform(self.loaded_data['datawords_nonstop'])
            # self.vectorizer_idf = 
        elif USE_TEST_DATA:
            self.test_df = pd.read_json(test_dataset_name)
            self.vectorizer_idf = vectorizer.fit_transform(self.test_df['text'])
        else:
            self.vectorizer_idf = vectorizer.fit_transform(self.df.text.values.tolist())
        self.initial = True
        self.user_labels = set()

        if mode != 0:
            if mode == 1 or mode == 2:
                # self.model = TopicModel(corpus_path=doc_dir, model_type=model_types_map[mode], min_num_topics= 5, num_iters= num_iter, load_model=load_data, save_model= save_model, load_path=load_model_path.format(model_types_map[mode]), hypers = None)
                # self.model.preprocess(5, 100)

                # self.model.train(num_topics)
                self.model = TopicModel('./Model/{}_{}.pkl'.format(model_types_map[mode], num_topics), model_types_map[mode], doc_dir, num_topics)
                self.topics = self.model.print_topics(verbose=False)
                self.string_topics = {str(k): v for k, v in self.topics.items()}
                # print(self.string_topics)

                self.document_probas, self.doc_topic_probas = self.model.group_docs_to_topics()
                self.word_topic_distributions = self.model.get_word_topic_distribution()
                
                print('Mode {}'.format(model_types_map[mode]))
                # print(self.document_probas)

                self.alto = NAITM(self.raw_texts, self.document_probas,  self.doc_topic_probas, self.df, inference_alg, self.vectorizer_idf, training_length, 1, self.test_df)
            elif mode == 3:
                self.model = Neural_Model('./Model/ETM_{}.pkl'.format(num_topics), processed_doc_dir, doc_dir)
                self.topics = self.model.print_topics(verbose=False)
                self.string_topics = {str(k): v for k, v in self.topics.items()}
                # print(self.string_topics)
                self.document_probas, self.doc_topic_probas = self.model.document_probas, self.model.doc_topic_probas
                self.model.get_topic_word_dist()

                self.alto = NAITM(self.raw_texts, self.document_probas,  self.doc_topic_probas, self.df, inference_alg, self.vectorizer_idf, training_length, 1, self.test_df)
        else:
            self.alto = NAITM(self.raw_texts, None,  None, self.df, inference_alg, self.vectorizer_idf, training_length, 0, self.test_df)

    def get_doc_info(self, doc_id):
        result = dict()
        if self.mode == 1 or self.mode == 2 or self.mode == 3:
            # result['raw_text'] = self.raw_texts[random_document]
            topic_distibution, topic_res_num = self.model.predict_doc_with_probs(int(doc_id), self.topics)
            result['topic_order'] = topic_distibution
            # result['topic_keywords'] = topic_keywords
            result['topic'] = self.model.get_word_span_prob(int(doc_id), topic_res_num, 0.001)

            if len(self.user_labels) >= 2:
                result['prediction'] = self.alto.predict_label(int(doc_id))
            else:
                result['prediction'] ='Create at least 2 labels to start model suggestion'

            # if len(self.user_labels) < 2:
            #     self.user_labels.add(label)
            #     result['prediction'] = "Create at least two labels to start active learning"
            # else:
            #     result['prediction'] = random.sample(self.user_labels, 1)[0]
                
            
            # print(result)
            # print('unique user labels length is {}'.format(len(self.user_labels)))
            # if len(self.user_labels) >= 2 and REGRESSOR_PREDICT:
            #     local_training_acc, local_testing_preds, global_training_acc, global_testing_acc = self.alto.eval_classifier()
            #     return local_training_acc, local_testing_preds, global_training_acc, global_testing_acc, result
        
            return result
        elif self.mode ==1:
            # if len(self.user_labels) < 2:
            #     self.user_labels.add(label)
            #     result['prediction'] = "Create at least two labels to start active learning"
            # else:
            #     result['prediction'] = random.sample(self.user_labels, 1)[0]
            result['prediction'] = self.alto.predict_label(int(doc_id))
            topics = {"1": {"spans": [], "keywords": []}}
            result['topic'] = topics
            result['topic_order'] = {}
            # print('unique user labels length is {}'.format(len(self.user_labels)))
            # if len(self.user_labels) >= 2 and REGRESSOR_PREDICT:
            #     local_training_acc, local_testing_preds, global_training_acc, global_testing_acc = self.alto.eval_classifier()
            #     return local_training_acc, local_testing_preds, global_training_acc, global_testing_acc, result
        
            return result


    def round_trip1(self, label, doc_id, response_time):
        result = dict()
        if self.mode == 1 or self.mode == 2 or self.mode == 3:
            # if label:
            if not self.initial and isinstance(label, str):
                self.user_labels.add(label)

            if not self.initial and isinstance(label, str):
                print('calling self.label...')
                self.alto.label(int(doc_id), label)
                
            self.initial = False
            # print(self.topics)
            random_document, _ = self.alto.recommend_document()
            # result['raw_text'] = self.raw_texts[random_document]
            # result['raw_text'] = str(random_document)
            result['document_id'] = str(random_document)
            # topic_distibution, topic_res_num = self.model.predict_doc_with_probs(int(doc_id), self.topics)
            # result['topic_order'] = topic_distibution
            # # result['topic_keywords'] = topic_keywords
            # result['topic'] = self.model.get_word_span_prob(random_document, topic_res_num, 0.001)

            
            # result['prediction'] = self.alto.predict_label(random_document)

            # if len(self.user_labels) < 2:
            #     self.user_labels.add(label)
            #     result['prediction'] = "Create at least two labels to start active learning"
            # else:
            #     result['prediction'] = random.sample(self.user_labels, 1)[0]
                
            
            # print(result)
            print('unique user labels length is {}'.format(len(self.user_labels)))
            if len(self.user_labels) >= 2 and REGRESSOR_PREDICT:
                local_training_acc, local_testing_preds, global_training_acc, global_testing_acc = self.alto.eval_classifier()
                return local_training_acc, local_testing_preds, global_training_acc, global_testing_acc, result
        
            return -1, -1, -1, -1, result
        elif self.mode == 0:
            if not self.initial and isinstance(label, str):
                self.user_labels.add(label)

            if not self.initial and isinstance(label, str):
                self.alto.label(int(doc_id), label)
                
            self.initial = False

            random_document, _ = self.alto.recommend_document()
            # result['raw_text'] = self.raw_texts[random_document]
            # result['raw_text'] = str(random_document)
            result['document_id'] = str(random_document)

            # result['prediction'] = self.alto.predict_label(int(random_document))

            # if len(self.user_labels) < 2:
            #     self.user_labels.add(label)
            #     result['prediction'] = "Create at least two labels to start active learning"
            # else:
            #     result['prediction'] = random.sample(self.user_labels, 1)[0]

            # topics = {"1": {"spans": [], "keywords": []}}
            # result['topic'] = topics
            # result['topic_order'] = {}
            print('unique user labels length is {}'.format(len(self.user_labels)))
            if len(self.user_labels) >= 2 and REGRESSOR_PREDICT:
                local_training_acc, local_testing_preds, global_training_acc, global_testing_acc = self.alto.eval_classifier()
                return local_training_acc, local_testing_preds, global_training_acc, global_testing_acc, result
        
            return -1, -1, -1, -1, result

    def get_document_topic_list(self):
        if self.mode == 1 or self.mode == 2 or self.mode == 3:
            document_probas = self.model.document_probas
            result = {}
            cluster = {}
            for k, v in document_probas.items():
                cluster[str(k)] = [ele[0] for ele in v]

            result['cluster'] = cluster
            self.doc_topic_distribution = cluster


            recommend_result = self.round_trip1('start', 13, "3")
            # print(recommend_result)

            result['document_id'] = recommend_result[4]['document_id']
            result['keywords'] = self.string_topics
            # random_document, _ = self.alto.recommend_document()
            # result['document_id'] = random_document
        else:
            result = {}
            cluster = {}
            cluster["1"] = list(range(len(self.df)))
            result['cluster'] = cluster
            recommend_result = self.round_trip1('start', 13, "3")
            # print(recommend_result)

            result['document_id'] = recommend_result[4]['document_id']
            result['keywords'] = {}
            # random_document, _ = self.alto.recommend_document()
            # result['document_id'] = random_document


        return result