from Topic_Models.topic_model import Topic_Model
import pandas as pd
from flask_app.classifier import Active_Learning
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Topic_Models.Neural_Topic_Model import Neural_Model
import pickle
from multiprocessing import Process, Manager
import copy
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csr_matrix, hstack
import time

'''
Mapping the mode numbers to which model we use
LA: active learning baseline
'''
model_types_map = {0: 'LA' , 1: 'LDA', 2: 'SLDA', 3: 'ETM', 4: 'CTM', 7: 'LLDA', 8: 'PLDA', 9: 'Bertopic'}

def save_results(lst, file_name):
    a_obj = np.array(lst, dtype=object)
    np.save(file_name, a_obj)

'''
Active learning baseline to test the metrics of clf without topic modeling
'''
def calculate_activelearning_diversity(process=False, unigram=False):
    accs, purity, ri, nmi = [], [], [], []
    test_accs, test_purity, test_ri, test_nmi = [], [], [], []
    test_df = pd.read_json(test_dataset_name)
    # print('test length is', len(test_df))

    '''
    Read the documents and the labels
    '''
    df = pd.read_json(doc_dir)
    labels = df.sub_labels.values.tolist()

    raw_texts = df.text.values.tolist()[0:training_length]
    test_texts = test_df.text.values.tolist()
    # raw_texts.extend(test_texts)


    '''
    Tried to test the performance of unigram and bigrams, the performance of bigram vectorizer is better
    '''
    if unigram == True:
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 1))
    else:
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))
    '''
    If process is True, load the processed data and use processed data as inputs for the clf
    else, just use the raw texts to as input for the clf. Results indicate that using raw texts
    has higher performance than using processed data
    '''
    if not process:
        vectorizer_idf = vectorizer.fit_transform(raw_texts)
        test_vecorizer = vectorizer.transform(test_texts)
    else:
        with open(processed_doc_dir, 'rb') as inp:
            loaded_train_data = pickle.load(inp)
            processed_train_data = loaded_train_data['datawords_nonstop']


        text_data = [' '.join(doc) for doc in processed_train_data]
        # from sklearn.feature_extraction.text import CountVectorizer
        # vectorizer = CountVectorizer(binary=True)
        vectorizer_idf = vectorizer.fit_transform(text_data)


    '''
    Initialize the active learning classifier
    '''
    alto = Active_Learning(raw_texts, None,  None, df, inference_alg, vectorizer_idf, training_length, 0, test_df, test_vecorizer)

    recommended_docs1 = []
    recommended_topics1 = []
    scores = []

    '''
    Label 300 hundred documents and save the three metrics
    '''
    for i in range(len(df)):
        '''
        Get the recommended document id and the score (entropy) for the current document
        '''
        recommend_id, score = alto.recommend_document(True)

        alto.label(recommend_id, labels[recommend_id])
        recommended_docs1.append(recommend_id)
        recommended_topics1.append(alto.last_recommended_topic)
        scores.append(score)
        # print(raw_texts[recommend_id])

        '''
        clf only works for at least 2 distinct classes
        '''
        if len(alto.classes) >= 3:
            train_acc, b, p, r, n, e, f, g = alto.eval_classifier()
            accs.append(train_acc)
            purity.append(p)
            ri.append(r)
            nmi.append(n)
            test_accs.append(b)
            test_purity.append(e)
            test_ri.append(f)
            test_nmi.append(g)

    return accs, purity, ri, nmi, test_accs, test_purity, test_ri , test_nmi


'''
Evaluate the performance of activalearning clf with topic model features added to the clf
'''
def calculate_topic_diversity(module, concat, num_topics, Model=None, topic_features_only=True):
    '''
    topic_features_only: If false, then concatenate TF-IDF with topic features
    '''
    accs, purity, ri, nmi = [], [], [], []
    test_accs, test_purity, test_ri, test_nmi = [], [], [], []
    test_df = pd.read_json(test_dataset_name)
    # print('test length is', len(test_df))

    if Model is None:
        if module == 3:
            model = Neural_Model('./Topic_Models/Model/ETM_{}.pkl'.format(num_topics), processed_doc_dir, doc_dir)
        elif module == 4:
            model = Neural_Model('./Topic_Models/Model/CTM_{}.pkl'.format(num_topics), processed_doc_dir, doc_dir)
        elif module == 9:
            model = Neural_Model('./Topic_Models/Model/Bertopic_{}.pkl'.format(num_topics), processed_doc_dir, doc_dir)
        else:
            model = Topic_Model(num_topics, 0, model_types_map[module], processed_doc_dir, training_length, {}, True, './Topic_Models/Model/{}_{}.pkl'.format(model_types_map[module], num_topics))
    else:
        '''
        If no module is selected, then use the Model passed in in the parameter
        '''
        model = Model

    '''
    Read texts, construct a text encoding for the clf
    '''
    df = pd.read_json(doc_dir)
    labels = df.sub_labels.values.tolist()
    raw_texts = df.text.values.tolist()[0:training_length]
    document_probas, doc_topic_probas = model.group_docs_to_topics()
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,1))
    test_texts = test_df.text.values.tolist()

    with open(test_processed_doc_dir, 'rb') as inp:
        loaded_test_data = pickle.load(inp)
        test_processed_data = loaded_test_data['datawords_nonstop']

    '''
    If concatenation is true, concatenate the topic distribution with the text features
    '''
    if concat == True:
        if topic_features_only == False:

            vectorizer_idf = vectorizer.fit_transform(raw_texts)

            '''
            concatenate the features from topic model with the clf
            '''
            concatenated_features = model.concatenate_features(model.doc_topic_probas, vectorizer_idf)

            vectorizer_idf = concatenated_features

            # test_vectorizer = vectorizer.transform(test_texts)

            # test_document_prob_dist = model.predict_topic_distribution_unseen_doc(test_processed_data)
        
            # test_vecotizer_concatenated_features = model.concatenate_features(test_document_prob_dist, test_vectorizer)

            # test_vecotizer_concatenated_features = csr_matrix(test_document_prob_dist).astype(np.float64)
        else:
            vectorizer_idf = csr_matrix(model.doc_topic_probas).astype(np.float64)

        

        test_vecotizer_concatenated_features = None
    else:
        model_concate_texts = raw_texts
        vectorizer_idf = vectorizer.fit_transform(model_concate_texts)



    alto = Active_Learning(raw_texts, copy.deepcopy(document_probas), doc_topic_probas, df, inference_alg, vectorizer_idf, training_length, 1, test_df, test_vecotizer_concatenated_features)

    print('start synthetic labeling')
    recommended_docs1 = []
    recommended_topics1 = []
    scores = []

    '''
    Label 300 documents
    '''
    for i in range(len(df)):
        recommend_id, score = alto.recommend_document(True)



        alto.label(recommend_id, labels[recommend_id])
        recommended_docs1.append(recommend_id)
        recommended_topics1.append(alto.last_recommended_topic)
        scores.append(score)
        # print(raw_texts[recommend_id])
        # print(alto.user_labels)
        if len(alto.classes) >= 3:
            train_acc, b, p, r, n, e, f, g = alto.eval_classifier()
            accs.append(train_acc)
            purity.append(p)
            ri.append(r)
            nmi.append(n)
            test_accs.append(b)
            test_purity.append(e)
            test_ri.append(f)
            test_nmi.append(g)

        '''
        Update and retrain supervised LDA based on available labels. set i%N == 0, where N
        is the number of documents you labled before you want to update sLDA
        '''
        if module == 2 and i != 0 and i % 20000 == 0:
            model = Topic_Model(num_topics, num_iter, model_types_map[module], processed_doc_dir, training_length, alto.user_labels, False, None)
            model.train('./Topic_Models//Model/SLDA_test.pkl')
            model = Topic_Model(num_topics, 0, model_types_map[module], processed_doc_dir, training_length, {}, True, './Topic_Models//Model/SLDA_test.pkl')
            document_probas, doc_topic_probas = model.group_docs_to_topics()
            alto.update_doc_probs(copy.deepcopy(document_probas), doc_topic_probas)
            vectorizer_idf = vectorizer.fit_transform(raw_texts)
            '''
            concatenate the features from topic model with the classifier
            '''
            concatenated_features = model.concatenate_features(vectorizer_idf)
            


            vectorizer_idf = concatenated_features
            alto.update_text_vectorizer(vectorizer_idf)


    return recommended_topics1, accs, purity, ri, nmi, test_accs, test_purity, test_ri , test_nmi


'''
Given a matrix, make its rows consistent length
'''
def make_consistent_rows(matrix):
    # Find the length of the shortest row
    min_length = min(len(row) for row in matrix)

    # Trim each row to the length of the shortest row
    a_uniform = [row[-min_length:] for row in matrix]
    a_np = np.array(a_uniform)
    return a_np


def run_experiment_and_save(module, save_path):
  '''
  module: model_types_map = {0: 'LA' , 1: 'LDA', 2: 'SLDA', 3: 'ETM', 4: 'CTM', 7: 'LLDA', 8: 'PLDA', 9: 'Bertopic'}
  Enter the module number to run the the model
  '''

  if module == 0:
    a0, p0, r0, n0, ta0, tp0, tr0, tn0 = calculate_activelearning_diversity(False, True)
    result_lst = [a0, p0, r0, n0, ta0, tp0, tr0, tn0]
    save_results(result_lst, save_path)
  else:
    recommended_topics1, a1, p1, r1, n1, ta1, tp1, tr1, tn1 = calculate_topic_diversity(module, True)
    result_lst = [a1, p1, r1, n1, ta1, tp1, tr1, tn1]
    print('purity list length is ', len(p1))
    save_results(result_lst, save_path)


'''
Run the experiment over n runs then take the medium of each run to 
save
'''
def run_experiment_over_n_runs(module, save_path, n_runs, num_topics, topic_features_only=True):
     a, p, r, n, ta, tp, tr, tn = [], [], [], [], [], [], [], []

     running_time = []
     if module == 0:
          for i in range(n_runs):
               start_time = time.time()
               a0, p0, r0, n0, ta0, tp0, tr0, tn0 = calculate_activelearning_diversity(False, True)
               end_time = time.time()
               a.append(a0)
               p.append(p0)
               r.append(r0)
               n.append(n0)
               ta.append(ta0)
               tp.append(tp0)
               tr.append(tr0)
               tn.append(tn0)
               running_time.append(end_time-start_time)
     else:
          for i in range(n_runs):
               start_time = time.time()
               recommended_topics1, a1, p1, r1, n1, ta1, tp1, tr1, tn1 = calculate_topic_diversity(module, True, num_topics ,None, topic_features_only=topic_features_only)
               end_time = time.time()

               a.append(a1)
               p.append(p1)
               r.append(r1)
               n.append(n1)
               ta.append(ta1)
               tp.append(tp1)
               tr.append(tr1)
               tn.append(tn1)
               running_time.append(end_time-start_time)
               print('finished iteration {} for model {} topic {}'.format(i, model_types_map[module], num_topics))
               time.sleep(5)

     a = make_consistent_rows(a)
     p = make_consistent_rows(p)
     r = make_consistent_rows(r)
     n = make_consistent_rows(n)
     ta = make_consistent_rows(ta)
     tp = make_consistent_rows(tp)
     tr = make_consistent_rows(tr)
     tn = make_consistent_rows(tn)

     result_lst = [a, p, r, n, ta, tp, tr, tn, [running_time]]
     save_results(result_lst, save_path)





'''
The processed document path and the raw documents path
'''

doc_dir = './Topic_Models/Data/newsgroups/newsgroup_test.json'
processed_doc_dir = './Topic_Models/Data/newsgroups/newsgroup_test_processed.pkl'



'''
num_iter: number of iterations to run when updating sLDA model
load_data: If use load_data, a trained topic model would be loaded.
'''
num_iter = 1000
load_data = True
save_model = False

'''
Enter the number of topics for the model you just trained
'''
inference_alg = 'logreg'
test_dataset_name = './Topic_Models/Data/newsgroups/newsgroup_test.json'
test_processed_doc_dir = './Topic_Models/Data/newsgroups/newsgroup_test_processed.pkl'

'''
Keep those and don't change
'''
USE_TEST_DATA = True
USE_PROCESSED_TEXT = False
CONCATENATE_KEYWORDS = True
table = pd.read_json(doc_dir)
training_length = len(table)
REGRESSOR_PREDICT = True
mode = 2


'''
Run the experiments on different topic models
'''
# run_experiment_over_n_runs(4, './model_testing_results/CTM_test_docs_5_runs_topics_features_only_{}_topics.npy'.format(16), 5, 16)

# num_topics = [16, 20, 30, 50, 100]

# for topic_num in num_topics:
#     if topic_num != 16:
#         run_experiment_over_n_runs(1, './model_testing_results/log_testing/LDA_test_docs_5_runs_topics_features_only_{}_topics.npy'.format(topic_num), 5, topic_num)
#         run_experiment_over_n_runs(1, './model_testing_results/log_testing/LDA_test_docs_5_runs_{}_topics.npy'.format(topic_num), 5, topic_num, topic_features_only=False)

#         run_experiment_over_n_runs(2, './model_testing_results/log_testing/SLDA_test_docs_5_runs_topics_features_only_{}_topics.npy'.format(topic_num), 5, topic_num)
#         run_experiment_over_n_runs(2, './model_testing_results/log_testing/SLDA_test_docs_5_runs_{}_topics.npy'.format(topic_num), 5, topic_num, topic_features_only=False)

#     run_experiment_over_n_runs(4, './model_testing_results/log_testing/CTM_test_docs_5_runs_topics_features_only_{}_topics.npy'.format(topic_num), 5, topic_num)
#     run_experiment_over_n_runs(4, './model_testing_results/log_testing/CTM_test_docs_5_runs_{}_topics.npy'.format(topic_num), 5, topic_num, topic_features_only=False)

topic_num = 16
run_experiment_over_n_runs(1, './model_testing_results/log_testing/LDA_test_docs_5_runs_topics_features_only_{}_topics.npy'.format(topic_num), 5, topic_num)
# run_experiment_over_n_runs(1, './model_testing_results/log_testing/LDA_test_docs_5_runs_{}_topics.npy'.format(topic_num), 5, topic_num, topic_features_only=False)
# run_experiment_over_n_runs(2, './model_testing_results/log_testing/SLDA_test_docs_5_runs_topics_features_only_{}_topics.npy'.format(topic_num), 5, topic_num)
# run_experiment_over_n_runs(2, './model_testing_results/log_testing/SLDA_test_docs_5_runs_{}_topics.npy'.format(topic_num), 5, topic_num, topic_features_only=False)