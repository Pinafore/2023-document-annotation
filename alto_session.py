import random
from sklearn.linear_model import SGDClassifier
from User import User
from scipy.sparse import vstack
import numpy as np

# This var means if True, when labeling the same doc, the next recommend doc would 
# be the same
switch_doc = True

#NATM stands for neural interactive active topic modeling
class NAITM():
    def __init__(self, documents, doc_prob, doc_topic_prob, df, running_type, text_vectorizer, train_len, mode):
        """
        Documents: the text corpus
        Doc_prob: a dictionary contains all the topics. Each topic 
        contains a list of documents along with their probabilities 
        belong to a specific topic
        [topic 1]: [(document a, prob 1), (document b, prob 2)...]
        """
        # self.classes = list(range(num_topics))
        # print('mode is {}'.format(mode))
        self.mode = mode
        self.classes = []
        self.docs = documents
        self.num_docs_labeled = 0
        self.last_recommended_topic = None
        self.recommended_doc_ids = set()
        self.text_vectorizer = text_vectorizer
        
        self.train_length = train_len
        self.id_vectorizer_map = {}
        '''
        stores the scores of the documents after the user updates the classifier
        '''
        self.scores = []


        '''
        user_labels: labels the user creates for documents the user views
        curr_label_num: topic numbers computed by LDA
        '''
        self.user_labels = dict()

        '''
        Parameters or regressor can be changed later
        '''
        self.classifier = self.initialize_classifier(running_type)

        # Stores and track labeled documents and labels
        self.documents_track = None
        self.labels_track = []

        self.simulate_user_data = dict()
        self.user_label_number_map = {}

        '''
        Mode 1 mean using topic modeling. Otherwise, just use active learning
        '''
        if self.mode == 1:
            self.doc_topic_prob = np.array(doc_topic_prob)
            self.doc_probs = doc_prob
            '''
            get the median robability of every topic
            '''
            self.median_pro = dict()
            self.cal_median_topics()
            # self.curr_label_num = list(doc_prob.keys())

            self.existing_labels = df['label'].tolist()
        

    def initialize_classifier(self, classifier_type: str):
        classifier_type = classifier_type.lower()

        if classifier_type == 'logreg':
            return SGDClassifier(loss="log", penalty="l2", max_iter=1000, tol=1e-3, random_state=42, learning_rate="adaptive", eta0=0.1, validation_fraction=0.2)
        elif classifier_type == 'logreg_modal':
            print('Implement other types')
            return None
        else:
            print('Implement other types')
            return None

    def cal_median_topics(self):
        for k, v in self.doc_probs.items():
            self.median_pro[k] = self.get_median_pro(v)

    def get_median_pro(self, lst):
        n = len(lst)

        if n % 2 == 0:
            return (lst[n//2][1] + lst[n//2-1][1]) / 2
        else:
            return lst[n//2][1]
     
    def update_median_prob(self, topic_num):
        try:
            # print('Poping the first document...')
            # self.median_pro[topic_num].pop(0)
            self.doc_probs[topic_num].pop(0)
            self.cal_median_topics()
        except:
            pass
        

    '''
    Preference function can be changed or chosen for NAITM
    '''
    def preference(self):
        if self.mode == 1:
            if len(self.classes) < 2:
                print('-----------')
                print('classes smaller than 2')
                print('-----------')
                # print('median pro is {}'.format(self.median_pro))
                

                probs = list(self.median_pro.values())

                max_topic_idx = probs.index(max(probs))
                
                # if self.last_recommended_topic == max_topic_idx:
                #     # probs.pop(max_topic_idx)
                #     probs[max_topic_idx] = -1
                #     max_topic_idx = probs.index(max(probs))
                print('starting while loop')
                
                i = 0
                # while self.doc_probs[max_topic_idx][0][0] in self.recommended_doc_ids and i < len(self.classes):
                while self.doc_probs[max_topic_idx][0][0] in self.recommended_doc_ids and i < self.train_length: 
                    probs[max_topic_idx] = -1
                    max_topic_idx = probs.index(max(probs))
                    i += 1

                print('finish while loop')

                self.last_recommended_topic = max_topic_idx

                '''
                return the document id and its probability belong to the topic
                '''

                # print('current topic id is {}'.format(max_topic_idx))
                self.recommended_doc_ids.add(self.doc_probs[max_topic_idx][0][0])

                print('all recommended ids')
                print(self.recommended_doc_ids)
                return self.doc_probs[max_topic_idx][0][0], -1
            else:
                max_idx = np.argmax(self.scores)

                '''
                Don't show the users an already shown document
                '''
                while max_idx in self.recommended_doc_ids:
                    self.scores = np.delete(self.scores, max_idx)
                    try:
                        max_idx = np.argmax(self.scores)
                    except:
                        print('current len of the score list is {}'.format(len(self.scores)))
                

                print('Classifier in progess...')
                print('\033[1mScore of the current document is {}\033[0m'.format(self.scores[max_idx]))
                self.recommended_doc_ids.add(max_idx)
                return max_idx, self.scores[max_idx]
        else:
            if len(self.classes) < 2:
                print('-----------')
                print('num classes smaller than 2')
                print('-----------')
                # print('median pro is {}'.format(self.median_pro))
                random_doc_id = random.randint(0, self.train_length)

                return random_doc_id, -1
            else:
                # print('-----------')
                # print('\033[1mCurr num docs labeled {}\033[0m'.format(self.num_docs_labeled))
                # print('-----------')
                # return 
                # return random.randint(0, len(self.median_pro)-1)
                max_idx = np.argmax(self.scores)

                '''
                Don't show the users an already shown document
                '''
                while max_idx in self.recommended_doc_ids:
                    # self.scores = np.delete(self.scores, max_idx)
                    self.scores[max_idx] = -1
                    try:
                        max_idx = np.argmax(self.scores)
                    except:
                        print('current len of the score list is {}'.format(len(self.scores)))
                

                print('Classifier in progess...')
                print('\033[1mScore of the current document is {}\033[0m'.format(self.scores[max_idx]))
                self.recommended_doc_ids.add(max_idx)
                return max_idx, self.scores[max_idx]

            
    def recommend_document(self):
        document_id, score = self.preference()
        # self.num_docs_labeled += 1
        print(self.classes)
 
        return document_id, score

    def is_labeled(self, doc_id):
        return doc_id in self.user_labels

    def update_classifier(self):
        if self.mode == 1:
            # print('doc topic prob shape {}'.format(self.doc_topic_prob.shape))
            guess_label_probas = self.classifier.predict_proba(self.text_vectorizer[0:self.train_length])
            guess_label_logprobas = self.classifier.predict_log_proba(self.text_vectorizer[0:self.train_length])
            scores = -np.sum(guess_label_probas*guess_label_logprobas*self.doc_topic_prob[:,:len(self.classes)], axis = 1)

            self.scores = scores
        else:
            guess_label_probas = self.classifier.predict_proba(self.text_vectorizer[0:self.train_length])
            guess_label_logprobas = self.classifier.predict_log_proba(self.text_vectorizer[0:self.train_length])
            scores = -np.sum(guess_label_probas*guess_label_logprobas, axis = 1)
            self.scores = scores

    def label(self, doc_id, user_label):
        # if self.mode == 1:
        if True:
            # print('\033[1mmode1\033[0m')
            # print(type(label_num))
            # label_num = int(label_num)
            if self.is_labeled(doc_id):
                if user_label in self.user_label_number_map:
                    label_num = self.user_label_number_map[user_label]
                    self.user_labels[doc_id] = label_num
                    self.labels_track[self.id_vectorizer_map[doc_id]] = label_num
                    if len(self.classes) >= 2:
                        self.classifier.partial_fit(self.documents_track, self.labels_track, self.classes)
                        self.update_classifier()
                else:
                    label_num = len(self.classes)
                    self.user_label_number_map[user_label] = label_num
                    self.user_labels[doc_id] = label_num
                    self.classes.append(label_num)
                    self.labels_track[self.id_vectorizer_map[doc_id]] = label_num
                    if len(self.classes) >= 2:
                        self.classifier = self.initialize_classifier('logreg')
                        self.classifier.partial_fit(self.documents_track, self.labels_track, self.classes)
                        self.update_classifier()  
            elif user_label in self.user_label_number_map:  
                print('all labels have, keep classes same but increase documents')
                label_num = self.user_label_number_map[user_label]   
                self.user_labels[doc_id] = label_num
                self.id_vectorizer_map[doc_id] = self.num_docs_labeled
                # print(self.id_vectorizer_map)
                '''
                Adding documents and labels and track them
                '''
                self.labels_track.append(label_num)
                # self.id_vectorizer_map[doc_id] = len(self.labels_track-1)
                # self.documents_track.append(self.docs[doc_id])
                if self.text_vectorizer == None:
                    self.documents_track = self.text_vectorizer[doc_id]
                else:
                    self.documents_track = vstack((self.documents_track, self.text_vectorizer[doc_id]))
                
                if len(self.classes) < 2:
                    if self.mode == 1:
                        self.update_median_prob(label_num)
                else:
                    print('-----------')
                    print('start incremental learning')
                    print('-----------')
                    self.classifier.partial_fit(self.documents_track, self.labels_track, self.classes)
                    # self.classifier.partial_fit(self.text_vectorizer[doc_id], [label_num], self.classes)

                    self.update_classifier()
                    

                self.num_docs_labeled += 1

                print('\033[1mnum docs labeled {}\033[0m'.format(self.num_docs_labeled))
            else:
                print('\033[1mImplement this part: Creating a new label\033[0m')
                label_num = len(self.classes)
                self.classes.append(label_num)
                self.id_vectorizer_map[doc_id] = self.num_docs_labeled
                self.user_label_number_map[user_label] = label_num
                # self.curr_label_num.append(label_num)
                self.user_labels[doc_id] = label_num

                self.labels_track.append(label_num)
                if self.text_vectorizer == None:
                    self.documents_track = self.text_vectorizer[doc_id]
                else:
                    self.documents_track = vstack((self.documents_track, self.text_vectorizer[doc_id]))

                if len(self.classes) < 2:
                    if self.mode == 1:
                        self.update_median_prob(label_num)
                else:
                    print('-----------')
                    print('start incremental learning')
                    print('-----------')
                    self.classifier = self.initialize_classifier('logreg')
                    self.classifier.partial_fit(self.documents_track, self.labels_track, self.classes)
                    # self.classifier.partial_fit(self.text_vectorizer[doc_id], [label_num], self.classes)

                    self.update_classifier()
                    

                self.num_docs_labeled += 1
        
    
