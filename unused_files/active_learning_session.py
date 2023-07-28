import random
from sklearn.linear_model import SGDClassifier
from User import User
from scipy.sparse import vstack
import numpy as np

# This var means if True, when labeling the same doc, the next recommend doc would 
# be the same
switch_doc = True

class LA():
    def __init__(self, documents, df, running_type, text_vectorizer, train_len):
        """
        Documents: the text corpus
        Doc_prob: a dictionary contains all the topics. Each topic 
        contains a list of documents along with their probabilities 
        belong to a specific topic
        [topic 1]: [(document a, prob 1), (document b, prob 2)...]
        """
        # self.classes = list(range(num_topics))
        self.classes = list(range(20))
        self.docs = documents

        self.running_type = running_type
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

        self.existing_labels = df['label'].tolist()

        '''
        user_labels: labels the user creates for documents the user views
        curr_label_num: topic numbers computed by LDA
        '''
        self.curr_label_num = []
        self.user_labels = dict()

        '''
        Parameters or regressor can be changed later
        '''
        self.classifier = self.initialize_classifier(running_type)

        # Stores and track labeled documents and labels
        self.documents_track = None
        self.labels_track = []

        self.simulate_user_data = dict()

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

    def is_labeled(self, doc_id):
        return doc_id in self.user_labels

    '''
    Preference function can be changed or chosen for NAITM
    '''
    def preference(self):


        if self.num_docs_labeled < 1:
            print('-----------')
            print('Labeling smaller than 1')
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

    def update_classifier(self):
        guess_label_probas = self.classifier.predict_proba(self.text_vectorizer[0:self.train_length])
        guess_label_logprobas = self.classifier.predict_log_proba(self.text_vectorizer[0:self.train_length])
        scores = -np.sum(guess_label_probas*guess_label_logprobas, axis = 1)
        self.scores = scores

    def label(self, doc_id, label_num):
        # print(type(label_num))
        label_num = int(label_num)
        if self.is_labeled(doc_id):
            print('keep labeling the same document too many times')
            # print('\033[1mImplement this part: switching labels for a document already labeled\033[0m')
            print(doc_id)
            # print(self.id_vectorizer_map)
            self.labels_track[self.id_vectorizer_map[doc_id]] = label_num

            if label_num not in self.classes:
                self.classifier = self.initialize_classifier()
                # self.classes.append(self.classes[-1]+1)
                self.classes.append(label_num)
                self.classifier.partial_fit(self.documents_track, self.labels_track, self.classes)
            else:
                self.classifier.partial_fit(self.documents_track, self.labels_track, self.classes)
            
            self.update_classifier()
        elif label_num in self.curr_label_num:
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
            
            
          
            print('-----------')
            print('start incremental learning')
            print('-----------')
            self.classifier.partial_fit(self.documents_track, self.labels_track, self.classes)
            # self.classifier.partial_fit(self.text_vectorizer[doc_id], [label_num], self.classes)

            '''
            Test for classifier by updating all the labeled texts
            '''

            self.update_classifier()
                

            self.num_docs_labeled += 1

            print('\033[1mnum docs labeled {}\033[0m'.format(self.num_docs_labeled))
        else:
            print('\033[1mImplement this part: Creating a new label\033[0m')
            self.curr_label_num.append(label_num)
            self.user_labels[doc_id] = label_num
            self.id_vectorizer_map[doc_id] = self.num_docs_labeled


            if self.text_vectorizer == None:
                self.documents_track = self.text_vectorizer[doc_id]
            else:
                self.documents_track = vstack((self.documents_track, self.text_vectorizer[doc_id]))

            self.labels_track.append(label_num)

            if label_num not in self.classes:
                self.classifier = self.initialize_classifier(self.running_type)
                # self.classes.append(self.classes[-1]+1)
                self.classes.append(label_num)
                self.classifier.partial_fit(self.documents_track, self.labels_track, self.classes)
            else:
                self.classifier.partial_fit(self.documents_track, self.labels_track, self.classes)
            
            print('updating classifier')
            self.update_classifier()

            self.num_docs_labeled += 1
    