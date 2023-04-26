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
    def __init__(self, documents, doc_prob, doc_topic_prob, df, running_type, text_vectorizer, num_topics, train_len):
        """
        Documents: the text corpus
        Doc_prob: a dictionary contains all the topics. Each topic 
        contains a list of documents along with their probabilities 
        belong to a specific topic
        [topic 1]: [(document a, prob 1), (document b, prob 2)...]
        """
        self.classes = list(range(num_topics))
        self.docs = documents
        self.doc_probs = doc_prob
        self.num_docs_labeled = 0
        self.last_recommended_topic = None
        self.recommended_doc_ids = set()
        self.text_vectorizer = text_vectorizer
        self.doc_topic_prob = doc_topic_prob
        self.train_length = train_len
        self.id_vectorizer_map = {}
        '''
        stores the scores of the documents after the user updates the classifier
        '''
        self.scores = []

        '''
        get the median robability of every topic
        '''
        self.median_pro = dict()
        self.cal_median_topics()

        self.existing_labels = df['label'].tolist()

        '''
        user_labels: labels the user creates for documents the user views
        curr_label_num: topic numbers computed by LDA
        '''
        self.user_labels = dict()
        self.curr_label_num = list(doc_prob.keys())

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
        if self.num_docs_labeled <= 2:
            print('-----------')
            print('Labeling smaller than 3')
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
            while self.doc_probs[max_topic_idx][0][0] in self.recommended_doc_ids and i < len(self.classes):
                # self.scores = np.delete(self.scores, max_idx)
                # max_idx = np.argmax(self.scores)
                # probs.pop(max_topic_idx)
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
            return self.doc_probs[max_topic_idx][0][0], -1
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
                self.scores = np.delete(self.scores, max_idx)
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

 
        return document_id, score

    def is_labeled(self, doc_id):
        return doc_id in self.user_labels

    def update_classifier(self):
        guess_label_probas = self.classifier.predict_proba(self.text_vectorizer[0:self.train_length])
        guess_label_logprobas = self.classifier.predict_log_proba(self.text_vectorizer[0:self.train_length])
        scores = -np.sum(guess_label_probas*guess_label_logprobas*self.doc_topic_prob, axis = 1)
        self.scores = scores

    def label(self, doc_id, label_num):
        # print(type(label_num))
        label_num = int(label_num)
        if self.is_labeled(doc_id):
            print('keep labeling the same document too many times')
            # print('\033[1mImplement this part: switching labels for a document already labeled\033[0m')
            print(doc_id)
            self.labels_track[self.id_vectorizer_map[doc_id]] = label_num
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
            
            if self.num_docs_labeled < 2:
                self.update_median_prob(label_num)
            
            if self.num_docs_labeled >= 2:
                print('-----------')
                print('start incremental learning')
                print('-----------')
                self.classifier.partial_fit(self.documents_track, self.labels_track, self.classes)
                # self.classifier.partial_fit(self.text_vectorizer[doc_id], [label_num], self.classes)

                '''
                Test for classifier by updating all the labeled texts
                '''
                # print()
                # print('-----------')
                # print('labels track...')
                # print('-----------')
                # print()

                # print(self.documents_track)
                # self.classifier.fit(self.documents_track, self.labels_track)
                # vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
                # vectorized_texts = vectorizer.fit_transform(self.documents_track)

                # Temp added
                # self.classifier = self.initialize_classifier('logreg')
                # from sklearn.linear_model import LogisticRegression
                # self.classifier = LogisticRegression(multi_class='multinomial')
                # print(self.documents_track)
                # print(vectorized_texts)
                # print(self.labels_track)

                # Temp added

                # self.classifier.fit(self.documents_track, self.labels_track)


                self.update_classifier()
                

            self.num_docs_labeled += 1

            print('\033[1mnum docs labeled {}\033[0m'.format(self.num_docs_labeled))
        else:
            print('\033[1mImplement this part: Creating a new label\033[0m')
            self.curr_label_num.append(label_num)
            self.user_labels[doc_id] = label_num


