import numpy as np

class User():
    def __init__(self, dataframe, curr_label_num):
        self.existing_labels = dataframe['label'].tolist()

        '''
        user_labels: labels the user creates for documents the user views
        curr_label_num: topic numbers computed by LDA
        '''
        self.user_labels = dict()
        self.curr_label_num = curr_label_num
    
    def is_labeled(self, doc_id):
        return doc_id in self.user_labels
        
    def label(self, doc_id, label_num):
        if self.is_labeled(doc_id):
            print('Implement here: switching labels for a document already labeled')
        elif label_num in self.curr_label_num:
            self.user_labels[doc_id] = label_num
        else:
            print('Creating a new label')
            self.curr_label_num.append(label_num)
            self.user_labels[doc_id] = label_num

    
        