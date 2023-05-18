import random
import pickle
import tomotopy as tp
import numpy as np

class TopicModel():
    def __init__(self, model_path, model_type, dataset_dir, num_topics):
        print(model_path)
        with open(model_path, 'rb') as inp:
            self.loaded_data = pickle.load(inp)

        if model_type == 'LDA':
            self.model = tp.LDAModel.load(model_path.replace('pkl', 'bin'))
        elif model_type =='SLDA':
            self.model = tp.SLDAModel.load(model_path.replace('pkl', 'bin'))
        self.document_probas = self.loaded_data['document_probas']
        self.doc_topic_probas = self.loaded_data['doc_topic_probas']
        # self.get_document_topic_dist = self.loaded_data['get_document_topic_dist']
        self.topic_keywords = None
        self.topic_word_dist = None
        self.model_type = model_type
        self.num_topics = num_topics

        self.data_words_nonstop = self.loaded_data['datawords_nonstop']
        self.word_spans = self.loaded_data['spans']
        self.texts = self.loaded_data['texts']
        self.maked_docs = [self.model.make_doc(ele) for ele in self.data_words_nonstop]

    def print_topics(self, verbose=False):
        mdl = self.model
        out_topics = dict()
   
        # print(self.model_type == 'LDA')
        # print('label len is {}'.format(len(labels)))
        for k in range(mdl.k):
            topic_words = [tup[0] for tup in mdl.get_topic_words(k, top_n=20)]
            if verbose:
                print(topic_words)

            out_topics[k] = topic_words
        
        return out_topics
    
    def group_docs_to_topics(self):
        return self.document_probas, self.doc_topic_probas
    
    def get_word_topic_distribution(self):
        '''
        Data structure
        {
        [word1]: [topic1, topic2, topic3...]
        [word2]: [topic1, topic2, topic3...]
        [word3]: [topic1, topic2, topic3...]
        ...
        }
        '''

        # vocabs = self.model.vocabs
        vocabs = self.model.used_vocabs
        topic_dist = dict()

        for i in range(self.num_topics):
            topic_dist[i] = self.model.get_topic_word_dist(i)
        
        # print(vocabs[0])
        # print(vocabs[-1])
        print(len(topic_dist[0]))
        print(len(vocabs))
        print(len(self.model.used_vocabs))
        word_topic_distribution = dict()
        for i, word in enumerate(vocabs):
            word_topic_distribution[word] = [v[i] for k, v in topic_dist.items()]

        self.word_topic_distribution = word_topic_distribution

        return word_topic_distribution
    
    def predict_doc_with_probs(self, doc_id, topics):
        # print(topics)
        if self.model_type != 'SLDA':
            inferred, _= self.model.infer(self.maked_docs[doc_id])
        else:
            inferred = self.model.estimate(self.maked_docs[doc_id])
            
        result = list(enumerate(inferred))
        # print(result)
        # result.sort(key = lambda a: a, reverse= True)
        result = sorted(result, key=lambda x: x[1], reverse=True)
        # print(result)
        topic_res = [[str(k), str(v)] for k, v in result]
        topic_res_num = []

        # topic_word_res = {}
        # print(self.topics)
        for num, prob in result:
            keywords = topics[num]
            # topic_word_res[str(num)] = keywords
            topic_res_num.append((num, keywords))

        # print(topic_res_num)
        return topic_res, topic_res_num
    
    def get_word_span_prob(self, doc_id, topic_res_num, threthold):
        if threthold <= 0:
            return dict()
        
        doc = self.data_words_nonstop[doc_id]
        doc_span = self.word_spans[doc_id]
        # print(doc_span)
        # self.word_topic_distribution
        result = dict()
        # if self.model_type == 'LDA':
        # for j in range(self.num_topics):
        #     result[str(j)] = []
        # for topic, keywords in topic_res_num:
        for ele in topic_res_num:
            topic = ele[0]
            # keywords = ele[1]
            result[str(topic)] = {}
            result[str(topic)]['spans'] = []
            # result[str(topic)]['score'] = []

        for i, word in enumerate(doc):
            # for topic in range(self.num_topics):
            # for topic, keywords in topic_res_num:
            for ele in topic_res_num:
                topic = ele[0]
                keywords = ele[1]

                '''
                This part (word in self.word_topic_distribution) might filter out a lot of the vocabularies
                '''
                if word in self.word_topic_distribution and self.word_topic_distribution[word][topic] >= threthold:
                    if len(doc_span[i])>0 and doc_span[i][0] <= len(self.texts[doc_id]) and doc_span[i][1] <= len(self.texts[doc_id]):
                        result[str(topic)]['spans'].append([doc_span[i][0], doc_span[i][1]])
                    # result[str(topic)]['score'].append(str(self.word_topic_distribution[word][topic]))
                result[str(topic)]['keywords'] = keywords

        return result

    def concatenate_keywords(self, topic_keywords, datawords):
        result = []
        for i, doc in enumerate(datawords):
            if i < len(self.data_words_nonstop):
                topic_idx = np.argmax(self.doc_topic_probas[i])
                keywords = topic_keywords[topic_idx]
                curr_ele = doc + keywords
                res_ele = ' '.join(curr_ele)
                result.append(res_ele)
            else:
                res_ele = ' '.join(doc)
                result.append(res_ele)
                
        return result