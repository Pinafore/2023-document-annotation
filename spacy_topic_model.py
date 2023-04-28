import tomotopy as tp
from tomotopy.utils import Corpus
from gensim.utils import simple_preprocess
import pickle
import os
import re
import gensim.corpora as corpora

import numpy as np
import pandas as pd
from pprint import pprint

# spacy for lemmatization
import spacy
import nltk
from nltk.corpus import stopwords

import sklearn
import gensim
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

USE_MODEL = True

class TopicModel():
    def __init__(self, corpus_path, model_type, min_num_topics, num_iters, load_model, save_model, load_path, hypers):
        self.hyperparameters = hypers
        self.word_spans = []
        self.num_topic_words = 15
        self.train_length = 500
        self.df = pd.read_json(corpus_path)
        self.load_model, self.load_path, self.save_model = load_model, load_path, save_model
        # self.load_path = './Model/{}_model_data.pkl'.format(model_type)
        self.corpus_data = corpus_path.replace('.json', '')
        self.corpus_data = self.corpus_data.replace('./Data/', '')

        # print('corpus data is {}'.format(self.corpus_data))

        self.load_path = './Model/{}_model_{}.pkl'.format(model_type, self.corpus_data)

        self.labels = self.df['label'].tolist()
        self.texts = self.df.text.values.tolist()
        # self.stop_words = stopwords.words('english')
        self.stop_words = STOP_WORDS
        self.saved = dict()
        self.read_data = None
        ###
        with open("newsgroup_removewords.txt") as f:
            words_to_remove = f.readlines()

        words_to_remove = [x.strip() for x in words_to_remove]
        new_words_to_remove = [j.replace('\'', '') for i in words_to_remove for j in i.split(',')]
        new_words_to_remove = [j.replace(' ', '') for j in new_words_to_remove]
        # self.stop_words.extend(new_words_to_remove)
        for word in new_words_to_remove:
            self.stop_words.add(word)
        ###

        # self.bigram_mod = None
        # self.trigram_mod = None
        # self.id2word = None
        self.data_words_nonstop = []
        self.word_spans = []
        self.topics, self.topics_probs, self.doc_to_topics = dict(), dict(), dict()

        self.corpus = None
        self.lda_model = None
        self.maked_docs = []

        ''''''
        self.model_type = model_type
        if model_type == 'LLDA':
            self.model_lib = tp.LLDAModel
        elif model_type == 'SLDA':  
            self.model_lib = tp.SLDAModel
        elif model_type == 'LDA':
            self.model_lib = tp.LDAModel
        elif model_type == 'PLDA':
            self.model_lib = tp.PLDAModel
        else:
            raise Exception("unsupported model!")

        self.label_set = []
        self.num_topics = min_num_topics # minimum number of topics
        self.num_iters = num_iters # number of passes through the corpus
        self.word_topic_distribution = None

    def get_num_docs(self):
        try:
            return len(self.corpus)
        except:
            return 0

    def retrive_texts(self, doc_id):
        return self.texts[doc_id]

    def preprocess(self, min_count, threthold):
        if self.load_model:
            # print('Loading model data')
            with open(self.load_path, 'rb') as inp:
                self.read_data = pickle.load(inp)
            
            self.data_words_nonstop = self.read_data['data_words_nonstop']
            self.label_set = self.read_data['label_set']
            self.corpus = self.read_data['corpus']
            self.word_spans = self.read_data['word_spans']
        else:
            print('Processing data...')
            data = self.df.text.values.tolist()

            # print('Finish lemmatization')

            nlp = spacy.load('en_core_web_sm')
            # nlp.add_pipe('sentencizer')
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
                            
            docs = [nlp(x) for x in data]


            #Creating and updating our list of tokens using list comprehension 
            if 'newsgroup' in self.corpus_data:
                for doc in docs:
                    temp_doc = []
                    temp_span = []
                    for token in doc:
                        if not len(str(token)) == 1 and (re.search('[a-z0-9]+',str(token))) \
                                                and not token.pos_ == 'PROPN' and not token.is_digit and not token.is_space \
                                                and str(token).lower() not in self.stop_words:
                            temp_doc.append(token.lemma_)
                            temp_span.append((token.idx, token.idx + len(token)))
                    self.data_words_nonstop.append(temp_doc)
                    self.word_spans.append(temp_span)
            else:
                for doc in docs:
                    temp_doc = []
                    temp_span = []
                    for token in doc:
                        if (re.search('[a-z0-9]+',str(token))) \
                            and not len(str(token)) == 1 and not token.is_digit and not token.is_space \
                            and str(token).lower() not in self.stop_words:
                            temp_doc.append(token.lemma_)
                            temp_span.append((token.idx, token.idx + len(token)))
                    self.data_words_nonstop.append(temp_doc)
                    self.word_spans.append(temp_span)

                

            print('length of datawords nonstop is {}'.format(len(self.data_words_nonstop)))

            corpus = Corpus()
            
            # make label dict
            label_set = list(set(self.labels))
           

            # print('In processing, length of label set is {}'.format(len(label_set)))

            print('Finish getting labelset')
            if np.nan in label_set:
                label_set.remove(np.nan)
            if None in label_set:
                label_set.remove(None)
            if 'None' in label_set:
                label_set.remove('None')

            label_dict = dict()
            for i,label in enumerate(label_set):
                label_dict[label] = i
            self.label_set = label_set

            

            if self.model_type == 'LLDA' or self.model_type == 'PLDA':
                print('Enumerating grams...')
                for i, ngrams in enumerate(self.data_words_nonstop):
                    # if i % 100 == 0:
                    #     print('PLDA {}'.format(i))
                    # if self.labels and type(self.labels[i]) == str:
                    if self.labels and not self.labels[i] == 'None':
                        label = self.labels[i]
                        corpus.add_doc(ngrams, labels=[label])
                    else:
                        corpus.add_doc(ngrams)
            elif self.model_type == 'LDA':
                '''
                Change something here
                '''
                for i, ngrams in enumerate(self.data_words_nonstop):
                    corpus.add_doc(ngrams)
            elif self.model_type == 'SLDA':
                for i, ngrams in enumerate(self.data_words_nonstop):
                    y = [0 for _ in range(len(label_set))]
                    null_y = [np.nan for _ in range(len(label_set))]
                    # corpus.add_doc(ngrams, y=null_y)
                    # if self.labels and type(self.labels[i]) == str:
                    if self.labels and not self.labels[i] == 'None':
                        label = self.labels[i]
                        y[label_dict[label]] = 1
                        corpus.add_doc(ngrams, y=y)
                        # print(y)
                    else:
                        corpus.add_doc(ngrams, y=null_y)
                        # print(null_y)

                    # print(corpus[i])
                    # if i>10: break
            else:
                raise Exception("unsupported model type!")

            self.corpus = corpus
            
            print('Finish processing. Start saving...')
            print('length of the corpus is {}'.format(len(corpus)))
            assert len(corpus) == len(self.texts)

            if self.save_model:
                # saved = dict()
                self.saved['data_words_nonstop'] = self.data_words_nonstop
                self.saved['label_set'] = self.label_set
                self.saved['corpus'] = self.corpus
                self.saved['word_spans'] = self.word_spans
                
                print('Saving data to ./Model/{}_model_{}.pkl'.format(self.model_type, self.corpus_data))
                with open('./Model/{}_model_{}.pkl'.format(self.model_type, self.corpus_data), 'wb+') as outp:
                    pickle.dump(self.saved, outp)
        
    def train(self, num_topics=None):
        # print('training...')
        if not num_topics: 

            num_topics = max(self.num_topics, len(self.label_set))
            self.num_topics = num_topics
            # if self.label_set:
            # else: num_topics = self.num_topics
        else:
            self.num_topics = num_topics

        print('num topics:', num_topics)

    
        # print('Model type is {}'.format(self.model_type))

        if self.load_model and USE_MODEL:
            if self.model_type == 'LLDA':
                mdl = tp.LLDAModel.load('./Model/{}_model_{}.bin'.format(self.model_type, self.corpus_data))
            elif self.model_type == 'SLDA':
                mdl = tp.SLDAModel.load('./Model/{}_model_{}.bin'.format(self.model_type, self.corpus_data))
            elif self.model_type == 'LDA':
                mdl = tp.LDAModel.load('./Model/{}_model_{}.bin'.format(self.model_type, self.corpus_data))
            elif self.model_type == 'PLDA':
                mdl = tp.PLDAModel.load('./Model/{}_model_{}.bin'.format(self.model_type, self.corpus_data))

        else:
            if self.model_type == 'LLDA':
                print('Created LLDA model')
                mdl = tp.LLDAModel(k=num_topics)
                # mdl = tp.LLDAModel(k=20)
                # mdl = tp.PLDAModel(k=20)
            elif self.model_type == 'SLDA':
                print('Created SLDA model')
                # print('Getting into SLDA...')

                mdl = tp.SLDAModel(k=num_topics, vars=['b' for _ in range(len(self.label_set))], glm_param= [1.1 for i in range(len(self.label_set))], nu_sq = [5 for i in range(len(self.label_set))])
                # mdl = tp.SLDAModel(k=num_topics, vars=self.label_set)
                
            elif self.model_type == 'LDA':
                print('Created LDA model')
                mdl = tp.LDAModel(k=num_topics)
            elif self.model_type == 'PLDA':
                mdl = tp.PLDAModel()
                print('Created PLDA model')

            mdl.add_corpus(self.corpus)

            print('starting training...')

            # print('total # topics {}'.format(mdl.k))
            # for i in range(0, self.num_iters, 10):
            #     # print('training iter {}'.format(i))
            #     mdl.train(10)
            mdl.train(self.num_iters)
            

            mdl.save('./Model/{}_model_{}.bin'.format(self.model_type, self.corpus_data))

        # print('total # topics after {}'.format(mdl.k))
        # if verbose:
        #     mdl.summary()

        self.lda_model = mdl

        if self.model_type == 'LLDA' or self.model_type == 'PLDA':
            self.label_set = mdl.topic_label_dict

        '''
        Make documents for normal LDA
        '''
        # if self.model_type == 'LDA':
        for doc in self.data_words_nonstop:
            curr_doc = mdl.make_doc(doc)
            self.maked_docs.append(curr_doc)
        
    def print_topics(self, verbose=False):
        mdl = self.lda_model
        out_topics = dict()
        # labels = mdl.topic_label_dict
        labels = self.label_set

        # print(self.model_type == 'LDA')
        # print('label len is {}'.format(len(labels)))
        for k in range(mdl.k):
            if verbose:
                print('Top 10 words of topic #{}'.format(k))
            if not self.model_type == 'LDA' and k<len(labels) and verbose:
                # print('enter here')
                print(labels[k])
            topic_words = [tup[0] for tup in mdl.get_topic_words(k, top_n=10)]
            if verbose:
                print(topic_words)
            if self.model_type == 'LDA' and k<len(labels):
                out_topics[k] = topic_words
            elif self.model_type == 'LLDA' or self.model_type == 'PLDA' or self.model_type == 'SLDA':
                out_topics[k] = (topic_words, labels[k])

        return out_topics

    def group_docs_to_topics(self, verbose=False, load=False):
        # doc_topics, word_topics, phi_values = self.model.get_document_topics(self.corpus[doc_id], per_word_topics=True)
        '''
        probabilities of a document belonging to a particular topic with the highest probability
        probabilities stored with the highest order
        '''

        if self.load_model and 'topics_probs' in self.read_data:
            if self.read_data:
                self.topics_probs = self.read_data['topics_probs']
                doc_prob_topic = self.read_data['doc_prob_topic']
            else:
                with open(self.load_path, 'rb') as inp:
                    read_data = pickle.load(inp)
                
                self.topics_probs = read_data['topics_probs']
                doc_prob_topic = read_data['doc_prob_topic']

        # print('grouping documents...')
        else:
            doc_prob_topic = []

            '''
            Modify this part to make it infer only once. Instead of using a for loop to do it
            '''



            if verbose:
                print('inferring topic probabilities')
            for doc_id, doc in enumerate(self.maked_docs[0:self.train_length]):
            # for doc_id, doc in enumerate(self.corpus):
                # if self.model_type == 'SLDA':
                #     print('maked {}'.format(doc))
                #     inferred, _ = self.lda_model.infer(doc)
                #     # inferred = self.lda_model.estimate(inferred)
                # else:
                if self.model_type == 'SLDA':
                    inferred = self.lda_model.estimate(doc)
                else:
                    inferred, _ = self.lda_model.infer(doc)


                # print(inferred)
                # break
                doc_topics = list(enumerate(inferred))
                doc_prob_topic.append(inferred)
                

                # Infer the top three topics of the document
                doc_topics.sort(key = lambda a: a[1], reverse= True)
                
                # print('doc topics {}'.format(doc_topics))
                self.doc_to_topics[doc_id] = doc_topics

                '''
                doc_topics[0][0] is the topic id. doc_topics[0][1] is doc id probability.
                '''
                if doc_topics[0][0] in self.topics_probs:
                    self.topics_probs[doc_topics[0][0]].append((doc_id, doc_topics[0][1]))
                else:
                    self.topics_probs[doc_topics[0][0]] = [(doc_id, doc_topics[0][1])]


            if verbose:
                print('sorting topic probabilities')
            '''
            Sort the documents by topics based on their probability in descending order
            '''
            for k, v in self.topics_probs.items():
                self.topics_probs[k].sort(key = lambda a: a[1], reverse= True)
                # print('{} documents in topic {}'.format(len(self.topics_probs[k]), k))

            self.saved['topics_probs'] = self.topics_probs
            self.saved['doc_prob_topic'] = doc_prob_topic

            with open('./Model/{}_model_{}.pkl'.format(self.model_type, self.corpus_data), 'wb+') as outp:
                    pickle.dump(self.saved, outp)

        return self.topics_probs, doc_prob_topic

    '''
    Predict a document's topic based on its document id
    '''
    def predict_doc(self, doc_id, top_n):
        inferred, _= self.lda_model.infer(self.maked_docs[doc_id])
        # inferred, log_ll= self.lda_model.infer(self.corpus[doc_id])
        result = list(enumerate(inferred))
        result.sort(key = lambda a: a[1], reverse= True)
        
        return result[:top_n]

    def predict_labels(self, document_data):

        ldamodel = self.lda_model
        corpus = self.corpus

        # topic_df = pd.DataFrame()
        topic_df = dict()
        topic_df['text'] = dict()
        topic_df['topic_model_prediction'] = dict()
        topic_df['topic_model_prediction_score'] = dict()
        topic_df['label'] = dict()
        topic_df['Dominant_Topic'] = dict()
        topic_df['topic_keywords'] = dict()
        if self.model_type == 'SLDA':
            inferred, _ = self.lda_model.infer(self.corpus)
            # print(inferred)
            preds = self.lda_model.estimate(inferred)

            # pred_labels = []
            # pred_scores = []
            # topic_words_per_doc = []

            for i, scores in enumerate(preds):
                topic_num = int(np.argmax(scores))
                # pred_scores.append(scores[topic_num])
                # pred_labels.append(self.label_set[topic_num])
                # pred_labels.append(topic_num)
                # print(pred, scores[i], label_set[i])
                topic_words = ', '.join([tup[0] for tup in ldamodel.get_topic_words(topic_num, top_n=self.num_topic_words)])
                # topic_words_per_doc.append(topic_words)
                topic_df['text'][str(i)] = self.texts[i]
                topic_df['topic_model_prediction'][(str(i))] = self.label_set[topic_num]
                topic_df['topic_model_prediction_score'][str(i)] = str(round(scores[topic_num], 4))
                topic_df['topic_keywords'][str(i)] = topic_words
                topic_df['label'][str(i)] = self.labels[i]
                topic_df['Dominant_Topic'][str(i)] = topic_num

            # print('length of prediction label is {}'.format(len(topic_df['topic_model_prediction']['text'])))
            # topic_df['topic_model_prediction'] = pred_labels
            # topic_df['topic_model_prediction_score'] = pred_scores
            # topic_df['topic_keywords'] = topic_words_per_doc

            # return document_data

        elif self.model_type == 'LLDA' or self.model_type == 'PLDA':

            # Init output
            # sent_topics_df = pd.DataFrame()

            labels = ldamodel.topic_label_dict
            # print('topic labels:', labels)

            # Get main topic in each document
            topic_dist, ll = ldamodel.infer(corpus)
            for i, doc in enumerate(topic_dist):
    #             row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
                topic_num, topic_pct = doc.get_topics(top_n=1)[0]
        
                # get topic words
                topic_words = ', '.join([tup[0] for tup in ldamodel.get_topic_words(topic_num, top_n=10)])
                if topic_num < len(labels):
                    topic_label = labels[topic_num]
                else:
                    topic_label = 'other'
                
                # topic_df = topic_df.append(pd.Series([
                #     int(topic_num), 
                #     topic_label,
                #     round(topic_pct,4), 
                #     topic_words,
                #     ]), ignore_index=True)
                topic_df['text'][str(i)] = self.texts[i]
                topic_df['topic_model_prediction'][(str(i))] = topic_label
                topic_df['topic_model_prediction_score'][str(i)] = str(round(topic_pct,4))
                topic_df['topic_keywords'][str(i)] = topic_words
                topic_df['label'][str(i)] = self.labels[i]
                topic_df['Dominant_Topic'][str(i)] = topic_num
            # sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
            # topic_df.columns = [
            #     'dominant_topic_id', 
            #     'topic_model_prediction', 
            #     'topic_model_prediction_score', 
            #     'topic_keywords']
            # # display(sent_topics_df)
            # print(document_data.shape, topic_df.shape)

            
            # return(topic_df)
        elif self.model_type == 'LDA':
            topic_dist, ll = ldamodel.infer(corpus)
            for i, doc in enumerate(topic_dist):
    #             row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
                topic_num, topic_pct = doc.get_topics(top_n=1)[0]
        
                # get topic words
                topic_words = ', '.join([tup[0] for tup in ldamodel.get_topic_words(topic_num, top_n=10)])
                
                
                topic_df['text'][str(i)] = self.texts[i]
                topic_df['topic_model_prediction'][(str(i))] = str(topic_num)
                topic_df['topic_model_prediction_score'][str(i)] = str(round(topic_pct,4))
                topic_df['topic_keywords'][str(i)] = topic_words
                topic_df['label'][str(i)] = self.labels[i]
                topic_df['Dominant_Topic'][str(i)] = topic_num

        # Add original text to the end of the output
        # merged = pd.concat([document_data, topic_df], axis=1)
        # y_true = merged['tag_1']
        # y_pred = merged['lda_pred']
        # print(sklearn.metrics.accuracy_score(y_true, y_pred))

        return topic_df

    def predict_new_doc(self, new_document):
        new_doc_obj = self.lda_model.make_doc(new_document)
        pred, _ = self.lda_model.infer(new_doc_obj)
        top_topic_num = np.argmax(pred)
        # top_key_words = self.lda_model.get_topic_words(top_topic_num, top_n=10)
        top_key_words = ' '.join([tup[0] for tup in self.lda_model.get_topic_words(top_topic_num, top_n=self.num_topic_words)])
        
        return [str(top_topic_num), top_key_words]

    def get_dominant_topic(self, document_data):
        # get dominant topic for each document, along with topic keywords

        ldamodel = self.lda_model
        corpus = self.corpus

        # Init output
        sent_topics_df = pd.DataFrame()

        labels = self.label_set
        print('topic labels:', labels)

        # Get main topic in each document
        topic_dist, _ = ldamodel.infer(corpus)
        # print('corpus: {}'.format(corpus))
        print('topic dist {}'.format(topic_dist))
        for i, doc in enumerate(topic_dist):
#             row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            topic_num, topic_pct = doc.get_topics(top_n=1)[0]
    
            # get topic words
            topic_words = ', '.join([tup[0] for tup in ldamodel.get_topic_words(topic_num, top_n=10)])
            if topic_num < len(labels):
                topic_label = labels[topic_num]
            else:
                topic_label = 'other'
            
            sent_topics_df = sent_topics_df.append(pd.Series([
                int(topic_num), 
                topic_label,
                round(topic_pct,4), 
                topic_words,
                ]), ignore_index=True)
          

        # sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
        sent_topics_df.columns = ['dominant_topic_id', 'dominant_topic', 'dominant_topic_percent', 'topic_keywords']
        # display(sent_topics_df)

        # Add original text to the end of the output
        sent_topics_df = pd.concat([document_data, sent_topics_df], axis=1)
        return (sent_topics_df)

    def label_document(self, i, label):
        self.corpus[i].metadata = [label]

    def retrive_texts(self, doc_id):
        return self.texts[doc_id]

    def get_texts(self):
        return self.texts
    
    def update_model(self, new_label, doc_id):
        if self.model_type == 'SLDA' or self.model_type == 'LLDA':
            print('Implement it here')
            return
        return
    
    def retrain(self):
        self.lda_model.train(10)
        
    
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

        vocabs = self.lda_model.vocabs
        topic_dist = dict()

        for i in range(self.num_topics):
            topic_dist[i] = self.lda_model.get_topic_word_dist(i)
        
        word_topic_distribution = dict()
        for i, word in enumerate(vocabs):
            word_topic_distribution[word] = [v[i] for k, v in topic_dist.items()]

        self.word_topic_distribution = word_topic_distribution

        return word_topic_distribution
    
    def get_word_span_prob(self, doc_id, topic_res_num, threthold):
        if threthold <= 0:
            return dict()
        

        # print('doc id {}'.format(doc_id))
        # print('word span length {}'.format(len(self.word_spans)))
        doc = self.data_words_nonstop[doc_id]
        doc_span = self.word_spans[doc_id]
        # print(doc_span)
        # self.word_topic_distribution
        result = dict()
        # if self.model_type == 'LDA':
        # for j in range(self.num_topics):
        #     result[str(j)] = []
        for topic, keywords in topic_res_num:
            result[str(topic)] = {}
            result[str(topic)]['spans'] = []
            # result[str(topic)]['score'] = []

        for i, word in enumerate(doc):
            # for topic in range(self.num_topics):
            for topic, keywords in topic_res_num:
                if self.word_topic_distribution[word][topic] >= threthold:
                    # result[str(topic)].append((doc_span[i], self.word_topic_distribution[word][topic]))
                    result[str(topic)]['spans'].append([doc_span[i][0], doc_span[i][1]])
                    # result[str(topic)]['score'].append(str(self.word_topic_distribution[word][topic]))
                result[str(topic)]['keywords'] = keywords[0]

        return result
    
    def predict_doc_with_probs(self, doc_id, topics):
        # print(topics)
        if self.model_type != 'SLDA':
            inferred, _= self.lda_model.infer(self.maked_docs[doc_id])
        else:
            inferred = self.lda_model.estimate(self.maked_docs[doc_id])
            
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
            topic_res_num.append((num, topics[num]))

        # print(topic_res_num)
        return topic_res, topic_res_num
    
    
