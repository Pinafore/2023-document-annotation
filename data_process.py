'''
This file processes the data for models to do topic model or active learning
'''
import pandas as pd
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS
import pickle
import argparse

class Preprocessing():
    def __init__(self, data_path):
        print('Processing data...')
        df = pd.read_json(data_path)
        self.df = df
        
        data = df.text.values.tolist()
        self.texts = data
        # print('Finish lemmatization')

        nlp = spacy.load('en_core_web_sm')
        # nlp.add_pipe('sentencizer')
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
                                
        docs = [nlp(x) for x in data]

        stop_words = STOP_WORDS
        self.data_words_nonstop, self.word_spans = [], []

        with open("newsgroup_removewords.txt") as f:
                words_to_remove = f.readlines()

        words_to_remove = [x.strip() for x in words_to_remove]
        new_words_to_remove = [j.replace('\'', '') for i in words_to_remove for j in i.split(',')]
        new_words_to_remove = [j.replace(' ', '') for j in new_words_to_remove]
        # self.stop_words.extend(new_words_to_remove)
        for word in new_words_to_remove:
            stop_words.add(word)

                #Creating and updating our list of tokens using list comprehension 
        if 'newsgroup' in data_path:
            for doc in docs:
                temp_doc = []
                temp_span = []
                for token in doc:
                    if not len(str(token)) == 1 and (re.search('[a-z0-9]+',str(token))) \
                        and not token.pos_ == 'PROPN' and not token.is_digit and not token.is_space \
                                                    and str(token).lower() not in stop_words:
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
                            and str(token).lower() not in stop_words:
                        temp_doc.append(token.lemma_)
                        temp_span.append((token.idx, token.idx + len(token)))
                self.data_words_nonstop.append(temp_doc)
                self.word_spans.append(temp_span)

        self.labels = df.label.values.tolist()
        
    def save_data(self, save_path):
         print('saving data...')
         result = {}
         result['datawords_nonstop'] = self.data_words_nonstop
         result['spans'] = self.word_spans
         result['texts'] = self.texts
         result['labels'] = self.labels
         with open('./Data/{}.pkl'.format(save_path), 'wb+') as outp:
                    pickle.dump(result, outp)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--doc_dir", help="Where we read the source documents",
                       type=str, default="./Data/newsgroup_sub_500.json", required=False)
    argparser.add_argument("--save_path", help="Whether we save the data",
                       type=str, default='newsgroup_sub_500_processed', required=False)
    
    args = argparser.parse_args()
    process_obj = Preprocessing(args.doc_dir)
    process_obj.save_data(args.save_path)

if __name__ == "__main__":
    main()