'''
This file processes the data for models to do topic model or active learning
'''
def process(data_path):
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