import random
from sklearn.linear_model import SGDClassifier
from User import User
from scipy.sparse import vstack
import numpy as np
import json
import pandas as pd
import os
import pickle
import argparse
from alto_session import NAITM
from sklearn.feature_extraction.text import TfidfVectorizer

switch_doc = True

def group_docs_to_topics(model_inferred):
    doc_prob_topic = []
    doc_to_topics, topics_probs = {}, {}

    for doc_id, inferred in enumerate(model_inferred):
        doc_topics = list(enumerate(inferred))
        doc_prob_topic.append(inferred)

        doc_topics.sort(key = lambda a: a[1], reverse= True)

        doc_to_topics[doc_id] = doc_topics
        if doc_topics[0][0] in topics_probs:
            topics_probs[doc_topics[0][0]].append((doc_id, doc_topics[0][1]))
        else:
            topics_probs[doc_topics[0][0]] = [(doc_id, doc_topics[0][1])]

    for k, v in topics_probs.items():
        topics_probs[k].sort(key = lambda a: a[1], reverse= True)

    return topics_probs, doc_prob_topic

#NATM stands for neural interactive active topic modeling
def main():
    # parse which fllename to use
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--doc_dir", help="Where we read the source documents",
                       type=str, default="./Data/newsgroup_sub_500.json", required=False)
    argparser.add_argument("--save_model", help="Whether we save the model",
                       type=bool, default=True, required=False)
    argparser.add_argument("--load_model_path", help="path of the model loaded",
                       type=str, default="./Model/LDA_model_data.pkl", required=False)
    argparser.add_argument("--load_data", help="Whether we load the data",
                       type=bool, default=False, required=False)
    argparser.add_argument("--labeling_scheme", help="What scheme we use for users to label" 
                        + "Schemes we can use: TA, TR, LA, LR", 
                       type=str, default= 'TA', required=False)
    argparser.add_argument("--num_iter", help="number of iterations to train the model" , 
                       type=int, default= 300, required=False)
    argparser.add_argument("--inference_alg", help="choose the type of interence you want: logreg, baye, MoDAL" , 
                       type=str, default= 'logreg', required=False)
    
    args = argparser.parse_args()

    print('pre reading frame...')

    df = pd.read_json(args.doc_dir)
    documents = df.text.values.tolist()

    print('success reading frame...')

    model_path = './Model/ETM.pkl'
    with open(model_path, 'rb') as inp:
        model = pickle.load(inp)

    doc_topic = model.get_document_topic_dist()
    document_probas,  doc_topic_probas = group_docs_to_topics(doc_topic)

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
    vectorizer_idf = vectorizer.fit_transform(df.text.values.tolist())

    session = NAITM(documents, document_probas,  doc_topic_probas, df, args.inference_alg, vectorizer_idf, 20, 500)

    doc_count = 0
    while doc_count < len(df):
        # random_document = random.randint(0, model.get_num_docs()-1)
        print('Current Document is as follows, please choose a suggested label for it, or choose ' +
               'Disagree and choose a suitable label. Or create a new label for the new document ' +
               'by entering \'Create\'. Update classifier by entering \'update\'')
        print()
        
        random_document, _ = session.recommend_document()

        print()
        print('New Document {}'.format(random_document))
        print('-----------')
        print(documents)
        print('-----------')

        print()

        # inferred_topics = model.predict_doc(random_document, 3)
        # print('Top suggested topics and their related keywords for this document are as follows')
        # print()
        # print('inferred topics are {}'.format(inferred_topics))
        # print()
        # for num, prob in inferred_topics:
        #     print('Topic {}. Confidence: {}'.format(num, prob))
        #     if args.model_type == 'LDA':
        #         print('Keywords {}'.format(topics[num]))
        #     elif args.model_type == 'LLDA' or args.model_type == 'PLDA' or args.model_type == 'SLDA':
        #         print(topics[num][1])
        #         print('Keywords {}'.format(topics[num][0]))
                # print('Keywords {}'.format(topics[num]))
        
        print()
        print('----------')
        val = input("Please enter your decided label below: ")
        print('----------')

        if val.isdigit():
            user_input = int(val)
            if user_input >= 0 and user_input < 20:
                print()
                print('User input label is {}'.format(user_input))
                print()
                session.label(random_document, user_input)
                doc_count += 1


if __name__ == "__main__":
    main()