from spacy_topic_model import TopicModel
import time
import pandas as pd
import argparse
import random
import pandas as pd
import pickle
from alto_session import NAITM
from sklearn.feature_extraction.text import TfidfVectorizer

def save_model(model, filename):
    pickle.dump(model, open(filename,'wb'))


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
    argparser.add_argument("--model_type", help="LDA, LLDA, SLDA, PLDA",
                       type=str, default='LDA', required=False)
    argparser.add_argument("--labeling_scheme", help="What scheme we use for users to label" 
                        + "Schemes we can use: TA, TR, LA, LR", 
                       type=str, default= 'TA', required=False)
    argparser.add_argument("--num_iter", help="number of iterations to train the model" , 
                       type=int, default= 300, required=False)
    argparser.add_argument("--inference_alg", help="choose the type of interence you want: logreg, baye, MoDAL" , 
                       type=str, default= 'logreg', required=False)

    args = argparser.parse_args()

    # print('File to be read is {}'.format(args.doc_dir))
    # print('load data argument is {}'.format(args.load_data))

    print('pre reading frame...')

    df = pd.read_json(args.doc_dir)

    print('success reading frame...')

    model = TopicModel(corpus_path=args.doc_dir, model_type=args.model_type, min_num_topics= 5, num_iters= args.num_iter, load_model=args.load_data, save_model=args.save_model, load_path=args.load_model_path, hypers=None)
    
    model.preprocess(5, 100)


    num_topics = 20
    print('Start preparing for the model...')

    start = time.time()
    model.train(num_topics)
    end = time.time()
    print('Took {} seconds to prepare the model'.format(end-start))

    
    topics = model.print_topics(verbose=True)

    start = time.time()
    document_probas, doc_topic_probas = model.group_docs_to_topics(True)
    end = time.time()
    print('finish grouping probabs. Took {} seconds'.format(end-start))

    # print('topic keys are {}'.format(topics.keys()))
    # print('topics are {}'.format(topics))

    for k, v in document_probas.items():
        print('Topic {} has {} docs'.format(k, len(v)))

    '''
    Initialize the neural interactive active topic model
    '''
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
    vectorizer_idf = vectorizer.fit_transform(df.text.values.tolist())
    session = NAITM(model.get_texts(), document_probas,  doc_topic_probas, df, args.inference_alg, vectorizer_idf, len(topics), 500)


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
        print(model.retrive_texts(random_document))
        print('-----------')

        print()

        inferred_topics = model.predict_doc(random_document, 3)
        print('Top suggested topics and their related keywords for this document are as follows')
        print()
        print('inferred topics are {}'.format(inferred_topics))
        print()
        for num, prob in inferred_topics:
            print('Topic {}. Confidence: {}'.format(num, prob))
            if args.model_type == 'LDA':
                print('Keywords {}'.format(topics[num]))
            elif args.model_type == 'LLDA' or args.model_type == 'PLDA' or args.model_type == 'SLDA':
                print(topics[num][1])
                print('Keywords {}'.format(topics[num][0]))
                # print('Keywords {}'.format(topics[num]))

        print()
        print('----------')
        val = input("Please enter your decided label below: ")
        print('----------')

        if val.isdigit():
            user_input = int(val)
            if user_input >= 0 and user_input < len(topics):
                print()
                print('User input label is {}'.format(user_input))
                print()
                session.label(random_document, user_input)
                doc_count += 1


if __name__ == "__main__":
    main()