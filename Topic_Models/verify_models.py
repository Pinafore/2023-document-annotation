'''
Verify the number of topics for the trained model is as expected
'''

# from spacy_topic_model import TopicModel
from topic_model import Topic_Model
from Neural_Topic_Model import Neural_Model
import argparse
import pickle
import pandas as pd

def main():
    # __init__(self, num_topics, model_type, load_data_path, train_len)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_type", help="type of the model",
                       type=str, default='LDA', required=False)
    
    argparser.add_argument("--num_topics", help="num of topics",
                       type=int, default=20, required=False)

    argparser.add_argument("--print_topics", help="print topics",
                       type=bool, default=False, required=False)
    
    argparser.add_argument("--model_path", help="path of the model",
                       type=str, default='none', required=False)
    
    argparser.add_argument("--print_docs", help="get document for each topic",
                       type=bool, default=False, required=False)
    
    argparser.add_argument("--doc_path", help="path of the raw text documents",
                       type=str, default='./Data/newsgroups/newsgroup_test.json', required=False)

    args = argparser.parse_args()
    

    # doc_dir = './Data/newsgroup_sub_500.json'
    # processed_doc_dir = './Data/newsgroup_sub_500_processed.pkl'
    doc_dir = './Data/CongressionalBill/congressional_bills.json'
    processed_doc_dir = './Data/congressional_bill_processed.pkl'

    if args.model_type == 'ETM':
        model = Neural_Model('./Model/ETM_{}.pkl'.format(args.num_topics), processed_doc_dir, doc_dir)
    elif args.model_type == 'CTM' or args.model_type == 'SCTM' or args.model_type == 'Bertopic':
        model = Neural_Model('./Model/{}_{}.pkl'.format(args.model_type, args.num_topics), processed_doc_dir, doc_dir)
    else:
        if args.model_path == 'none':
            # model = TopicModel('./Model/{}_{}.pkl'.format(args.model_type, args.num_topics), args.model_type, doc_dir, args.num_topics)
            model = Topic_Model(args.num_topics, 0, args.model_type, processed_doc_dir, 258, {}, True, './Model/{}_{}.pkl'.format(args.model_type, args.num_topics))
        else:
            model = Topic_Model(args.num_topics, 0, args.model_type, processed_doc_dir, 258, {}, True, args.model_path)

    topics = model.print_topics(verbose=args.print_topics)
    if args.print_docs == True:
        df = pd.read_json(args.doc_path)
        texts = df.text.values.tolist()
        with open('./Model/{}_{}.pkl'.format(args.model_type, args.num_topics), 'rb') as inp:
            loaded_data = pickle.load(inp)
            topic_document_probas = loaded_data['document_probas']

        '''
        Print the top 3 documents for each topic
        '''
        i = 0
        for topic, docs in topic_document_probas.items():
            for idx in range(3):
                print('*'*30)
                print('keywords:', topics[i])
                print('DOCUMENT ', docs[idx][0])
                print(texts[docs[idx][0]])
                print('END OF DOCUMENT ', docs[idx][0])
                print('*'*30)


            print('*'*30)
            i += 1

    print(list(model.document_probas.keys()))
    print(len(list(model.document_probas.keys())))

    with open('./Model/{}_{}.pkl'.format(args.model_type, args.num_topics), 'rb') as inp:
        loaded_data = pickle.load(inp)

    print('cv coherence', loaded_data['coherence_cv'])
    print('umass coherence', loaded_data['coherence_umass'])
    

if __name__ == "__main__":
    main()