'''
Verify the number of topics for the trained model is as expected
'''

from spacy_topic_model import TopicModel
from Neural_Topic_Model import Neural_Model
import argparse

def main():
    # __init__(self, num_topics, model_type, load_data_path, train_len)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_type", help="type of the model",
                       type=str, default='LDA', required=False)
    
    argparser.add_argument("--num_topics", help="num of topics",
                       type=int, default=20, required=False)

    argparser.add_argument("--print_topics", help="print topics",
                       type=bool, default=False, required=False)

    args = argparser.parse_args()
    

    doc_dir = './Data/newsgroup_sub_500.json'
    processed_doc_dir = './Data/newsgroup_sub_500_processed.pkl'

    if args.model_type == 'ETM':
        model = Neural_Model('./Model/ETM_{}.pkl'.format(args.num_topics), processed_doc_dir, doc_dir)
    else:
        model = TopicModel('./Model/{}_{}.pkl'.format(args.model_type, args.num_topics), args.model_type, doc_dir, args.num_topics)
    

    topics = model.print_topics(verbose=args.print_topics)
    print(list(model.document_probas.keys()))
    print(len(list(model.document_probas.keys())))

    

if __name__ == "__main__":
    main()