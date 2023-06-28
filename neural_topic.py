# from embedded_topic_model.utils import preprocessing
from embedded_topic_model.utils import preprocessing
import json
import pandas as pd
import os
import pickle
import argparse
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


def group_docs_to_topics(model_inferred):
    doc_prob_topic = []
    doc_to_topics, topics_probs = {}, {}

    for doc_id, inferred in enumerate(model_inferred):
        doc_topics = list(enumerate(inferred))
        doc_prob_topic.append(inferred)

        doc_topics.sort(key = lambda a: a[1], reverse= True)

        doc_to_topics[doc_id] = doc_topics
        # print(doc_topics)
        if doc_topics[0][0] in topics_probs:
            topics_probs[doc_topics[0][0]].append((doc_id, doc_topics[0][1]))
        else:
            topics_probs[doc_topics[0][0]] = [(doc_id, doc_topics[0][1])]

    for k, v in topics_probs.items():
        topics_probs[k].sort(key = lambda a: a[1], reverse= True)

    print(topics_probs.keys())
    
    return topics_probs, doc_prob_topic

def get_coherence(keywords, corpus):
    dictionary = Dictionary(corpus)

    coherence_model = CoherenceModel(
        topics=keywords,
        texts=corpus,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()

    # print("C_v coherence score:", coherence_score)
    return coherence_score

def create_neural_model_and_save(num_topics, iters, dataset, train_length):

    # corpus_file = './Data/newsgroup_sub_500.json'
    save_model_path = './Model/ETM_{}.pkl'.format(num_topics)

    # with open(dataset, 'rb') as inp:
    #     saved_data = pickle.load(inp)

   
    # processed_data = saved_data['datawords_nonstop']
    # spans = saved_data['spans']
    with open(dataset, 'rb') as inp:
        saved_data = pickle.load(inp)

    # print(len(saved_data['datawords_nonstop']))
    # return
    processed_data = saved_data['datawords_nonstop'][0:train_length]

    # print(processed_data[0])
    spans = saved_data['spans'][0:train_length]
    
    
    documents = [' '.join(doc) for doc in processed_data]

    # print(documents)
    # exit(0)
    # Preprocessing the dataset
    vocabulary, train_dataset, test_dataset, = preprocessing.create_etm_datasets(
        documents, 
        min_df=0, 
        max_df=1.0, 
        train_size=1.0
    )
    
    # print(len(test_dataset['tokens']))
    # return
    # test_dataset = test_dataset['test']

    # print(train_dataset.keys())
    # print(test_dataset.keys())

    # exit(0)

    from embedded_topic_model.utils import embedding

    # Training word2vec embeddings
    embeddings_mapping = embedding.create_word2vec_embedding_from_dataset(documents)

    from embedded_topic_model.models.etm import ETM

    # Training an ETM instance
    etm_instance = ETM(
        vocabulary,
        embeddings=embeddings_mapping, # You can pass here the path to a word2vec file or
                                    # a KeyedVectors instance
        num_topics=num_topics,
        epochs=iters,
        debug_mode=True,
        train_embeddings=True, # Optional. If True, ETM will learn word embeddings jointly with
                                # topic embeddings. By default, is False. If 'embeddings' argument
                                # is being passed, this argument must not be True
        eval_perplexity = False
    )


    # print(train_dataset)
    # exit(0)

    etm_instance.fit(train_data = train_dataset)

    topics = etm_instance.get_topics(20)
    topic_coherence = etm_instance.get_topic_coherence(top_n=20)
    topic_diversity = etm_instance.get_topic_diversity()
    topic_dist = etm_instance.get_topic_word_dist()

    for i, ele in enumerate(topics):
        print('-'*20)
        print('Topic {}'.format(i))
        print(ele)
        print('-'*20)
        print()

    # print(topic_dist.shape)
    coherence = get_coherence(topics, processed_data)
    print('coherence: ', coherence)

    print('number of topics {}'.format(len(topics)))
    doc_topic = etm_instance.get_document_topic_dist()
    doc_topic = doc_topic.cpu().numpy()
    
    print(doc_topic.shape)
    print(doc_topic[1])
    document_probas,  doc_topic_probas = group_docs_to_topics(doc_topic)

    # processed_data = saved_data['datawords_nonstop']
    # spans = saved_data['spans']
    result = {}
    result['model'] = etm_instance
    result['document_probas'] = document_probas
    result['doc_topic_probas'] = doc_topic_probas
    result['spans'] = spans
    result['get_document_topic_dist'] = doc_topic
    result['datawords_nonstop'] = processed_data
    result['spans'] = spans
    result['texts'] = saved_data['texts']
    
    with open(save_model_path, 'wb+') as outp:
        pickle.dump(result, outp)


# file = './Data/newsgroup_sub_500_processed.pkl'
# file = './Data/nist_all_labeled.pkl'
# file = './Data/congressional_bill_processed.pkl'

# create_neural_model_and_save(20, file)
# create_neural_model_and_save(21, file)
# create_neural_model_and_save(22, file)

def main():
    # __init__(self, num_topics, model_type, load_data_path, train_len)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_topics", help="number of topics",
                       type=int, default=20, required=False)
    argparser.add_argument("--num_iters", help="number of iterations",
                       type=int, default=1000, required=False)
    argparser.add_argument("--model_type", help="type of the model",
                       type=str, default='ETM', required=False)
    argparser.add_argument("--load_data_path", help="Whether we LOAD the data",
                       type=str, default='./Data/congressional_bill_processed.pkl', required=False)
    argparser.add_argument("--train_len", help="number of training samples",
                       type=int, default=500, required=False)

    args = argparser.parse_args()
    
    # Model = Topic_Model(args.num_topics, args.num_iters, args.model_type, args.load_data_path, args.train_len, {}, False, None)
    # save_path = './Model/{}_{}.pkl'.format(args.model_type, args.num_topics)
    # Model.train(save_path)
    create_neural_model_and_save(args.num_topics, args.num_iters, args.load_data_path, args.train_len)

    

if __name__ == "__main__":
    main()