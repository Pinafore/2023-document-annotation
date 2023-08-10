import argparse
from topic_model import Topic_Model
import pickle
from embedded_topic_model.utils import preprocessing

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

    return topics_probs, doc_prob_topic

def create_neural_model_and_save(num_topics, dataset):

    # corpus_file = './Data/newsgroup_sub_500.json'
    save_model_path = './Model/ETM_{}.pkl'.format(num_topics)

    # with open(dataset, 'rb') as inp:
    #     saved_data = pickle.load(inp)

   
    # processed_data = saved_data['datawords_nonstop']
    # spans = saved_data['spans']
    with open(dataset, 'rb') as inp:
        saved_data = pickle.load(inp)

    processed_data = saved_data['datawords_nonstop']

    # print(processed_data[0])
    spans = saved_data['spans']
    # processed_data = processed_data[0:500]
    # Loading a dataset in JSON format. As said, documents must be composed by string sentences
    # documents_raw = json.load(open('all_emails.json', 'r'))
    # documents_raw = documents_raw['Body']

    # documents_raw = pd.read_json(corpus_file)
    # documents_raw = documents_raw.text.values.tolist()
    # documents = [document for document in documents_raw]
    
    documents = [' '.join(doc) for doc in processed_data]

    # print(documents)
    # exit(0)
    # Preprocessing the dataset
    vocabulary, train_dataset, test_dataset, = preprocessing.create_etm_datasets(
        documents, 
        min_df=0.01, 
        max_df=0.75, 
        train_size=1.0
    )

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
        epochs=2000,
        debug_mode=True,
        train_embeddings=True, # Optional. If True, ETM will learn word embeddings jointly with
                                # topic embeddings. By default, is False. If 'embeddings' argument
                                # is being passed, this argument must not be True
        eval_perplexity = True
    )


    # print(train_dataset)
    # exit(0)

    etm_instance.fit(train_data = train_dataset, test_data=test_dataset)

    topics = etm_instance.get_topics(20)
    topic_coherence = etm_instance.get_topic_coherence()
    topic_diversity = etm_instance.get_topic_diversity()
    topic_dist = etm_instance.get_topic_word_dist()

    for i, ele in enumerate(topics):
        print('-'*20)
        print('Topic {}'.format(i))
        print(ele)
        print('-'*20)
        print()

    # print(topic_dist.shape)
    print('coherence: ', topic_coherence)

    doc_topic = etm_instance.get_document_topic_dist()
    doc_topic = doc_topic.cpu().numpy()
    
    # print(doc_topic.shape)
    # print(doc_topic[1])
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
    
def main():
    # __init__(self, num_topics, model_type, load_data_path, train_len)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_topics", help="number of topics",
                       type=int, default=20, required=False)
    argparser.add_argument("--num_iters", help="number of iterations",
                       type=int, default=1000, required=False)
    argparser.add_argument("--model_type", help="type of the model",
                       type=str, default='LDA', required=False)
    argparser.add_argument("--load_data_path", help="Whether we LOAD the data",
                       type=str, default='./Data/congressional_bill_processed.pkl', required=False)
    


    args = argparser.parse_args()
    
    with open(args.load_data_path, 'rb') as inp:
        saved_data = pickle.load(inp)
        full_len = len(saved_data['datawords_nonstop'])


    if args.model_type == 'ETM':
        create_neural_model_and_save(args.num_topics, args.load_data_path)
    else:
        Model = Topic_Model(args.num_topics, args.num_iters, args.model_type, args.load_data_path, full_len, {}, False, None)
        save_path = './Model/{}_{}.pkl'.format(args.model_type, args.num_topics)
        Model.train(save_path)
    

if __name__ == "__main__":
    main()