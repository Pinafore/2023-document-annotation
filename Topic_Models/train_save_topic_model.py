import pickle
import argparse
from topic_model import Topic_Model
from embedded_topic_model.utils import preprocessing
import pickle
import argparse
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


'''
Returns
doc_prob_topic: D X T matrix, where D is is the number of documents
T is the number of topics from the topic model. For each document, it
contains a list of topic probabilities.

topics_probs: A dictionary, for each topic key, contains a list of tuples
[(dic_id, probability)...]. Each tuple has a document id, representing 
the row id of the document, and probability the document belong to this topic.
For each topic, it only contains a list of documents that are the most likely 
associated with that topic.
'''
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

    print(topics_probs.keys())
    
    return topics_probs, doc_prob_topic


'''
Calculate the coherence of the topic model given a list of keywords 
produced by the topic model, and the corpus (after data processing)
'''
def get_coherence(keywords, corpus):
    dictionary = Dictionary(corpus)

    coherence_model = CoherenceModel(
        topics=keywords,
        texts=corpus,
        dictionary=dictionary,
        coherence='u_mass'
    )

    coherence_score = coherence_model.get_coherence()

    # print("C_v coherence score:", coherence_score)
    return coherence_score


'''
Initialize and save the neural topic model
'''
def create_neural_model_and_save(num_topics, iters, dataset, train_length):

    # The path to save the Topic Model
    save_model_path = './Model/ETM_{}.pkl'.format(num_topics)

    # Load the processed dataset
    with open(dataset, 'rb') as inp:
        saved_data = pickle.load(inp)

    
    processed_data = saved_data['datawords_nonstop'][0:train_length]
    spans = saved_data['spans'][0:train_length]
    
    
    documents = [' '.join(doc) for doc in processed_data]


    
    # Create an ETM dataset to train the model
    vocabulary, train_dataset, test_dataset, = preprocessing.create_etm_datasets(
        documents, 
        min_df=0, 
        max_df=10000, 
        train_size=1.0
    )
    
    
    print('length of dataset after using the ETM instance', len(train_dataset['tokens']))
    
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
        train_embeddings=False, # Optional. If True, ETM will learn word embeddings jointly with
                                # topic embeddings. By default, is False. If 'embeddings' argument
                                # is being passed, this argument must not be True
        eval_perplexity = False
    )


    # print(train_dataset)
    # exit(0)

    etm_instance.fit(train_data = train_dataset)

    topics = etm_instance.get_topics(20)
    

    for i, ele in enumerate(topics):
        print('-'*20)
        print('Topic {}'.format(i))
        print(ele)
        print('-'*20)
        print()


    coherence = get_coherence(topics, processed_data)
    print('coherence: ', coherence)

    print('number of topics {}'.format(len(topics)))
    doc_topic = etm_instance.get_document_topic_dist()
    doc_topic = doc_topic.cpu().numpy()
    
    print(doc_topic.shape)
    print(doc_topic[1])
    document_probas,  doc_topic_probas = group_docs_to_topics(doc_topic)


    # Save the top 30 keywords for each topic
    model_topics = etm_instance.get_topics(30)
    
    result = {}
    
    result['model_topics'] = model_topics
    result['document_probas'] = document_probas
    result['doc_topic_probas'] = doc_topic_probas
    result['spans'] = spans
    result['get_document_topic_dist'] = doc_topic
    result['datawords_nonstop'] = processed_data
    result['spans'] = spans
    result['texts'] = saved_data['texts']
    result['topic_word_dist'] = etm_instance.get_topic_word_dist().cpu().numpy()
    result['vocabulary'] = etm_instance.vocabulary
    result['cogerence'] = coherence
    
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
                       type=str, default='./Data/congressional_bill_train_processed.pkl', required=False)


    args = argparser.parse_args()
    
    with open(args.load_data_path, 'rb') as inp:
        saved_data = pickle.load(inp)
        full_len = len(saved_data['datawords_nonstop'])

    if args.model_type == 'LDA' or args.model_type == 'SLDA':
        Model = Topic_Model(args.num_topics, args.num_iters, args.model_type, args.load_data_path, full_len, {}, False, None)
        
        save_path = './Model/{}_{}.pkl'.format(args.model_type, args.num_topics)
        Model.train(save_path)
    elif args.model_type == 'ETM':
        create_neural_model_and_save(args.num_topics, args.num_iters, args.load_data_path, full_len)
    

if __name__ == "__main__":
    main()