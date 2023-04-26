# from embedded_topic_model.utils import preprocessing
from embedded_topic_model.utils import preprocessing
import json
import pandas as pd
import os
import pickle

# corpus_file = './Data/newsgroup_sub_500.json'
save_model_path = './Model/ETM.pkl'

with open('./Data/newsgroup_sub_1000.pkl', 'rb') as inp:
        processed_data = pickle.load(inp)

# processed_data = processed_data[0:500]
# Loading a dataset in JSON format. As said, documents must be composed by string sentences
# documents_raw = json.load(open('all_emails.json', 'r'))
# documents_raw = documents_raw['Body']

# documents_raw = pd.read_json(corpus_file)
# documents_raw = documents_raw.text.values.tolist()
# documents = [document for document in documents_raw]

documents = [' '.join(doc) for doc in processed_data]

# Preprocessing the dataset
vocabulary, train_dataset, test_dataset, = preprocessing.create_etm_datasets(
    documents, 
    min_df=0.01, 
    max_df=0.75, 
    train_size=0.85
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
    num_topics=20,
    epochs=300,
    debug_mode=True,
    train_embeddings=False, # Optional. If True, ETM will learn word embeddings jointly with
                            # topic embeddings. By default, is False. If 'embeddings' argument
                            # is being passed, this argument must not be True
    eval_perplexity = True
)

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

print(topic_dist.shape)


with open(save_model_path, 'wb+') as outp:
    pickle.dump(etm_instance, outp)