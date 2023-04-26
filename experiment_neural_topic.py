import json
import pandas as pd
import os
import pickle

model_path = './Model/ETM.pkl'

with open('./Data/newsgroup_sub_1000.pkl', 'rb') as inp:
        processed_data = pickle.load(inp)

documents = [' '.join(doc) for doc in processed_data]

with open(model_path, 'rb') as inp:
    etm_instance = pickle.load(inp)

# topics = etm_instance.get_topics(20)
# topic_coherence = etm_instance.get_topic_coherence()
# topic_diversity = etm_instance.get_topic_diversity()
# topic_dist = etm_instance.get_topic_word_dist()

# for i, ele in enumerate(topics):
#     print('-'*20)
#     print('Topic {}'.format(i))
#     print(ele)
#     print('-'*20)
#     print()

# print(etm_instance.perplexity())
# print(etm_instance.test_perplexity)

