import json
import pandas as pd
import os
import pickle


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

model_path = './Model/ETM.pkl'

with open('./Data/newsgroup_sub_1000.pkl', 'rb') as inp:
        processed_data = pickle.load(inp)

documents = [' '.join(doc) for doc in processed_data]

with open(model_path, 'rb') as inp:
    etm_instance = pickle.load(inp)


doc_topic = etm_instance.get_document_topic_dist()

group_docs_to_topics(doc_topic)




# print(doc_topic[0])
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

