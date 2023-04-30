import json
import pandas as pd
import os
import pickle


# with open('./Data/newsgroup_sub_500.pkl', 'rb') as inp:
#         processed_data = pickle.load(inp)


# result_set = set()
# for ele in processed_data:
#      for j in ele:
#         result_set.add(j)

# print(len(result_set))

# exit(0)


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

model_path = './Model/ETM_20.pkl'

with open('./Data/newsgroup_sub_500.pkl', 'rb') as inp:
        processed_data = pickle.load(inp)

documents = [' '.join(doc) for doc in processed_data]

with open(model_path, 'rb') as inp:
    saved_data = pickle.load(inp)

etm_instance = saved_data['model']

doc_topic = etm_instance.get_document_topic_dist()

group_docs_to_topics(doc_topic)



topic_word_dist = etm_instance.get_topic_word_dist().cpu().numpy()
print(topic_word_dist.shape)

print(len(etm_instance.vocabulary))

# print(saved_data['spans'][0])


topic_word_probas = {}
for i, ele in enumerate(topic_word_dist):
     topic_word_probas[i] = {}
     for word_idx, word_prob in enumerate(ele):
          topic_word_probas[i][etm_instance.vocabulary[word_idx]] = word_prob


print(topic_word_probas)
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

