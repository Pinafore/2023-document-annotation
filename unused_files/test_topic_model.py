import pandas as pd

# data_path = '/fs/clip-quiz/amao/Github/alto-boot/data/nist_ras_documents_cleaned.csv'
data_path = 'Nist.json'

df = pd.read_json(data_path)


labels = df['label']
from topic_model import TopicModel

# %load_ext autoreload
# %autoreload 2

# document_data = df.drop(['dominant_topic','dominant_topic_percent','topic_keywords'], axis=1)
# documents = df['text']
# topic_model_path = 'data'

# print()
# print('262')
# print(documents[262])
# print()

topic_model = TopicModel(corpus_path=data_path, model_type='LDA', min_num_topics= 5, num_iters= 1)
# topic_model.load_model('data')

# make partial labels
labels = df['label']
N = 100
partial_labels = [None for _ in labels]
for i in range(N):
    partial_labels[i] = labels[i]
# labels = list(df.tag_1)

topic_model.preprocess(5, 100)
print('Finished preprocessing data')

topic_model.train(num_topics=12)
topic_model.print_topics()
