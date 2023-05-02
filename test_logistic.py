from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import accuracy_score

classifier = SGDClassifier(loss="log", penalty="l2", max_iter=1000, tol=1e-3, random_state=42,
                        learning_rate="adaptive", eta0=0.1)
incremental_classifier = SGDClassifier(loss="log", penalty="l2", max_iter=1000, tol=1e-3, random_state=42, learning_rate="adaptive", eta0=0.1, validation_fraction=0.2)

test_dataset_name = './Data/newsgroup_sub_1000.json'
train_len = 500
test_len = 88

test_df = pd.read_json(test_dataset_name)
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
vectorizer_idf = vectorizer.fit_transform(test_df['text'])
labels = test_df.label.values.tolist()

print(vectorizer_idf.shape)

train_set, train_label = vectorizer_idf[0:train_len], labels[0:train_len]
test_set, test_label = vectorizer_idf[train_len:train_len+test_len], labels[train_len:train_len+test_len]
classes = np.unique(labels)

# print(len(train_set))
print(train_set[0:5].shape)
print(np.shape(classes))

classifier_train_acc, classifier_test_acc = [], []
incremental_train_acc, incremental_test_acc = [], []


for i in range(train_len):
    if len(np.unique(train_label[0:i])) >= 2:
        classifier.fit(train_set[0:i+1], train_label[0:i+1])
        training_pred = classifier.predict(train_set[0:i+1])
        training_acc = accuracy_score(labels[0:i+1], training_pred)

        testing_pred = classifier.predict(test_set)
        test_acc = accuracy_score(test_label, testing_pred)

        incremental_classifier.partial_fit(train_set[0:i+1], train_label[0:i+1], classes)
        incremental_training_pred = incremental_classifier.predict(train_set[0:i+1])
        incremental_training_acc = accuracy_score(labels[0:i+1], incremental_training_pred)

        incremental_testing_pred = incremental_classifier.predict(test_set)
        incremental_testing_acc = accuracy_score(test_label, incremental_testing_pred)

        classifier_train_acc.append(training_acc)
        classifier_test_acc.append(test_acc)
        incremental_train_acc.append(incremental_training_acc)
        incremental_test_acc.append(incremental_testing_acc)


        print(i+1)
        print('training acc {}'.format(training_acc))
        print('testing acc {}'.format(test_acc))
        print('incremental training acc {}'.format(incremental_training_acc))
        print('incremental testing acc {}'.format(incremental_testing_acc))


np.save('./np_files/classifier_results.npy', [classifier_train_acc, classifier_test_acc, incremental_train_acc, incremental_test_acc])