## Model synthetic experiment result details

Note:
for 16 topics, 20 topics synthetic experiments, the topic models are all run on the 20newsgroup train dataset.

The goal is to test the topic model active learning performance when the number of topics match the number of unique 
labels in the labelset and when they don't match.

filenames and corresponding experiments as follows

### Active Learning Algorigthmn

baseline.npy: 
ran the active learning on 300 documents labeled (training set)

baseline_2000_docs_5_runs_med.npy: 
ran the active learning on 2000 documents 5 times and take the medium of each run (training set)

baseline_1000_docs.npy: 
ran the active learning on 1000 documents labeled (training set)

baseline_test_docs_5_runs.npy: 
ran the active learning on the whole test dataset 5 times, recorded each run's all stats

20_topics_active_median.npy: 
ran active learning in 300 documents over five runs, take the median (training set)

baseline_test_docs_regularized_one_run: 
Ran active learning on all test documents and use training set as test set to see test set performance

baseline_test_docs_bacth_update_one_run: 
Ran active learning on all test documents, batch update for every 100 documents

### LDA Baseline Algorithm

16_topics_LDA_1000_docs.npy: 
Ran LDA on 1000 documents labeled (training set)

16_topics_LDA_topic_features_only.npy: 
Ran LDA on 300 documents labeled with only topic features (training set)

16_topics_LDA.npy: 
Ran LDA on 300 documents labeled (training set)

20_topics_LDA_median.npy: 
Ran LDA on 300 documents labeled over 5 runs median (training set)

20_topics_LDA.npy: 
Ran LDA on 300 documents labeled (training set)

50_topics_LDA_topic_features_only.npy: 
Ran LDA on 300 documents labeled (training set)

LDA_test_docs_5_runs_topics_features_only_16_topics.npy: 
Ran LDA on the whole testset over 5 runs median

16_topics_LDA_test_docs_regularized_one_run.npy: 
Ran LDA on all test documents and use training set as test set to see test set performance

16_topics_LDA_test_docs_bacth_update_one_run: 
Ran LDA on all test documents, batch update clf for every 100 documents

### sLDA algorithm

16_topics_SLDA_1000.npy: 
Ran SLDA on 1000 documents for 16 topics (training set)

16_topics_SLDA_topic_features_only.npy: 
Ran SLDA on 300 documents for 16 topics (training set)

16_topics_SLDA.npy: 
Ran SLDA on 300 documents for 16 topics (training set)

20_topics_SLDA.npy: 
Ran SLDA on 300 documents for 20 topics (training set)

50_topics_SLDA_topic_features_only.npy: 
Ran SLDA on 300 documents for 50 topics (training set)

20_topics_SLDA_median.npy: 
Ran SLDA on 300 documents for 20 topics over 5 run median (training set)

16_topics_SLDA_test_docs_regularized_one_run.npy: 
Ran SLDA on all test documents and use training set as test set to see test set performance

16_topics_SLDA_test_docs_bacth_update_one_run: 
Ran SLDA on all test documents, batch update clf for every 100 documents

### CTM algorithm

16_topics_CTM_topic_features_only.npy: 
Ran CTM on 300 documents for 16 topics (training set)

16_topics_CTM.npy: 
Ran CTM on 300 documents for 16 topics (training set)

20_topics_CTM.npy: 
Ran CTM on 300 documents for 20 topics (training set)

50_topics_CTM.npy: 
Ran CTM on 300 documents for 50 topics (training set)

16_topics_CTM_test_docs_regularized_one_run.npy: 
Ran CTM on all test documents and use training set as test set to see test set performance

16_topics_CTM_test_docs_bacth_update_one_run: 
Ran CTM on all test documents, batch update clf for every 100 documents