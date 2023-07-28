import subprocess
import pickle
import tomotopy as tp
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import subprocess
import tomotopy as tp
from topic_model import Topic_Model
from Neural_Topic_Model import Neural_Model


def determine_best_lda_params(model_type):
    coherences = []
    doc_dir = './Data/newsgroup_sub_500_processed.pkl'
    for num_topics in range(5, 50):
        train_len = 500
        
        best_coherence_score = -1
        best_num_iters = -1
        
        num_iters_list = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
        
        for num_iters in num_iters_list:
            # Call the command-line program to create the classical LDA model
            if model_type == 'LDA' or model_type == 'SLDA':
                result = subprocess.run(["python", "create_classical_model.py",
                                        "--num_topics", str(num_topics),
                                        "--train_len", str(train_len),
                                        "--num_iters", str(num_iters),
                                        "--model_type", model_type,
                                        "--load_data_path", doc_dir],
                                        capture_output=True, text=True)
            elif model_type == 'ETM':
                result = subprocess.run(["python", "neural_topic.py",
                                        "--num_topics", str(num_topics),
                                        "--train_len", str(train_len),
                                        "--num_iters", str(num_iters),
                                        "--model_type", model_type,
                                        "--load_data_path", doc_dir],
                                        capture_output=True, text=True)
            
            # Load the trained LDA model
            if model_type == 'LDA' or model_type == 'SLDA':
                model = Topic_Model(num_topics, 0, model_type, doc_dir, train_len, {}, True, './Model/{}_{}.pkl'.format(model_type, num_topics))
            elif model_type == 'SLDA':
                model = Topic_Model(num_topics, 0, model_type, doc_dir, train_len, {}, True, './Model/{}_{}.pkl'.format(model_type, num_topics))
            elif model_type == 'ETM':
                model = Neural_Model('./Model/ETM_{}.pkl'.format(num_topics), doc_dir, doc_dir)
            
            # Compute coherence score
            # coherence_model = tp.coherence.Coherence(corpus=model, coherence="c_v", top_n=20)
            # coherence_score = coherence_model.get_score()
            
            coherence_score = model.get_coherence()
            print('Number of topics: {}, Number of iterations: {}, Coherence score: {}'.format(num_topics, num_iters, coherence_score))

            if coherence_score > best_coherence_score:
                best_coherence_score = coherence_score
                best_num_iters = num_iters
        
        coherences.append((num_topics, best_num_iters, best_coherence_score))
    
    return coherences

coherences = determine_best_lda_params('ETM')
print(coherences)
