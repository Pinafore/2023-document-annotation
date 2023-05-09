import tomotopy as tp
import itertools
import numpy as np
import pickle 
import tomotopy as tp
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from tomotopy.utils import Corpus
import argparse

def calculate_perplexity(model):
    return model.perplexity

def calculate_coherence(model):
    coherence_model = tp.coherence.Coherence(
            corpus=model, coherence="c_v", top_n=10
        )
    
    return coherence_model.get_score()

def train_lda_model(alpha, eta, k, min_cf, min_df, corpus, iterations):
    model = tp.LDAModel(alpha=alpha, eta=eta, k=k, min_cf=min_cf, min_df=min_df)
    model.add_corpus(corpus)
    
    for _ in range(iterations):
        model.train(10)
    
    return model

def train_supervised_lda_model(alpha, eta, k, min_cf, min_df, vars, glm_param, nu_sq, corpus, iterations):
    model = tp.SLDAModel(alpha=alpha, eta=eta, k=k, min_cf=min_cf, min_df=min_df, vars=vars, glm_param=glm_param, nu_sq=nu_sq)
    model.add_corpus(corpus)
    for i in range(iterations):
        # model.train(10, workers=4)
        if i % 50 == 0:
            print(i)
        model.train(10)
    
    return model

def build_corpus(path, model_type):
    # Perform the random search
    with open(path, 'rb') as inp:
        loaded_data = pickle.load(inp)

    corpus, labels = loaded_data['datawords_nonstop'], loaded_data['labels']
    
    result_corpus = tp.utils.Corpus()
    if model_type =='SLDA':
        label_set = np.unique(labels)
        label_dict = dict()
        for i,label in enumerate(label_set):
            label_dict[label] = i
        for i, ngrams in enumerate(corpus):
            y = [0 for _ in range(len(label_set))]
            null_y = [np.nan for _ in range(len(label_set))]
            # y = [3 for i in range(20)]
            if labels and not labels[i] == 'None':
                label = labels[i]
                y[label_dict[label]] = 1
                result_corpus.add_doc(ngrams, y=y)
                # print(y)
            else:
                result_corpus.add_doc(ngrams, y=null_y)
            # result_corpus.add_doc(ngrams, labels= labels[i])
    else: 
        for i, ngrams in enumerate(corpus):
            result_corpus.add_doc(ngrams)
    return result_corpus

def optimize_hyperparameters(document_path, alpha_range, eta_range, k_range, min_cf_range, min_df_range, iterations_range):
    corpus = build_corpus(document_path)
    best_hyperparameters = None
    best_perplexity = float('inf')
    best_coherence = float('-inf')

    for alpha, eta, k, min_cf, min_df, iterations in itertools.product(alpha_range, eta_range, k_range, min_cf_range, min_df_range, iterations_range):
        model = train_lda_model(alpha, eta, k, min_cf, min_df, corpus, iterations)
        perplexity = calculate_perplexity(model)
        coherence = calculate_coherence(model)


        if perplexity < best_perplexity and coherence > best_coherence:
            best_hyperparameters = {'alpha': alpha, 'eta': eta, 'k': k, 'min_cf': min_cf, 'min_df': min_df, 'iterations': iterations}
            best_perplexity = perplexity
            best_coherence = coherence
            print(f'New best hyperparameters: {best_hyperparameters}, perplexity: {best_perplexity}, coherence: {best_coherence}')
        else:
            curr_parameters = {'alpha': alpha, 'eta': eta, 'k': k, 'min_cf': min_cf, 'min_df': min_df, 'iterations': iterations}
            print(f'Current hyperparameters: {curr_parameters}, perplexity: {perplexity}, coherence: {coherence}')

    return best_hyperparameters


    return best_hyperparameters

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)



def objective(params, corpus, min_perplexity, max_perplexity, min_coherence, max_coherence, perplexity_weight, coherence_weight, model_type):
    # alpha, eta, k, min_cf, min_df, iterations = params
    k = 20
    if model_type == 'LDA':
        alpha, eta, min_cf, min_df, iterations = params

        # print('alpha is {}, eta is {}'.format(alpha, eta))
        model = train_lda_model(alpha, eta, k, min_cf, min_df, corpus, iterations)
    elif model_type =='SLDA':
        # print('getting into objective')
        alpha, eta, min_cf, min_df, iterations, var_type, glm_param, nu_sq = params
        # vars = list(vars)
        # print('vars is {}'.format(vars))
        print('number of iteratons {}'.format(iterations))
        vars = [var_type for _ in range(k)]
        model = train_supervised_lda_model(alpha, eta, k, min_cf, min_df, vars, glm_param, nu_sq, corpus, iterations)
        print('Finished training...')

    perplexity = calculate_perplexity(model)
    coherence = calculate_coherence(model)

    # Since we want to minimize the objective function, we negate the coherence score
    # and add it to perplexity to get a single metric to optimize
    if model_type == 'LDA':
        curr_parameters = {'alpha': alpha, 'eta': eta, 'k': k, 'min_cf': min_cf, 'min_df': min_df, 'iterations': iterations}
        print(f'Current hyperparameters: {curr_parameters}, perplexity: {perplexity}, coherence: {coherence}')
    elif model_type =='SLDA':
        curr_parameters = {'alpha': alpha, 'eta': eta, 'k': k, 'min_cf': min_cf, 'min_df': min_df, 'iterations': iterations, 'var': var_type, 'glm': glm_param, 'nu_sq': nu_sq}
        print(f'Current hyperparameters: {curr_parameters}, perplexity: {perplexity}, coherence: {coherence}')
    # Normalize perplexity and coherence to [0, 1]
    normalized_perplexity = normalize(perplexity, min_perplexity, max_perplexity)
    normalized_coherence = normalize(coherence, min_coherence, max_coherence)

    # Combine normalized perplexity and coherence using weights
    combined_score = perplexity_weight * normalized_perplexity - coherence_weight * normalized_coherence
    return combined_score



if __name__ == '__main__':
    # ...
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", help="number of topics",
                       type=str, default='LDA', required=False)
    args = argparser.parse_args()
    MODEL = args.model

    k = 20
    file_path = './Data/newsgroup_sub_500_processed.pkl'
    corpus = build_corpus(file_path, MODEL)

    def create_callback():
        call_counter = 0

        def print_iteration_number(res):
            nonlocal call_counter
            call_counter += 1
            print(f'Iteration: {call_counter}')
            print('-------------------')

        return print_iteration_number
    
    min_perplexity = 10
    max_perplexity = 1500
    min_coherence = -1
    max_coherence = 1
    perplexity_weight = 0.2
    coherence_weight = 0.8


    if MODEL == 'LDA':
        space = [
            Real(0.1, 1.0, name='alpha'),
            Real(0.1, 1.0, name='eta'),
            # Integer(20, 20, name='k'),
            Integer(1, 5, name='min_cf'),
            Integer(1, 5, name='min_df'),
            Integer(50, 250, name='iterations')
        ]

        res_gp = gp_minimize(
            func=lambda params: objective(params, corpus, min_perplexity, max_perplexity, min_coherence, max_coherence, perplexity_weight, coherence_weight, MODEL),
            dimensions=space,
            n_calls=50,  # Number of iterations, increase for better results at the cost of longer runtime
            random_state=0,
            verbose=True,
            callback=[create_callback()]
        )

        best_hyperparameters = {
            'alpha': res_gp.x[0],
            'eta': res_gp.x[1],
            # 'k': res_gp.x[2],
            'min_cf': res_gp.x[2],
            'min_df': res_gp.x[3],
            'iterations': res_gp.x[4]
        }

        print(f'Best hyperparameters: {best_hyperparameters}')
    elif MODEL == 'SLDA':
        print('SLDA')
     
        space = [
        Real(0.1, 1.0, name='alpha'),
        Real(0.1, 1.0, name='eta'),
        # Integer(2, 20, name='k'),
        Integer(1, 5, name='min_cf'),
        Integer(1, 5, name='min_df'),
        Integer(50, 250, name='iterations'),
        # Categorical([tuple(np.random.uniform(0.01, 10.0, 20)) for _ in range(10)], name='vars'),  # Change 20 to the maximum number of topics you expect
        # [Categorical(['l', 'b'], name=f'vars_{i}') for i in range(k)],
        Categorical(['l', 'b'], name='vars'),
        Real(0.1, 10.0, name='glm_param'),
        Real(0.01, 10.0, name='nu_sq')
        ]

        def create_callback():
            call_counter = 0

            def print_iteration_number(res):
                nonlocal call_counter
                call_counter += 1
                print(f'Iteration: {call_counter}')
                print('-------------------')

            return print_iteration_number

        min_perplexity = 10
        max_perplexity = 2000
        min_coherence = -1
        max_coherence = 1
        perplexity_weight = 0.2
        coherence_weight = 0.8

        res_gp = gp_minimize(
            func=lambda params: objective(params, corpus, min_perplexity, max_perplexity, min_coherence, max_coherence, perplexity_weight, coherence_weight, MODEL),
            dimensions=space,
            n_calls=100,  # Number of iterations, increase for better results at the cost of longer runtime
            random_state=0,
            verbose=True,
            callback=[create_callback()]
        )

        best_hyperparameters = {
            'alpha': res_gp.x[0],
            'eta': res_gp.x[1],
            # 'k': res_gp.x[2],
            'min_cf': res_gp.x[2],
            'min_df': res_gp.x[3],
            'iterations': res_gp.x[4],
            'var': res_gp.x[5],
            'glm_param': res_gp.x[6],
            'nu_sq': res_gp.x[7]
        }

        print(f'Best hyperparameters: {best_hyperparameters}')
# LDA: Best hyperparameters: {'alpha': 0.1, 'eta': 1.0, 'min_cf': 4, 'min_df': 5, 'iterations': 173}