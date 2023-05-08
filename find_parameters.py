import tomotopy as tp
import itertools
import numpy as np
import pickle 
import tomotopy as tp
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

def calculate_perplexity(model):
    return model.perplexity

def calculate_coherence(model):
    coherence_model = tp.coherence.Coherence(
            corpus=model, coherence="c_v", top_n=10
        )
    
    return coherence_model.get_score()

def train_lda_model(alpha, eta, k, min_cf, min_df, corpus, iterations):
    model = tp.LDAModel(alpha=alpha, eta=eta, k=k, min_cf=min_cf, min_df=min_df)
    for doc in corpus:
        model.add_doc(doc)
    
    for _ in range(iterations):
        model.train(10)
    
    return model

def build_corpus(path):
    # Perform the random search
    with open(path, 'rb') as inp:
        loaded_data = pickle.load(inp)

    corpus = loaded_data['datawords_nonstop']
    return corpus

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

def objective(params, corpus, min_perplexity, max_perplexity, min_coherence, max_coherence, perplexity_weight, coherence_weight):
    # alpha, eta, k, min_cf, min_df, iterations = params
    alpha, eta, min_cf, min_df, iterations = params
    k = 20
    model = train_lda_model(alpha, eta, k, min_cf, min_df, corpus, iterations)
    perplexity = calculate_perplexity(model)
    coherence = calculate_coherence(model)

    # Since we want to minimize the objective function, we negate the coherence score
    # and add it to perplexity to get a single metric to optimize

    curr_parameters = {'alpha': alpha, 'eta': eta, 'k': k, 'min_cf': min_cf, 'min_df': min_df, 'iterations': iterations}
    print(f'Current hyperparameters: {curr_parameters}, perplexity: {perplexity}, coherence: {coherence}')
    # Normalize perplexity and coherence to [0, 1]
    normalized_perplexity = normalize(perplexity, min_perplexity, max_perplexity)
    normalized_coherence = normalize(coherence, min_coherence, max_coherence)

    # Combine normalized perplexity and coherence using weights
    combined_score = perplexity_weight * normalized_perplexity - coherence_weight * normalized_coherence
    return combined_score



if __name__ == '__main__':
    # ...

    space = [
        Real(0.1, 1.0, name='alpha'),
        Real(0.1, 1.0, name='eta'),
        # Integer(20, 20, name='k'),
        Integer(1, 5, name='min_cf'),
        Integer(1, 5, name='min_df'),
        Integer(100, 1000, name='iterations')
    ]

    file_path = './Data/newsgroup_sub_500_processed.pkl'
    corpus = build_corpus(file_path)

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

    res_gp = gp_minimize(
        func=lambda params: objective(params, corpus, min_perplexity, max_perplexity, min_coherence, max_coherence, perplexity_weight, coherence_weight),
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