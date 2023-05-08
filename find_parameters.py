import tomotopy as tp
import itertools
import numpy as np
import pickle 

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

    return best_hyperparameters


    return best_hyperparameters

if __name__ == '__main__':
    file_path = './Data/newsgroup_sub_500_processed.pkl'

    alpha_range = np.arange(0.1, 1.1, 0.1)
    eta_range = np.arange(0.1, 1.1, 0.1)
    k_range = range(2, 21)
    min_cf_range = range(1, 6)
    min_df_range = range(1, 6)
    iterations_range = range(100, 1100, 100)

    best_hyperparameters = optimize_hyperparameters(file_path, alpha_range, eta_range, k_range, min_cf_range, min_df_range, iterations_range)
    print(f'Best hyperparameters: {best_hyperparameters}')


