import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import tomotopy as tp
import pickle


class TomotopyLDASklearnWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, alpha=0.1, eta=0.01, iterations=1000):
        self.k = k
        self.alpha = alpha
        self.eta = eta
        self.iterations = iterations
        self.model = None

    def fit(self, X, y=None):
        self.model = tp.LDAModel(k=self.k, alpha=self.alpha, eta=self.eta)
        for doc in X:
            self.model.add_doc(doc)

        for _ in range(self.iterations):
            self.model.train(1)
        
        return self

    def transform(self, X, y=None):
        topic_distributions = []
        for doc in X:
            topic_dist = self.model.infer(self.model.make_doc(doc))[0]
            topic_distributions.append(topic_dist)
        return np.array(topic_distributions)


# Define the coherence scorer
def coherence_scorer(model):
    coherence_model = tp.coherence.Coherence(model, coherence="c_v", top_n=10)
    return coherence_model.get_score()


# Preprocess the data and split it into training and validation sets
# ...

# Define the parameter grid for random search
param_grid = {
    'k': np.arange(5, 51, 5),
    'alpha': np.logspace(-3, 0, 10),
    'eta': np.logspace(-3, 0, 10),
    'iterations': [500, 1000, 2000],
}

# Create a TomotopyLDASklearnWrapper instance
wrapper = TomotopyLDASklearnWrapper()

# Initialize RandomizedSearchCV with the wrapper, parameter grid, and coherence scorer
random_search = RandomizedSearchCV(
    wrapper,
    param_distributions=param_grid,
    n_iter=50,
    scoring=make_scorer(coherence_scorer, greater_is_better=True),
    cv=3,
    n_jobs=-1,
    verbose=2,
)

# Perform the random search
with open('./Data/newsgroup_sub_500_processed.pkl', 'rb') as inp:
    loaded_data = pickle.load(inp)
random_search.fit(loaded_data['datawords_nonstop'])

# Retrieve the best hyperparameters
best_params = random_search.best_params_
print("Best hyperparameters:", best_params)

