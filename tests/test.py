from surprise import KNNBaseline
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
import numpy as np

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')
sizefinder = data.build_full_trainset()
# trainset, testset = train_test_split(data, test_size=1.0)
trainset, testset = train_test_split(data, test_size=.2)

# Use the famous SVD algorithm.
algo = KNNBaseline()
s = (sizefinder.n_users+1,sizefinder.n_items+1)
weight = np.ones(s)
algo.weightUpdate(weight)
predictions = algo.fit(sizefinder).test(testset)





print('total number of users: ',trainset.n_users)
print('testset element',testset[2][1])
# Run 5-fold cross-validation and print results.
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print predictions[10]
print predictions[10][3]