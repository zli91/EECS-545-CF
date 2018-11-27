from surprise import KNNBaseline
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.25)
# Use the famous SVD algorithm.
algo = KNNBaseline()
weight = 1
algo.weightUpdate(weight)
predictions = algo.fit(trainset).test(testset)
# Run 5-fold cross-validation and print results.
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print predictions[10]
print predictions[10][2]