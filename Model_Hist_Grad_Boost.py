### Hist Gradient Boosting Classifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from matplotlib.pylab import rcParams
import time

# Initial Model

clf = HistGradientBoostingClassifier().fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_train_cf = confusion_matrix(y_train, y_pred_train)
f1 = f1_score(y_train, y_pred_train, zero_division=1)
score_train = balanced_accuracy_score(y_train, y_pred_train)
print('Accuracy:' + score_train)
print(y_train_cf)
print(f1)

# Model Tuning
from sklearn.model_selection import GridSearchCV
import multiprocessing

n_jobs = multiprocessing.cpu_count()

param_grid = {
    'learning_rate': (0.01, 0.1, 1, 10),
    'max_leaf_nodes': (3, 10, 30),
    'max_depth': (2,4,6,8),
    'random_state': [42]}

st = time.time() # Beginning time

model_grid_search = GridSearchCV(clf, param_grid=param_grid,
                                 n_jobs=n_jobs, cv=5)
model_grid_search.fit(X_train, y_train)

et = time.time() # End time

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

# Accuracy
accuracy = model_grid_search.score(X_train, y_train)
print(
    f"The test accuracy score of the grid-searched pipeline is: "
    f"{accuracy:.2f}"
)

model_grid_search.predict(X_train.iloc[0:5])
print(f"The best set of parameters is: "
      f"{model_grid_search.best_params_}")

cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values(
    "mean_test_score", ascending=False)
cv_results.head()

# get the parameter names
column_results = [f"param_{name}" for name in param_grid.keys()]
column_results += [
    "mean_test_score", "std_test_score", "rank_test_score"]
cv_results = cv_results[column_results]

def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name

cv_results = cv_results.rename(shorten_param, axis=1)
cv_results

y_pred_train = model_grid_search.predict(X_train.iloc[0])
f1 = f1_score(y_train, y_pred_train, zero_division=1)
print(f1)












