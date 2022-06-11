### Hist Gradient Boosting Classifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from matplotlib.pylab import rcParams
import time
rcParams['figure.figsize'] = 12, 4

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
















