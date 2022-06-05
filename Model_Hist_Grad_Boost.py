### Hist Gradient Boosting Classifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

clf = HistGradientBoostingClassifier().fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_train_cf = confusion_matrix(y_train, y_pred_train)
f1 = f1_score(y_train, y_pred_train, zero_division=1)
print('Accuracy:' + score_train)
print(y_train_cf)
print(f1)



### KNN Model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)
pred_prob = knn.predict_proba(X_test)
print(confusion_matrix(y_test['Default_Pred'], pred))
print(classification_report(y_test['Default_Pred'], pred))

