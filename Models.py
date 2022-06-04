from sklearn import model_selection
from sklearn.metrics import confusion_matrix
### Splitting into train/test sets

train_data, valid_data = model_selection.train_test_split(train_df,
                                                          test_size=0.4,
                                                          random_state=42,
                                                          shuffle=True,
                                                          stratify=train_df['target'])

X_train = train_data.drop(['customer_ID', 'S_2', 'target'], axis=1)
y_train = train_data['target']

X_test = valid_data.drop(['customer_ID', 'S_2', 'target'], axis=1)
y_test = valid_data['target']

X_real_test = test_df.drop(['customer_ID', 'S_2'], axis=1)

train_data.shape, valid_data.shape

### Hist Gradient Boosting Classifier
from sklearn.ensemble import HistGradientBoostingClassifier

clf = HistGradientBoostingClassifier().fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_train_cf = confusion_matrix(y_train, y_pred_train)
y_test_cf = confusion_matrix(y_test, y_pred_test)
score_train = clf.score(X_train, y_train)
score_test = clf.score(X_test, y_test)
print(score_test)
print(score_train)
print(y_train_cf)
print(y_test_cf)

y_pred_hgbm = clf.predict(X_real_test)

### KNN Model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)
pred_prob = knn.predict_proba(X_test)
print(confusion_matrix(y_test['Default_Pred'], pred))
print(classification_report(y_test['Default_Pred'], pred))

