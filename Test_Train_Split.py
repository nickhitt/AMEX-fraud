from sklearn import model_selection

### Splitting into train/test sets

train_data, valid_data = model_selection.train_test_split(train_df,
                                                          test_size=0.7,
                                                          random_state=42,
                                                          shuffle=True,
                                                          stratify=train_df['target'])

X_train = train_data.drop(['customer_ID', 'S_2', 'target'], axis=1)
y_train = train_data['target']

X_test = valid_data.drop(['customer_ID', 'S_2', 'target'], axis=1)
y_test = valid_data['target']

#X_real_test = test_df.drop(['customer_ID', 'S_2'], axis=1)

