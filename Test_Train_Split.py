import pandas as pd
from sklearn import model_selection
from random import sample

### Splitting into train/test sets


count_pos = train_df[train_df['target'] == 1]['target'].count().astype(int)
count_pos = count_pos.astype(int)
pos = train_df[train_df['target'] == 1]
neg = train_df[train_df['target'] == 0]
neg = neg.sample(count_pos)
train_df_balanced = pd.concat([pos,neg], axis=0).sample(frac=1)

train_data, valid_data = model_selection.train_test_split(train_df_balanced,
                                                          test_size=0.7,
                                                          random_state=42,
                                                          shuffle=True,
                                                          stratify=train_df_balanced['target'])

X_train = train_data.drop(['customer_ID', 'S_2', 'target'], axis=1)
y_train = train_data['target']

X_test = valid_data.drop(['customer_ID', 'S_2', 'target'], axis=1)
y_test = valid_data['target']

#X_real_test = test_df.drop(['customer_ID', 'S_2'], axis=1)

