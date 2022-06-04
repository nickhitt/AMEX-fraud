#exec(open("load_data.py").read())

### Distribution of target vars

fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x=train_labels.target)
plt.show()

train_labels['target'].value_counts() #counts

### Merging training labels with training df

train_df = train_df.merge(train_labels, on='customer_ID', how='left')

# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * mis_val / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

train_missing_vals = missing_values_table(train_df)
train_missing_vals.head(10) ## checking

# Checking data types in df
train_df.dtypes.value_counts()

### Encoding for categorical variables
var_cat = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

def one_hot(df, subset):
    x = pd.get_dummies(df[subset])
    names = []
    for i in range(1,len(x.columns)+1):
        names.append(subset + '_' + str(i))
    x.columns = names
    return x

one_hot_count = 0

# Encoding by iterating through the columns
for col in train_df.columns:
    if col in var_cat:
        # Encode the training and testing data
        temp_train = one_hot(train_df, col)
        temp_test = one_hot(test_df, col)
        # Drop column to encode
        train_df.drop(col, axis=1)
        test_df.drop(col, axis=1)
        # Bind encoded data to df
        train_df = pd.concat([train_df, temp_train], axis=1)
        test_df = pd.concat([test_df, temp_test], axis=1)
        # Keep track of how many columns were one hot encoded
        one_hot_count += 1
        # Cleaning temp vars
        temp_train = None
        temp_test = None

train_labels = train_df['target']

# Align the training and testing data, keep only columns present in both dataframes
train_df, test_df = train_df.align(test_df, join = 'inner', axis = 1)

# Add the target back in
train_df['target'] = train_labels

# Checking shapes
print('Training Features shape: ', train_df.shape)
print('Testing Features shape: ', test_df.shape)

#Correlations
# Find correlations with the target and sort
correlations = train_df.corr()['target'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))