import pandas as pd
import numpy as np
import sklearn as skl
import scipy as sc
import statsmodels as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

### Data Load

train_df = pd.read_feather('/Users/nicholashitt/Dropbox/My Mac (Nicholas’s MacBook Pro)/Downloads/archive (2)/train.feather')
test_df = pd.read_feather('/Users/nicholashitt/Dropbox/My Mac (Nicholas’s MacBook Pro)/Downloads/archive (2)/test.feather')
train_labels = pd.read_csv('/Users/nicholashitt/Dropbox/My Mac (Nicholas’s MacBook Pro)/Downloads/train_labels.csv')

