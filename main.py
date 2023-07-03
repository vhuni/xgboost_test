import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import xgboost as xgb


warnings.filterwarnings("ignore")

# Check dataset
diamonds = sns.load_dataset("diamonds")
print(diamonds.head())
print( diamonds.shape)
print(diamonds.describe(exclude=np.number))

# Dataset cleaning 

# Set y as prediction feature
X, y = diamonds.drop('price',axis=1), diamonds[['price']]

# Extract text fetures (categories)

cat = X.select_dtypes(exclude=np.number).columns.tolist() # Categories excluding columns with numbers
cat2 = X.columns.tolist()   # Categories including all columns except y(price)

print(cat)
print(cat2)

# Convert to pandas category
for col in cat:
    X[col] = X[col].astype('category')

print(X.dtypes)

# Splitting the dataset (train and test data)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)


# Define hyperparameters
params = {"objective":"reg:squarederror", "tree_method":"hist"}

# Set number of boosting iterations
n = 100

# Start training
model = xgb.train(params=params,
                  dtrain=dtrain_reg,
                  num_boost_round = n)

# Evaluate the model
prediction = model.predict(dtest_reg)
rmse = mean_squared_error(y_test, prediction, squared=False)

# Print the RMSE using sklearn mse
print(f"the result is: {rmse:.3f}")

# Print using xgboost with evals(tuple)
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

# Add verbose_eval to only print evals on defined rounds

model = xgb.train(params=params,
                  dtrain=dtrain_reg,
                  evals=evals,
                  num_boost_round=n,
                  verbose_eval=20 ) # Print every 20 rounds

# Addind early_stopping_rounds to stop xgb from overfitting (less training loss but high validation loss)
print("----------------------------------------------------")

model_pred = xgb.train(params=params,
                  dtrain=dtrain_reg,
                  num_boost_round=1000,
                  evals=evals,
                  verbose_eval=50,
                  early_stopping_rounds=50) # activate early stop after 50 rounds of no validation improvements


print("----------------------------------------------------")

# Adding cross-validation using cv and folds

results = xgb.cv(params=params,
                 dtrain=dtrain_reg,
                 num_boost_round=1000,
                 verbose_eval=30,
                 early_stopping_rounds=50,
                 nfold=5)

print(results.head())
best_rmse= results['test-rmse-mean'].min()
print(f"best rmse: {best_rmse:.3f}")

print("----------------------------------------------------")

# XGBoost classification (binary:logistic / multi:softprob)
from sklearn.preprocessing import OrdinalEncoder

X2, y2 = diamonds.drop("cut", axis=1), diamonds[["cut"]]

# Encode y to numeric
y_encoded = OrdinalEncoder().fit_transform(y2)

# Extract features of text
category = X2.select_dtypes(exclude=np.number).columns.tolist()

# Converts to pandas.Categorical
for col in category:
    X2[col] = X2[col].astype("category")

# Splitting the data
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y_encoded, random_state=1, stratify=y_encoded)

# Create classification matrices
dtrain_clf = xgb.DMatrix(X2_train, y2_train, enable_categorical=True)
dtest_clf = xgb.DMatrix(X2_test, y2_test, enable_categorical=True)

# Create parameters for classification
params2 = {"objective":"multi:softprob", "tree_method":"hist", "num_class":5}
n2 = 1000

results2 = xgb.cv(params=params2,
                 dtrain=dtrain_clf,
                 num_boost_round=n2,
                 nfold=5,
                 metrics=["mlogloss", "auc","merror"]
                 )

test_model = xgb.train(params=params2,
                 dtrain=dtrain_clf,
                 num_boost_round=100
                 )

pred = test_model.predict(dtest_clf)
print(f"classification accuracy: {accuracy_score(y_test,pred)}")

# prediction accuracy: 

# results2.keys()
# print("results: ", results2["test-auc-mean"].max())

# result accuracy: 0.9403