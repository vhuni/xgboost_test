import xgboost as xgb
import seaborn as sns
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Check dataset
diamonds = sns.load_dataset("diamonds")
print(diamonds.head())
print( diamonds.shape)
print(diamonds.describe(exclude=np.number))

# Set y as prediction feature
X, y = diamonds.drop('cut',axis=1), diamonds[['cut']]

# Encoding category
y_encoded = OrdinalEncoder().fit_transform(y)
print(y_encoded)

# Extract features of text
category = X.select_dtypes(exclude=np.number).columns.tolist()

# Converts to pandas.Categorical
for col in category:
    X[col] = X[col].astype("category")

# Splitting the dataset (train and test data)
X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,random_state=1, stratify=y_encoded)

# Train a model using the scikit-learn API
xgb_classifier = xgb.XGBClassifier(n_estimators=100, objective='multi:softprob', tree_method='hist', eta=0.1, max_depth=3, enable_categorical=True)
xgb_classifier.fit(X_train, y_train)

print(xgb_classifier)

# Convert the model to a native API model
model = xgb_classifier.get_booster()  # Used only when using xgboost native API, else use xgb_classifier for sklearn API

# Compare model score using y_test 
predict = xgb_classifier.predict(X_test)

print(f"model accuracy score: {accuracy_score(y_test,predict)}")

# result accuracy: 0.739