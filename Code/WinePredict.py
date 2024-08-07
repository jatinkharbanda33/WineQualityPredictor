#Wine Prediction

#Import Libraries & Load Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib
import yaml

#path config
path_config = '../Config/config.yaml'

#safeload config
config = yaml.safe_load(open(path_config))

filename = config['data_source']['directory'] + config['data_source']['filename']

#load dataset
wine = pd.read_csv(filename)

df = wine.copy()

df.head()

#Initial Imformation About Dataset
#Basic Information

print(df.columns)
print(df.shape)

df.info()
#Dataset consists of 1599 rows and 12 columns. Data type of all variable are float.

df.describe()
#We can see the average of each column of the dataset. 

#Checking for Null or Missing values
df.isnull().sum()
#It looks like there are no missing value. It means dataset can be processed.

df.rename(columns = {"fixed acidity": "fixed_acidity", "volatile acidity": "volatile_acidity",
                    "citric acid": "citric_acid", "residual sugar": "residual_sugar",
                    "chlorides": "chlorides", "free sulfur dioxide": "free_sulfur_dioxide",
                    "total sulfur dioxide": "total_sulfur_dioxide"}, inplace = True)

#Data Visualzation
#Distribution variable target quality

sns.countplot(data=df, x='quality')
plt.title('Distribution Variable Target')
plt.show()


#Visualize the correlation contents in red wine

plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True)
plt.title("Heatmap for Red Wine Quality")
plt.show()


#Exploratory Data Analysis

# - Feature Scaling

#1. Scale the dataset by quality

#After reading the Red Wine Quality dataset description we find,
#- quality >= 7 is "good"
#- quality <= 7 is "bad"

df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

sns.countplot(data = df, x = 'quality')
plt.xticks([0,1], ['bad wine','good wine'])
plt.title("Types of Wine")
plt.show()
#from the above visualization we can see that the Dataset is skewed or unbalanced.

#2. Resampling Dataset
#for skewed or unbalanced dataset, we can do over sampling using random over sampler for data balancing

# Sampling Dataset
X = df[['fixed_acidity', 'volatile_acidity', 'sulphates', 'alcohol', 'density']]
y = df.quality

random_value = config['data_source']['random_state']
#ran = RandomOverSampler(random_state=random_value)
#X_ros, y_ros = ran.fit_resample(X, y)
oversample = SMOTE()
X_ros, y_ros = oversample.fit_resample(X, y)


sns.countplot(x=y_ros)
plt.xticks([0,1], ['bad wine','good wine'])
plt.title("Types of Wine")
plt.show()


#3. Preprocessing Dataset

#Split dataset into train and test 
# use test size of 20% of the data proportion
X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size=config['data_source']['test_size'], random_state=random_value)
X_train.shape, X_test.shape


# Scale dataset with StandardScaler
scaler = StandardScaler()

# fit to data training
scaler.fit(X_train)

# transform
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)


#Training Models

# Random Forest
rfc = RandomForestClassifier(n_estimators=config['model_2']['n_estimator'], random_state=random_value)

# Cross Validation
rf_score = cross_val_score(estimator=rfc, X=x_train, y=y_train, scoring=config['model_2']['scoring'], cv=config['model_2']['cv'], verbose=config['model_2']['verbose'], n_jobs=config['model_2']['n_jobs'])

# Fit data training
rfc.fit(x_train, y_train)

# Predict data test
y_pred = rfc.predict(x_test)

print('Average Recall score:', np.mean(rf_score))
print('Test Recall score:', recall_score(y_test, y_pred))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Heatmap Confusion Matrix
sns.heatmap(conf_mat, cmap='Reds', annot=True, fmt='.1f')
plt.title('Confusion Matrix dari Prediksi Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Hyperparameter Tuning
rf_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Use GridSearchCV
rf_cv = GridSearchCV(estimator=rfc, param_grid=rf_grid, scoring=config['model_2']['scoring'], cv=config['model_2']['cv'])

# Fit to model
rf_cv.fit(x_train, y_train)

# Best Score
print('Best score:', rf_cv.best_score_)
print('Best params:', rf_cv.best_params_)

# Compare Score
rf_tuned = RandomForestClassifier(**rf_cv.best_params_, random_state=random_value)

# Cross Validation
rf_tuned_score = cross_val_score(estimator=rf_tuned, X=x_train, y=y_train, scoring=config['model_2']['scoring'], cv=config['model_2']['cv'], verbose=config['model_2']['verbose'])

# Fit data training
rf_tuned.fit(x_train, y_train)

# Predict data test
y_pred_tuned = rf_tuned.predict(x_test)

# Check Score
print('Average Recall score:', np.mean(rf_score))
print('Test Recall score:', recall_score(y_test, y_pred))
print('Average Recall score Tuning:', np.mean(rf_tuned_score))
print('Test Recall score Tuning:', recall_score(y_test, y_pred_tuned))

# Save Random Forest model
model_name = 'rf_model.pkl'
model_path = '../src/Model/{}'.format(model_name)
# Save the trained model
with open(model_path, 'wb') as f:
    pickle.dump(rf_tuned, f)
print('Model saved as {}'.format(model_name))
