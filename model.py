import numpy as np
import pandas as pd
import pickle
import json
from io import StringIO
setattr(pd, "Int64Index", pd.Index)
setattr(pd, "Float64Index", pd.Index)
import matplotlib.pyplot as plt
from plotly import express
from arrow import now
from umap import UMAP
from flaml import AutoML
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flaml import AutoML, tune
import os 
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

def scaling_function(df):
    """
    This function is designed to scaled the specific dataframe df with the following
    list of variables. That way the only function that needs to be changed is this one.
    """


    # we drop the following variables including time
    drop_list = ["ID",]

    df = df.drop(drop_list, axis=1)

    # Splitting up the data in the dataframe
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # need to take out game_rom_min since the distribution is chi-squared
    nu = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore',
          'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',]
    sc = StandardScaler()
    X[nu] = sc.fit_transform(X[nu])

    return X, y

def auto_ml(df, targets):
    
    for target in targets:
        X_train, X_test, y_train, y_test = train_test_split(df[float_columns], df[target], test_size=0.20, random_state=2024)

        automl = AutoML()
        # Specify automl goal and constraints
        automl_settings = {
            "time_budget": 50,  # In seconds
            "metric": "accuracy",
            "task": "classification",
        }

        # fit classification on the training data
        clf = automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

        # Export the best model
        print(automl.model)


file_name = "data/drug_consumption.csv"
file_path = os.path.join(os.getcwd(), file_name)  
df = pd.read_csv(file_path)
#print(df.head())
#print(df.describe) #(1885, 32)
#print(df.info())

cols = [
       # Numeric columns 
       'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore',
       'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
       # Categoriocal Columns
       'Alcohol','Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 
       'Crack','Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
       'Nicotine', 'Semer', 'VSA']

float_columns =  ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore',
                  'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',]

# we do not have enough positives for Semer to be meaningful so we're going to have to leave it out
targets = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy',
           'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

label_encoder = LabelEncoder()
for cat in targets:
    df[cat] = label_encoder.fit_transform(df[cat])

# Store each model of the target variable into a dictionary
model_output = {}

#auto_ml(df, targets)
for target in targets:
    X_train, X_test, y_train, y_test = train_test_split(df[float_columns], df[target], test_size=0.20, random_state=2024)

    automl = AutoML()

    automl_settings = {
        "time_budget": 1,  # In seconds
        "metric": "accuracy",
        "task": "classification",
        "log_file_name": 'mylog.log',
    }

    # fit classification on the training data
    clf = automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

    # Export the best model
    retrained_model = automl.model
    print(automl.model)

    model_output[target] = retrained_model.estimator
    
    # Save the model to a file
    with open(f'./trained_models/{target}_model.pkl', 'wb') as f:
        pickle.dump(retrained_model.estimator, f)

    # Load the model from the file
    with open(f'./trained_models/{target}_model.pkl', 'rb') as f:
        retrained_model = pickle.load(f)

print(model_output)

time_start = now()

# Make sure that you have the same split as which the models were trained on
X_train, X_test, y_train, y_test = train_test_split(df[float_columns], df[target], test_size=0.20, random_state=2024)

accuracy_results = {}
f1_results = {}

for target in targets:
    model = model_output[target].fit(X=X_train, y=y_train)
    f1_results[target] = f1_score(y_true=y_test, y_pred=model.predict(X_test), average='weighted')
    accuracy_results[target] = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))
    fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
    roc_auc = auc(fpr, tpr)
    roc_curves[target] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
    
print(accuracy_results)
print(f1_results)

label = list(f1_results.keys())
f1_score_results = list(f1_results.values())

# Plot the values
plt.figure(figsize=(8, 6))
plt.bar(label, f1_score_results)
plt.xlabel('')
plt.ylabel('f1-score')
plt.title('F1-Scores for each Drug')
plt.show()


#umap = UMAP(random_state=2024, verbose=True, n_jobs=1, low_memory=False, n_epochs=1000)
#df[['x', 'y']] = umap.fit_transform(X=df[float_columns])
# Data processing Label Encoder on the categorical columns

#label_encoder = LabelEncoder()
#for cat in targets:
#    df[cat] = label_encoder.fit_transform(df[cat])


"""
X_train, X_test, y_train, y_test = train_test_split(df[float_columns], df['Alcohol'], test_size=0.20, random_state=2024, stratify=df['Alcohol'])
model = xgb.XGBClassifier().fit(X=X_train, y=y_train)
f1_results = f1_score(y_true=y_test, y_pred=model.predict(X=X_test), average='weighted')
accuracy_results = accuracy_score(y_true=y_test, y_pred=model.predict(X=X_test))

"""
"""
accuracy_results = {}
f1_results = {}

for target in targets:
    
    X_train, X_test, y_train, y_test = train_test_split(df[float_columns], df[target], test_size=0.20, random_state=2024)
    model = xgb.XGBClassifier().fit(X=X_train, y=y_train)
    f1_results[target] = f1_score(y_true=y_test, y_pred=model.predict(X=X_test), average='weighted')
    accuracy_results[target] = accuracy_score(y_true=y_test, y_pred=model.predict(X=X_test))

express.histogram(data_frame=pd.DataFrame.from_dict(data=f1_results, orient='index').reset_index(), x='index', y=0, title='f1').show()
express.histogram(data_frame=pd.DataFrame.from_dict(data=accuracy_results, orient='index').reset_index(), x='index', y=0, title='accuracy').show()
print('model done in {}'.format(now() - time_start))


f1_df = pd.DataFrame.from_dict(data=f1_results, orient='index').reset_index().rename(columns={0: 'f1'})
accuracy_df = pd.DataFrame.from_dict(data=accuracy_results, orient='index').reset_index().rename(columns={0: 'accuracy'})

express.scatter(data_frame=accuracy_df.merge(how='inner', on='index', right=f1_df), x='accuracy', y='f1', hover_name='index').show()
"""


"""
Crack, herion are the only accurate results

Try xgboost
flaml?
    
"""

