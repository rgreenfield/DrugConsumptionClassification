import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import express
from arrow import now
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import os 
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

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


time_start = now()
#umap = UMAP(random_state=2024, verbose=True, n_jobs=1, low_memory=False, n_epochs=1000)
#df[['x', 'y']] = umap.fit_transform(X=df[float_columns])
# Data processing Label Encoder on the categorical columns


accuracy_results = {}
f1_results = {}

# we need to stratify differently because our classes are differently imbalanced depending on the target
for target in targets:
    
    X_train, X_test, y_train, y_test = train_test_split(df[float_columns], df[target], test_size=0.20, random_state=2024)
    model = LogisticRegression(max_iter=100000, tol=1e-12).fit(X=X_train, y=y_train)
    f1_results[target] = f1_score(y_true=y_test, y_pred=model.predict(X=X_test), average='weighted')
    accuracy_results[target] = accuracy_score(y_true=y_test, y_pred=model.predict(X=X_test))

express.histogram(data_frame=pd.DataFrame.from_dict(data=f1_results, orient='index').reset_index(), x='index', y=0, title='f1').show()
express.histogram(data_frame=pd.DataFrame.from_dict(data=accuracy_results, orient='index').reset_index(), x='index', y=0, title='accuracy').show()
print('model done in {}'.format(now() - time_start))


f1_df = pd.DataFrame.from_dict(data=f1_results, orient='index').reset_index().rename(columns={0: 'f1'})
accuracy_df = pd.DataFrame.from_dict(data=accuracy_results, orient='index').reset_index().rename(columns={0: 'accuracy'})

express.scatter(data_frame=accuracy_df.merge(how='inner', on='index', right=f1_df), x='accuracy', y='f1', hover_name='index').show()



"""
Crack, herion are the only accurate results

Try xgboost
flaml?
    
"""

