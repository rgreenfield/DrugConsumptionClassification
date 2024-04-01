# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import express
from arrow import now
from umap import UMAP
import os 
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

file_name = "data\drug_consumption.csv"
file_path = os.path.join(os.getcwd(), file_name)  
df = pd.read_csv(file_path)
print(df.head())
#print(df.describe) #(1885, 32)
#print(df.info())
print(df.columns)


cols = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore',
       'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Alcohol',
       'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack',
       'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
       'Nicotine', 'Semer', 'VSA']

float_columns =  ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore',
                  'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',]

# we do not have enough positives for Semer to be meaningful so we're going to have to leave it out
targets = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'VSA']

time_start = now()


umap = UMAP(random_state=2024, verbose=True, n_jobs=1, low_memory=False, n_epochs=1000)
df[['x', 'y']] = umap.fit_transform(X=df[float_columns])
express.scatter(data_frame=df, x='x', y='y').show()
print('done with UMAP in {}'.format(now() - time_start))

# we have to take a sample for performance reasons; if we try to plot the whole dataset our plots crash
sample_df = df.sample(n=300, random_state=2024)
for target in targets:
    express.scatter(data_frame=sample_df, x='x', y='y', color=target).show()
    
    
    
    
    
    