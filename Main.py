#---------IMPORT----------------------------------
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from Extraction import * 
from loads import *
from Transforms import * 
#-------------------IMPORT DATA---------------------------------
df = extract_csv('loans.csv')
Show_data(df.columns)
# Show_data(df.head(30))
# Show_data(df.info()) ##dont have null DATA
# Show_data(df.describe())
# Show_data(df['loan_type'].duplicated())
Show_data(df['loan_type'].unique()) #4 unique
data = LabelEncoding(df,['loan_type','client_id'])

#-------------------DATA ANALYSIS-------------------------------
#-----> Outlier DATA <-----
#use matplot

# plt.boxplot(df['rate'])   
# plt.show()
#---------------------------------
#use plotly
# Outlier_data(df,['rate'])
#---------------------------------
#remove outlier data
df = remove_rate_outliers(df,0,10)
#---------------------------------
#check again Outlier_data with ploty
# Outlier_data(df,['rate'])
#---------------------------------
#drop loan_start & loan_end
drop_columns(df,['loan_start','loan_end'])
#---------------------------------
#normalaize with min_max_scaler
df = min_max_scaler(df ,['client_id' , 'loan_type' , 'loan_amount' , 'repaid' , 'loan_id' , 'rate'] )
Show_data(df.head(30))
#---------------------------------
#load Pre-processed_data file
load(df,'Pre-processed_data.csv')