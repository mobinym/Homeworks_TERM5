import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import pandas as pd

#print data
def Show_data(data):
    print(data)
    print(100*'-')
#---------------------------------
#label encoding
def LabelEncoding(data,columns):
    le = LabelEncoder()
    for col in columns:
        data[col] = le.fit_transform(data[col]) 
    return data

#--------------------------------------
#outlier data 
def Outlier_data(data,columns):
    fig = px.box(data,y=columns)
    fig.show()
#--------------------------------------
#remove outlier data 
def remove_rate_outliers(data,min_rate,max_rate):
    df = pd.DataFrame(data)
    data = df[(df['rate']>=min_rate) & (df['rate']<=max_rate)]
    return data
#--------------------------------------
#drop loan_start & loan_end
def drop_columns(data, columns):
    for col in columns:
        data.drop(col, axis=1,inplace=True)
    return data
#--------------------------------------
#normalaize
def min_max_scaler(data,columns):
    scaler = MinMaxScaler()
    data= scaler.fit_transform(data)
    data = pd.DataFrame(data)
    data.columns = columns
    return data
#--------------------------------------