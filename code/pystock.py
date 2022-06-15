import csv
import pandas as pd

# https://github.com/eliangcs/pystock-data
price_file = r'prices.csv'
df = pd.read_csv(price_file,
                 names = ['symbol', 'date', 'open', 'high', 'low','close','volume','adj_close'],
                 usecols = ['symbol', 'date', 'close'],
                 low_memory=False) 
# read only 3 columns of interest

#! when using the "small" prices files need to remove odd or even lines
#! since they have *2* days per stock

df = df.drop(df.index[0]) # remove column labels
df['date'] = pd.to_datetime(df.date)
df = df.sort_values(['symbol', 'date'], ascending=[True, True])
df=df.pivot(index='symbol',columns='date',values='close')
#df = df.drop('date', axis=1) # remove date column
#dataframe now ready for processing by evobic
df=df.dropna(axis=0, thresh=10)
df=df.dropna(axis=1, thresh=10)
# for verification purposes
df.to_csv(r'output.csv')#,index = False,header=False)