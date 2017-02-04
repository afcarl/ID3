import pandas as pd
from pandas_datareader import data as web
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


style.use('ggplot')

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2015,1,1)

df = web.DataReader("XOM", "yahoo", start=start, end=end)
#print(df.head())
#df['Adj Close'].plot()
#plt.show()



# Pandas tutorial

web_stats = {
    'Day':[1,2,3,4,5,6],
    'Visitors':[23,34,56,1,45,6],
    'Bounce_rate':[12,3,4,56,7,8]
}

df = pd.DataFrame(web_stats)
#print(df)

# Change the index
df.set_index('Day', inplace=True)
#print(df)

#print(np.array(df[['Bounce_rate', 'Visitors']]))

#print(df.Visitors.tolist())


# To read a csv file

df = pd.read_csv('ZILL-Z77006_TT.csv', index_col=0)
print(df.head())

