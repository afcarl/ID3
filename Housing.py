import pandas as pd
import numpy as np
import Quandl

api_key = 'seb_UHjiaf6bEKswahhz'

df = Quandl.get("FMAC/HPI_WA", authtoken=api_key)

print(df.head())


fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')

#print(fifty_states[0][0])

for abbv in fifty_states[0][0][1:,]:
    print("FMAC/HPI_"+abbv)






