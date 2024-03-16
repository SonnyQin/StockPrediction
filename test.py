import pandas as pd
from Preprocess  import *
data=[[1,2,3,4,5,6],[11,12,13,14,15,16]]
data=pd.DataFrame(data)
x=data.iloc[-1]
print(pd.DataFrame(x))