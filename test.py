import pandas as pd
df=pd.DataFrame([11,12,13,14,15,16,17,18,19,20])
print(df)
df=df.shift(-1)
print(df)
