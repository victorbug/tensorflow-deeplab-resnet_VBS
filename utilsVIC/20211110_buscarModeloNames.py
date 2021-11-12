import re
import pandas as pd

df=pd.read_csv("20211110_buscarModeloNames.py.txt", "\t", header=None)
df.columns=["layers"]
df["layers"]=df.apply(lambda x: re.search(r'[^\=]+$', x["layers"]).group(0), axis=1)
#df["layers"]=df.apply(lambda x: re.search(r"\'(.*?)\'", x["layers"]).group(0), axis=1)

#df["layers"]=df.apply(lambda x: re.search(r'(?<=-)\w+', x), axis=1)
df.to_csv("modeloNames.txt", index=False)
#print(df)
#m = re.search(r'(?<=-)\w+', 'spam-egg')