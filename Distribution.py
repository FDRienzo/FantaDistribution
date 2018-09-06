import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from random import choice
from random import shuffle


df = pd.read_excel('FinManDBa.xlsx')
df = df.dropna()
portieri = df[df["R"]=="P"]
df = df[df["R"]!="P"]
portieri = portieri[portieri["Tit"]>=0.75].reset_index(drop=True)
df = pd.concat([df,portieri]).reset_index(drop=True)

print(df)

#print(df.columns)
list = ['Tit', 'Quote', 'Predict',
       'Predict STD', 'FinVal', 'Diff', 'Tot. (%)', 'SpesaPer', 'SpesaM',
       'SpesaDiff']
df[list]=(df[list]-df[list].min())/(df[list].max()-df[list].min())

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = df["SpesaPer"]
ys = df["Tit"]
zs = df["Quote"]
ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel('SpesaPer')
ax.set_ylabel('Tit')
ax.set_zlabel('Quote')

list = ["SpesaPer","SpesaM","Tit","Predict"]

kmeans = KMeans(n_clusters=6vc)
kmeans.fit(df[list])
y_kmeans = kmeans.predict(df[list])
plt.scatter(df["SpesaPer"], df["SpesaM"], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
#plt.show()
df["Clusters"] = y_kmeans

Primo = df[df["Clusters"]==0].sample(frac=1).reset_index(drop=True)
Primo = Primo.groupby("R")
Secondo= df[df["Clusters"]==1].sample(frac=1).reset_index(drop=True)
Secondo = Secondo.groupby("R")

Terzo= df[df["Clusters"]==2].sample(frac=1).reset_index(drop=True)
Terzo = Terzo.groupby("R")

Quarto= df[df["Clusters"]==3].sample(frac=1).reset_index(drop=True)
Quarto = Quarto.groupby("R")

Quinto= df[df["Clusters"]==4].sample(frac=1).reset_index(drop=True)
Quinto = Quinto.groupby("R")

Sesto= df[df["Clusters"]==5].sample(frac=1).reset_index(drop=True)
Sesto = Sesto.groupby("R")

#Quinto= df[df["Clusters"]==4].sample(frac=1).reset_index(drop=True)
Clusters = [Primo,Secondo,Terzo,Quarto,Quinto,Sesto]
#

Sq1=pd.DataFrame()
Sq2=pd.DataFrame()
Sq3=pd.DataFrame()
Sq4=pd.DataFrame()
Sq5=pd.DataFrame()
Sq6=pd.DataFrame()
Sq7=pd.DataFrame()
Sq8=pd.DataFrame()
Sq9=pd.DataFrame()
Sq10=pd.DataFrame()
Sq11=pd.DataFrame()
Sq12=pd.DataFrame()
squadre = [Sq1,Sq2,Sq3,Sq4,Sq5,Sq6,Sq7,Sq8]
     #,Sq9,Sq10,Sq11,Sq12]
#squadre[1] = squadre[1].append(Clusters[0].iloc[1])
#squadre[1] = squadre[1].append(Clusters[0].iloc[2])
#print(squadre[1])
a = 0
ruoli = ["A","C","D","P"]
for tier in Clusters:
    shuffle(squadre)
    for ruolo in ruoli:
        try:
            part = tier.get_group(ruolo).reset_index(drop=True)
        except:
            continue
        while part.shape[0]>0:
            for i in range(0,8):
                if part.shape[0] > 0:
                    player = part.loc[0]
                    squadre[i]=squadre[i].append(player)
                    part= part.drop(0).reset_index(drop=True)
                else:
                    print("Tier Finito")

squadre[0].to_excel("Squadra1.xlsx")
squadre[1].to_excel("Squadra2.xlsx")
squadre[2].to_excel("Squadra3.xlsx")
squadre[3].to_excel("Squadra4.xlsx")
squadre[4].to_excel("Squadra5.xlsx")
squadre[5].to_excel("Squadra6.xlsx")
squadre[6].to_excel("Squadra7.xlsx")
squadre[7].to_excel("Squadra8.xlsx")
'''
squadre[8].to_excel("Squadra9.xlsx")
squadre[9].to_excel("Squadra10.xlsx")
squadre[10].to_excel("Squadra11.xlsx")
squadre[11].to_excel("Squadra12.xlsx")
'''



