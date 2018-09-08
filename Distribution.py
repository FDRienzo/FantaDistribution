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


def prepare_data():
    df = pd.read_excel("Data\FinManDBa.xlsx")
    df = df.dropna()
    # tiene solo portieri titolari
    portieri = df[df["R"] == "P"]
    df = df[df["R"] != "P"]
    portieri = portieri[portieri["Tit"] >= 0.75].reset_index(drop=True)
    return pd.concat([df, portieri]).reset_index(drop=True)


def make_graph(df):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    xs = df["SpesaPer"]
    ys = df["Tit"]
    zs = df["Quote"]
    ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors="w")

    ax.set_xlabel("SpesaPer")
    ax.set_ylabel("Tit")
    ax.set_zlabel("Quote")


def save_to_excel(squadre):
    campionato = pd.ExcelWriter("Teams\Campionato.xlsx", engine="xlsxwriter")
    for i in range(0, len(squadre)):
        squadre[i] = (
            squadre[i]
            .sort_values(by=["SpesaPer"], ascending=False)
            .reset_index(drop=True)
        )
        squadre[i] = squadre[i][squadre[i].index < 30]
        squadre[i].to_excel(campionato, sheet_name=f"Squadra{i}.xlsx")
        print(f"Squadra{i}.xlsx")
    campionato.save()


def make_teams(clusters, squadre):
    ruoli = ["A", "C", "D", "P"]
    for tier in clusters:
        shuffle(squadre)
        for ruolo in ruoli:
            try:
                part = tier.get_group(ruolo).reset_index(drop=True)
            except:
                continue
            while part.shape[0] > 0:
                for i in range(0, len(squadre)):
                    if part.shape[0] > 0:
                        player = part.loc[0]
                        squadre[i] = squadre[i].append(player)
                        part = part.drop(0).reset_index(drop=True)
                    else:
                        print("Tier Finito")


def distrib_gioc(n_squadre=8, n_tiers=4):
    df = prepare_data()

    # print(df)

    # print(df.columns)
    cols = [
        "Tit",
        "Quote",
        "Predict",
        "Predict STD",
        "FinVal",
        "Diff",
        "Tot. (%)",
        "SpesaPer",
        "SpesaM",
        "SpesaDiff",
    ]
    df[cols] = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())

    make_graph(df)

    cols = ["SpesaPer", "SpesaM", "Tit", "Predict"]

    kmeans = KMeans(n_clusters=n_tiers)
    kmeans.fit(df[cols])
    y_kmeans = kmeans.predict(df[cols])
    plt.scatter(df["SpesaPer"], df["SpesaM"], c=y_kmeans, s=50, cmap="viridis")

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
    # plt.show()
    df["clusters"] = y_kmeans
    """
    Primo = df[df["clusters"]==0].sample(frac=1).reset_index(drop=True)
    Primo = Primo.groupby("R")
    Secondo= df[df["clusters"]==1].sample(frac=1).reset_index(drop=True)
    Secondo = Secondo.groupby("R")
    
    Terzo= df[df["clusters"]==2].sample(frac=1).reset_index(drop=True)
    Terzo = Terzo.groupby("R")
    
    Quarto= df[df["clusters"]==3].sample(frac=1).reset_index(drop=True)
    Quarto = Quarto.groupby("R")
    
    Quinto= df[df["clusters"]==4].sample(frac=1).reset_index(drop=True)
    Quinto = Quinto.groupby("R")
    
    Sesto= df[df["clusters"]==5].sample(frac=1).reset_index(drop=True)
    Sesto = Sesto.groupby("R")
    """

    # Quinto= df[df["clusters"]==4].sample(frac=1).reset_index(drop=True)
    #
    """
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
    """

    # squadre = [Sq1,Sq2,Sq3,Sq4,Sq5,Sq6,Sq7,Sq8]
    # ,Sq9,Sq10,Sq11,Sq12]
    clusters = [
        df[df["clusters"] == i].sample(frac=1).reset_index(drop=True).groupby("R")
        for i in range(0, n_tiers)
    ]
    squadre = [pd.DataFrame() for i in range(0, n_squadre)]
    # squadre[1] = squadre[1].append(Clusters[0].iloc[1])
    # squadre[1] = squadre[1].append(Clusters[0].iloc[2])
    # print(squadre[1])
    # '''

    make_teams(clusters, squadre)

    save_to_excel(squadre)


if __name__ == "__main__":
    distrib_gioc(10, 6)
