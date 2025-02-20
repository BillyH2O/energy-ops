import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pandas.plotting import lag_plot, autocorrelation_plot


def run_eda(df: pd.DataFrame) -> None:
    """
    Fait vos plots exploratoires et les sauvegarde sur disque
    plutôt que de les afficher dans une fenêtre GUI.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Crée un sous-dossier "figures" si besoin
    os.makedirs("data/08_reporting/figures", exist_ok=True)

    # 1) Exemple histogram
    plt.figure(figsize=(14, 7))
    plt.hist(df["Nuclear"], bins=50, alpha=0.5, label="Nuclear", color="blue")
    plt.hist(df["Wind"], bins=50, alpha=0.5, label="Wind", color="green")
    plt.hist(
        df["Hydroelectric"], bins=50, alpha=0.5, label="Hydroelectric", color="orange"
    )
    plt.xlabel("Electricity (MW)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Electricity Generation by Source")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "data/08_reporting/figures/hist_generation.png"
    )  # On sauvegarde l'image
    plt.close()  # On ferme la figure pour libérer la mémoire

    # 2) Exemple correlation matrix
    numeric_df = df.select_dtypes(include=["number"])
    plt.figure(figsize=(12, 6))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="viridis", linewidths=0.5)
    plt.title("Correlation Matrix of Electricity Data")
    plt.savefig("data/08_reporting/figures/corr_matrix.png")
    plt.close()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit la colonne 'DateTime' en datetime + crée 'hour' et 'dayofweek'.
    """
    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["hour"] = df["DateTime"].dt.hour
    df["dayofweek"] = df["DateTime"].dt.dayofweek
    df.drop(columns=["DateTime"], inplace=True)
    return df


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fait un MinMaxScaler sur vos colonnes d’intérêt (Consumption, Production, etc.).
    """
    df = df.copy()
    columns_to_scale = [
        "Consumption",
        "Production",
        "Nuclear",
        "Wind",
        "Hydroelectric",
        "Oil and Gas",
        "Coal",
        "Solar",
        "Biomass",
        "hour",
        "dayofweek",
    ]
    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df


def create_sequences(df: pd.DataFrame, seq_length: int):
    """
    Construit X, y pour la prédiction de 'Consumption' (fenêtre=seq_length).
    """
    sequences = []
    labels = []
    for i in range(len(df) - seq_length):
        seq = df.iloc[i : i + seq_length].values
        label = df["Consumption"].iloc[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    X = np.array(sequences)
    y = np.array(labels)
    return X, y


import pandas as pd


def split_dataset0(X: np.ndarray, y: np.ndarray, test_ratio: float):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    # Convertir en DataFrame
    # NB: Si X est 3D, vous devez décider comment l'aplatir en 2D pour CSV
    shape_3d = X_train.shape  # ex: (nb_samples, seq_length, nb_features)
    # Applatir si vous voulez absolument un CSV
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train, columns=["target"])  # 1D -> 2D
    y_test_df = pd.DataFrame(y_test, columns=["target"])

    return X_train_df, X_test_df, y_train_df, y_test_df


def split_dataset(X: np.ndarray, y: np.ndarray, test_ratio: float):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    # On NE convertit pas en DataFrame, on garde le 3D
    return X_train, X_test, y_train, y_test
