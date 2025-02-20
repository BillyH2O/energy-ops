import numpy as np
import mlflow
import mlflow.keras

# import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


#############################################################################
# 1) Modèles candidats : RNN, LSTM
#############################################################################
MODELS = [
    {
        "name": "RNN",
        "build_fn": "train_rnn_model",  # ou un identifiant; on utilisera la fonction correspondante
        "params": {
            "units_1": hp.quniform("rnn_units_1", 16, 64, 16),
            "units_2": hp.quniform("rnn_units_2", 16, 64, 16),
            "dropout": hp.uniform("rnn_dropout", 0.0, 0.4),
            "learning_rate": hp.loguniform("rnn_lr", np.log(1e-4), np.log(1e-2)),
            "epochs": hp.quniform("rnn_epochs", 5, 15, 1),
            "batch_size": hp.choice("rnn_batch_size", [16, 32]),
        },
    },
    {
        "name": "LSTM",
        "build_fn": "train_lstm_model",
        "params": {
            "units_1": hp.quniform("lstm_units_1", 16, 64, 16),
            "units_2": hp.quniform("lstm_units_2", 16, 64, 16),
            "dropout": hp.uniform("lstm_dropout", 0.0, 0.4),
            "learning_rate": hp.loguniform("lstm_lr", np.log(1e-4), np.log(1e-2)),
            "epochs": hp.quniform("lstm_epochs", 5, 15, 1),
            "batch_size": hp.choice("lstm_batch_size", [16, 32]),
        },
    },
]

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("http://34.38.243.113")
mlflow.set_experiment("energy_forecast")


#############################################################################
# 2) Fonctions "training" & "evaluation" (extraits de votre code)
#############################################################################
def train_rnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rnn_epochs: int,
    rnn_batch_size: int,
    units_1: int = 50,
    units_2: int = 50,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
):
    """
    Construit et entraîne un Simple RNN (2 couches) puis renvoie (modèle, history).
    Paramètres "units_1", "units_2", "dropout", etc. sont ajoutés pour Hyperopt.
    """
    seq_length = X_train.shape[1]
    nb_features = X_train.shape[2]

    with mlflow.start_run(run_name="train_rnn_model"):
        # 1) Log des hyperparamètres
        mlflow.log_param("rnn_epochs", rnn_epochs)
        mlflow.log_param("rnn_batch_size", rnn_batch_size)
        mlflow.log_param("rnn_units_1", units_1)
        mlflow.log_param("rnn_units_2", units_2)
        mlflow.log_param("rnn_dropout", dropout)
        mlflow.log_param("rnn_lr", learning_rate)

        rnn_model = Sequential(
            [
                SimpleRNN(
                    units_1,
                    return_sequences=True,
                    input_shape=(seq_length, nb_features),
                ),
                Dropout(dropout),
                SimpleRNN(units_2),
                Dropout(dropout),
                Dense(1),
            ]
        )
        opt = Adam(learning_rate=learning_rate)
        rnn_model.compile(optimizer=opt, loss="mean_squared_error")

        history = rnn_model.fit(
            X_train,
            y_train,
            epochs=rnn_epochs,
            batch_size=rnn_batch_size,
            validation_split=0.2,
            verbose=1,
        )

        val_loss_final = history.history["val_loss"][-1]
        mlflow.log_metric("val_loss", val_loss_final)
        mlflow.keras.log_model(rnn_model, artifact_path="rnn_model_artifact")

        rnn_model.save("data/06_models/rnn_basique.h5")

    return rnn_model, history


def evaluate_rnn_model(rnn_model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Fait la prédiction du RNN, calcule R², RMSE, etc. et affiche un plot.
    """

    with mlflow.start_run(run_name="evaluate_rnn_model"):

        rnn_predictions = rnn_model.predict(X_test)

        r2 = r2_score(y_test, rnn_predictions)
        mae = mean_absolute_error(y_test, rnn_predictions)
        mse = mean_squared_error(y_test, rnn_predictions)
        rmse = np.sqrt(mse)

        print("=== Simple RNN ===")
        print(f"R² Score: {r2}")
        print(f"MAE       : {mae}")
        print(f"MSE       : {mse}")
        print(f"RMSE      : {rmse}")

        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(y_test[:200], label="Actual", color="blue")
        plt.plot(
            rnn_predictions[:200], label="Predicted", color="red", linestyle="dashed"
        )
        plt.title("Simple RNN - Actual vs Predicted Consumption")
        plt.xlabel("Time")
        plt.ylabel("Consumption")
        plt.legend()
        plt.grid(True)
        fig_path = "data/08_reporting/figures/rnn_actual_vs_pred.png"
        plt.savefig(fig_path)
        # plt.show()
        mlflow.log_artifact(fig_path, artifact_path="figures")


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lstm_epochs: int,
    lstm_batch_size: int,
    units_1: int = 50,
    units_2: int = 50,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
):
    """
    Construit et entraîne un LSTM (avec callbacks) puis renvoie (modèle, history).
    """
    seq_length = X_train.shape[1]
    nb_features = X_train.shape[2]

    with mlflow.start_run(run_name="train_lstm_model"):
        mlflow.log_param("lstm_epochs", lstm_epochs)
        mlflow.log_param("lstm_batch_size", lstm_batch_size)
        mlflow.log_param("lstm_units_1", units_1)
        mlflow.log_param("lstm_units_2", units_2)
        mlflow.log_param("lstm_dropout", dropout)
        mlflow.log_param("lstm_lr", learning_rate)

        model = Sequential(
            [
                LSTM(
                    units_1,
                    return_sequences=True,
                    input_shape=(seq_length, nb_features),
                ),
                Dropout(dropout),
                LSTM(units_2),
                Dropout(dropout),
                Dense(1),
            ]
        )
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss="mean_squared_error")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=lstm_epochs,
            batch_size=lstm_batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )

        val_loss_final = history.history["val_loss"][-1]
        mlflow.log_metric("val_loss", val_loss_final)
        mlflow.keras.log_model(model, artifact_path="lstm_model_artifact")

        model.save("data/06_models/lstm_basique.h5")

    return model, history


def evaluate_lstm_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Fait la prédiction du LSTM, calcule R², RMSE, etc. et affiche un plot.
    """

    with mlflow.start_run(run_name="evaluate_lstm_model"):
        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        print("=== LSTM ===")
        print(f"R² Score: {r2}")
        print(f"MAE       : {mae}")
        print(f"MSE       : {mse}")
        print(f"RMSE      : {rmse}")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(y_test[:200], label="Actual Consumption", color="blue")
        plt.plot(
            predictions[:200],
            label="Predicted Consumption",
            color="orange",
            linestyle="--",
        )
        plt.title("LSTM - Actual vs Predicted Electricity Consumption")
        plt.xlabel("Time")
        plt.ylabel("Consumption")
        plt.legend()
        plt.grid(True)
        fig_path = "data/08_reporting/figures/lstm_actual_vs_pred.png"
        plt.savefig(fig_path)
        # plt.show()
        mlflow.log_artifact(fig_path, artifact_path="figures")


#############################################################################
# 3) run_hyperopt : optimisation bayésienne + petite modif pour être plus rapide
#############################################################################
def run_hyperopt(
    X_train: np.ndarray, y_train: np.ndarray, model_name: str, max_evals: int = 5
):
    """
    Lance l'optimisation bayésienne hyperopt pour un des modèles candidats (RNN ou LSTM).
    max_evals = 5 (par défaut) pour être plus rapide.
    Retourne le dictionnaire des meilleurs hyperparamètres.
    """
    # On récupère la config du modèle dans MODELS
    candidate = [m for m in MODELS if m["name"] == model_name][0]
    search_space = candidate["params"]

    # Selon le nom, on appelle la fonction d'entraînement correspondante
    # (pour la partie objective)
    def objective(params):
        # On extrait et convertit tout ce qu'il faut
        epochs = int(params["epochs"])
        batch_size = int(params["batch_size"])
        dropout = float(params["dropout"])
        units_1 = int(params["units_1"])
        units_2 = int(params["units_2"])
        lr = float(params["learning_rate"])

        # Construction du modèle "en dur" selon le modèle
        # On va entraîner un "mini modèle" pour la val_loss
        if model_name == "RNN":
            model, hist = train_rnn_model(
                X_train,
                y_train,
                rnn_epochs=epochs,
                rnn_batch_size=batch_size,
                units_1=units_1,
                units_2=units_2,
                dropout=dropout,
                learning_rate=lr,
            )
        else:  # LSTM
            model, hist = train_lstm_model(
                X_train,
                y_train,
                lstm_epochs=epochs,
                lstm_batch_size=batch_size,
                units_1=units_1,
                units_2=units_2,
                dropout=dropout,
                learning_rate=lr,
            )
        val_loss = hist.history["val_loss"][-1]
        return {"loss": val_loss, "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42),
    )
    return best


#############################################################################
# 4) auto_ml : pour prendre les meilleurs hyperparamètres,
#    ré-entraîner et sauvegarder les modèles candidats
#############################################################################
def auto_ml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_evals: int = 1,
):
    """
    1) Pour chaque modèle dans MODELS:
       - on trouve les meilleurs hyperparamètres via run_hyperopt,
       - on ré-entraîne sur tout X_train,y_train,
       - on évalue sur X_test,y_test,
       - on sauvegarde ou stocke le modèle.
    2) Retourne un dict avec les infos (modèles, scores...).
    """
    results = []
    for candidate in MODELS:
        model_name = candidate["name"]
        print(f"\n=== Hyperopt pour {model_name} ===")
        best_hyp = run_hyperopt(X_train, y_train, model_name, max_evals)
        print("Best hyperparams:", best_hyp)

        # Convertir properly
        prefix = "rnn_" if model_name == "RNN" else "lstm_"
        best_params = {
            "units_1": int(best_hyp[f"{prefix}units_1"]),
            "units_2": int(best_hyp[f"{prefix}units_2"]),
            "dropout": float(best_hyp[f"{prefix}dropout"]),
            "learning_rate": float(best_hyp[f"{prefix}lr"]),
            "epochs": int(best_hyp[f"{prefix}epochs"]),
            "batch_size": int(best_hyp[f"{prefix}batch_size"]),
        }

        # Ré-entraîner le modèle sur TOUT X_train
        if model_name == "RNN":
            final_model, _ = train_rnn_model(
                X_train,
                y_train,
                rnn_epochs=best_params["epochs"],
                rnn_batch_size=best_params["batch_size"],
                units_1=best_params["units_1"],
                units_2=best_params["units_2"],
                dropout=best_params["dropout"],
                learning_rate=best_params["learning_rate"],
            )
        else:  # LSTM
            final_model, _ = train_lstm_model(
                X_train,
                y_train,
                lstm_epochs=best_params["epochs"],
                lstm_batch_size=best_params["batch_size"],
                units_1=best_params["units_1"],
                units_2=best_params["units_2"],
                dropout=best_params["dropout"],
                learning_rate=best_params["learning_rate"],
            )

        # Évaluation
        preds = final_model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        final_model.save(f"data/06_models/{model_name}_optimized.h5")

        results.append(
            {
                "model_name": model_name,
                "model_obj": final_model,
                "best_params": best_params,
                "scores": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
            }
        )

        # Sauvegarde du modèle sur disque (exemple Keras)
        # final_model.save(f"data/06_models/{model_name}_best.h5")
        # ... ou pickle, ou autre

    # Comparer RNN vs LSTM
    best_overall = max(results, key=lambda d: d["scores"]["r2"])
    print("\n=== Résultats finaux ===")
    for r in results:
        print(
            f"{r['model_name']} => R2={r['scores']['r2']:.4f}, RMSE={r['scores']['rmse']:.4f}"
        )

    print(
        f"\nLe meilleur modèle est: {best_overall['model_name']} (R2={best_overall['scores']['r2']:.4f})"
    )

    return {"all_results": results, "best_model": best_overall}
