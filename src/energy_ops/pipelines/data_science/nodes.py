import os
import numpy as np
import mlflow
import mlflow.keras
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Définition des modèles candidats (RNN et LSTM)
MODELS = [
    {
        "name": "RNN",
        "build_fn": "train_rnn_model",
        "params": {
            "units_1": hp.quniform("rnn_units_1", 16, 64, 16),
            "units_2": hp.quniform("rnn_units_2", 16, 64, 16),
            "dropout": hp.uniform("rnn_dropout", 0.0, 0.4),
            "learning_rate": hp.loguniform("rnn_lr", np.log(1e-4), np.log(1e-2)),
            "epochs": hp.quniform("rnn_epochs", 5, 15, 1),  # 5, 15, 1
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
            "epochs": hp.quniform("lstm_epochs", 5, 15, 1),  # 1, 3, 1
            "batch_size": hp.choice("lstm_batch_size", [16, 32]),
        },
    },
]

# Definition du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:  # Éviter les doublons
    logger.addHandler(handler)


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
    Sauvegarde le modèle sur disque.
    """
    seq_length = X_train.shape[1]
    nb_features = X_train.shape[2]

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

    rnn_model.save("data/06_models/rnn_basique.h5")

    return rnn_model, history


def evaluate_rnn_model(rnn_model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Fait la prédiction du RNN, calcule R², RMSE, etc.
    """
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

    return {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse}


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
    Sauvegarde le modèle sur disque.
    """
    seq_length = X_train.shape[1]
    nb_features = X_train.shape[2]

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

    model.save("data/06_models/lstm_basique.h5")

    return model, history


def evaluate_lstm_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Fait la prédiction du LSTM, calcule R², RMSE, etc.
    """
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

    return {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse}


def run_hyperopt(
    X_train: np.ndarray, y_train: np.ndarray, model_name: str, max_evals: int = 5
):
    """
    Lance l'optimisation bayésienne Hyperopt pour un des modèles candidats (RNN ou LSTM).
    Retourne le dictionnaire des meilleurs hyperparamètres.
    """
    candidate = [m for m in MODELS if m["name"] == model_name][0]
    search_space = candidate["params"]

    def objective(params):
        epochs = int(params["epochs"])
        batch_size = int(params["batch_size"])
        dropout = float(params["dropout"])
        units_1 = int(params["units_1"])
        units_2 = int(params["units_2"])
        lr = float(params["learning_rate"])

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


def save_prediction_plot(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, limit: int = 200
):
    """
    Génère et sauvegarde un graphique comparant les valeurs réelles et prédites dans data/08_reporting.
    """
    os.makedirs("data/08_reporting", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:limit], label="Actual", color="blue")
    plt.plot(y_pred[:limit], label="Predicted", color="red", linestyle="dashed")
    plt.title(f"{model_name} - Actual vs Predicted Energy Consumption")
    plt.xlabel("Time")
    plt.ylabel("Consumption")
    plt.legend()
    plt.grid(True)
    fig_path = f"data/08_reporting/{model_name}_actual_vs_pred.png"
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def auto_ml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    max_evals: int = 1,
    log_to_mlflow: bool = False,
    experiment_id: int = -1,
) -> dict:
    """
    Optimise les hyperparamètres avec Hyperopt, entraîne les modèles candidats (RNN, LSTM),
    évalue leurs performances, et logue les résultats dans MLflow sous un seul run.
    Sauvegarde les modèles sur disque et retourne un dictionnaire avec le meilleur modèle et son run_id.
    """
    logger.info("Starting auto_ml")
    run_id = ""
    results = []

    """# Réduire à 10% des données pour les tests
    subset_size = int(0.01 * len(X_train))
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    X_test_subset = X_test[: int(0.01 * len(X_test))]
    y_test_subset = y_test[: int(0.01 * len(X_test))]
    logger.info(
        f"Reduced X_train size: {X_train_subset.shape}, X_test size: {X_test_subset.shape}"
    )
    X_train, y_train, X_test, y_test = (
        X_train_subset,
        y_train_subset,
        X_test_subset,
        y_test_subset,
    )"""

    if log_to_mlflow:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))
        mlflow.set_experiment("energy_forecast")
        run = mlflow.start_run(experiment_id=experiment_id)
        run_id = run.info.run_id

    try:
        for candidate in MODELS:
            model_name = candidate["name"]
            print(f"\n=== Hyperopt pour {model_name} ===")
            best_hyp = run_hyperopt(X_train, y_train, model_name, max_evals)
            print("Best hyperparams:", best_hyp)

            prefix = "rnn_" if model_name == "RNN" else "lstm_"
            best_params = {
                "units_1": int(best_hyp[f"{prefix}units_1"]),
                "units_2": int(best_hyp[f"{prefix}units_2"]),
                "dropout": float(best_hyp[f"{prefix}dropout"]),
                "learning_rate": float(best_hyp[f"{prefix}lr"]),
                "epochs": int(best_hyp[f"{prefix}epochs"]),
                "batch_size": int(best_hyp[f"{prefix}batch_size"]),
            }

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

            final_model.save(f"data/06_models/{model_name}_optimized.h5")

            metrics = (
                evaluate_rnn_model(final_model, X_test, y_test)
                if model_name == "RNN"
                else evaluate_lstm_model(final_model, X_test, y_test)
            )
            preds = final_model.predict(X_test)

            if log_to_mlflow:
                fig_path = save_prediction_plot(y_test, preds, model_name)
                mlflow.log_artifact(fig_path, artifact_path="plots")

            results.append(
                {
                    "model_name": model_name,
                    "model_obj": final_model,
                    "best_params": best_params,
                    "scores": metrics,
                }
            )

        best_result = max(results, key=lambda x: x["scores"]["r2"])
        best_model = best_result["model_obj"]
        best_params = best_result["best_params"]
        best_metrics = best_result["scores"]

        if log_to_mlflow:
            mlflow.log_params(
                {f"{best_result['model_name']}_{k}": v for k, v in best_params.items()}
            )
            mlflow.log_metrics(best_metrics)
            """for result in results:
                mlflow.log_artifact(
                    f"data/08_reporting/{result['model_name']}_actual_vs_pred.png",
                    artifact_path="plots",
                )"""
            mlflow.log_artifact(
                "data/04_feature/scaler.pkl", artifact_path="transformations"
            )
            mlflow.keras.log_model(best_model, "model")

    finally:
        if log_to_mlflow:
            mlflow.end_run()

    return {"model": best_model, "mlflow_run_id": run_id}
