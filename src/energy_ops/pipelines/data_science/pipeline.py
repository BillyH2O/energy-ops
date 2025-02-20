from kedro.pipeline import Pipeline, node
from .nodes import (
    train_rnn_model,
    evaluate_rnn_model,
    train_lstm_model,
    evaluate_lstm_model,
    auto_ml,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crée le pipeline d'entraînement pour le projet energy_forecast.
    Inclut l'entraînement simple de RNN et LSTM, leur évaluation, et l'optimisation via auto_ml.

    Returns:
        Pipeline: Pipeline Kedro avec des nodes pour chaque étape.
    """
    return Pipeline(
        [
            node(
                func=train_rnn_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:RNN_epochs",
                    "params:RNN_batch_size",
                    "params:RNN_units_1",
                    "params:RNN_units_2",
                    "params:RNN_dropout",
                    "params:RNN_learning_rate",
                ],
                outputs=["rnn_model", "rnn_history"],
                name="train_rnn_node",
            ),
            node(
                func=evaluate_rnn_model,
                inputs=["rnn_model", "X_test", "y_test"],
                outputs="rnn_metrics",
                name="evaluate_rnn_node",
            ),
            node(
                func=train_lstm_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:LSTM_epochs",
                    "params:LSTM_batch_size",
                    "params:LSTM_units_1",
                    "params:LSTM_units_2",
                    "params:LSTM_dropout",
                    "params:LSTM_learning_rate",
                ],
                outputs=["lstm_model", "lstm_history"],
                name="train_lstm_node",
            ),
            node(
                func=evaluate_lstm_model,
                inputs=["lstm_model", "X_test", "y_test"],
                outputs="lstm_metrics",
                name="evaluate_lstm_node",
            ),
            node(
                func=auto_ml,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "scaler",
                    "params:max_evals",
                    "params:mlflowEnabled",
                    "params:mlflowExperimentId",
                ],
                outputs=dict(model="optimized_model", mlflow_run_id="mlflow_run_id"),
                name="auto_ml_node",
            ),
        ]
    )
