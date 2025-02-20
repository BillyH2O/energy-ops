from kedro.pipeline import Pipeline, node
from .nodes import (
    train_rnn_model,
    evaluate_rnn_model,
    train_lstm_model,
    evaluate_lstm_model,
    run_hyperopt,
    auto_ml,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_rnn_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:RNN_EPOCHS",
                    "params:RNN_BATCH_SIZE",
                ],
                outputs=["rnn_model", "rnn_history"],
                name="train_rnn_node",
            ),
            node(
                func=evaluate_rnn_model,
                inputs=["rnn_model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_rnn_node",
            ),
            node(
                func=train_lstm_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:LSTM_EPOCHS",
                    "params:LSTM_BATCH_SIZE",
                ],
                outputs=["lstm_model", "lstm_history"],
                name="train_lstm_node",
            ),
            node(
                func=evaluate_lstm_model,
                inputs=["lstm_model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_lstm_node",
            ),
            node(
                func=auto_ml,
                inputs=["X_train", "y_train", "X_test", "y_test", "params:MAX_EVALS"],
                outputs="auto_ml_results",
                name="auto_ml_node",
            ),
        ]
    )
