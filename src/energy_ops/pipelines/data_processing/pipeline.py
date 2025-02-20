from kedro.pipeline import Pipeline, node
from .nodes import (
    run_eda,
    add_time_features,
    scale_data,
    create_sequences,
    split_dataset,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # 1) EDA
            node(
                func=run_eda, inputs="primary_data", outputs=None, name="run_eda_node"
            ),
            # 2) Ajout time features
            node(
                func=add_time_features,
                inputs="primary_data",
                outputs="df_with_time_features",
                name="add_time_features_node",
            ),
            # 3) Scale
            node(
                func=scale_data,
                inputs="df_with_time_features",
                outputs=dict(scaled_df="scaled_df", scaler="scaler"),
                name="scale_data_node",
            ),
            # 4) Create sequences
            node(
                func=create_sequences,
                inputs=["scaled_df", "params:SEQ_LENGTH"],
                outputs=["X_full", "y_full"],
                name="create_sequences_node",
            ),
            # 5) Split
            node(
                func=split_dataset,
                inputs=["X_full", "y_full", "params:TEST_RATIO"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_dataset_node",
            ),
        ]
    )
