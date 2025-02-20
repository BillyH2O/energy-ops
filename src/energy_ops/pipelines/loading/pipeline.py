from kedro.pipeline import Pipeline, node
from .nodes import load_csv_from_bucket


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_csv_from_bucket,
                inputs=["params:gcp_project_id", "params:gcs_primary_folder"],
                outputs="primary",  # on enregistre la DataFrame sous la clé "primary" dans le DataCatalog
                name="load_csv_node",
            )
        ]
    )
