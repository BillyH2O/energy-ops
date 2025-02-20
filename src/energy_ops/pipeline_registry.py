"""Project pipelines."""

'''
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
'''

# src/energy_ops/hooks.py

from typing import Dict
from kedro.framework.hooks import hook_impl
from kedro.pipeline import Pipeline

from energy_ops.pipelines.loading import pipeline as loading_pipeline
from energy_ops.pipelines.data_processing import pipeline as data_processing_pipeline
from energy_ops.pipelines.data_science import pipeline as data_science_pipeline


@hook_impl
def register_pipelines() -> Dict[str, Pipeline]:
    """
    Cette fonction déclare tous les pipelines existants dans le projet
    et les associe à un nom. Kedro l'utilise pour retrouver le pipeline
    lorsque vous lancez `kedro run --pipeline <nom>`.
    """
    # 1) On crée les objets Pipeline à partir de chaque module
    p_loading = loading_pipeline.create_pipeline()
    p_processing = data_processing_pipeline.create_pipeline()
    p_science = data_science_pipeline.create_pipeline()

    # 2) On retourne un dictionnaire {nom: pipeline}
    return {
        "loading": p_loading,
        "data_processing": p_processing,
        "data_science": p_science,
        # On peut aussi créer un pipeline "global" qui les enchaîne
        "global": Pipeline([p_loading, p_processing, p_science]),
    }
