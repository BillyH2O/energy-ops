from kedro.runner import SequentialRunner

from energy_ops.pipelines.loading.pipeline import create_pipeline


def test_pipeline(catalog_test):
    runner = SequentialRunner()
    pipeline = create_pipeline()
    pipeline_output = runner.run(pipeline, catalog_test)
    print(pipeline_output)
