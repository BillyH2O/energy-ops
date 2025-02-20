# src/energy_ops/pipelines/loading/nodes.py

import pandas as pd
import glob
from google.cloud import storage


def load_csv_from_bucket(project: str, bucket_path: str) -> pd.DataFrame:
    """
    Loads multiple CSV files from a GCS bucket folder
    (ex: my-bucket/some_folder/part-0000.csv).
    """
    # 1) On crée un client GCP
    storage_client = storage.Client(project=project)

    # 2) On sépare le bucket et le dossier
    bucket_name = bucket_path.split("/")[0]
    folder = "/".join(bucket_path.split("/")[1:]) + "/part-"

    # 3) On télécharge en local dans /tmp
    bucket_obj = storage_client.bucket(bucket_name)

    for blob in bucket_obj.list_blobs(prefix=folder):
        filename = blob.name.split("/")[-1]
        if filename.endswith(".csv"):
            blob.download_to_filename("/tmp/" + filename)

    # 4) On lit tous les CSV locaux
    all_files = glob.glob("/tmp/*.csv")
    df_list = []
    for filename in all_files:
        df_temp = pd.read_csv(filename, sep=",")
        df_list.append(df_temp)

    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df
