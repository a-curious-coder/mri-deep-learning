import boto3
import os
import pandas as pd

from dotenv import load_dotenv
from sklearn.decomposition import PCA

def normalize_tabular_data(mri_data):
    """Normalizes the values of each column (Excluding columns unrelated to the MRI scan itself)

    Args:
        mri_data (pd.DataFrame): MRI Data

    Returns:
        pd.DataFrame: Normalized MRI data
    """
    avoid = ["Study", "SID", "total CNR", "Gender", "Research Group", "Age"]
    # apply normalization techniques
    for column in mri_data.columns:
        if column not in avoid:
            mri_data[column] = mri_data[column]  / mri_data[column].abs().max()
    return mri_data


def clean_data(data):
    """Cleans MRI data, ready for deep learning model

    Args:
        data (pd.DataFrame): MRI data in tabular format
    """

    pass


def get_tabular_data(client):
    """Gets tabular MRI data from AWS

    Args:
        client (): AWS client

    Returns:
        pd.DataFrame: Tabular MRI data
    """
    response = client.get_object(Bucket='mri-deep-learning',
                                 Key='data/tabular/adni_ixi_rois_data_raw.csv')
    return pd.read_csv(response.get("Body"))


def main():
    # Loads access keys in from .env file
    load_dotenv()   
    access_key = os.getenv("ACCESS_KEY")
    secret_access_key = os.getenv("SECRET_ACCESS_KEY")

    client = boto3.client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_access_key)

    print("-------------------\nLoad Data\n -------------------")
    mri_data = get_tabular_data(client)
    print("-------------------\nNormalize Data\n -------------------")
    normalized_mri_data = normalize_tabular_data(mri_data)
    # NaN values occur because there are a bunch of columns filled with zeros, thus can't be normalized
    normalized_mri_data = normalized_mri_data.fillna(0)
    print("-------------------\nFiltered Data\n -------------------")
    # Filter data to remove last 6 columns (irrelevant)
    filtered_data = normalized_mri_data.iloc[:, :normalized_mri_data.shape[1]-6]
    # Checks if any values are NaN
    nan_values = filtered_data.isna()
    nan_columns = nan_values.any()
    columns_with_nan = filtered_data.columns[nan_columns].tolist()
    print(columns_with_nan)
    print("-------------------\nPCA\n -------------------")
    pca = PCA(n_components=2)
    pca.fit(filtered_data)
    print(pca.components_)


if __name__ == "__main__":
    main()