import boto3
import os
import pandas as pd
import plotly
from dotenv import load_dotenv
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def print_data_shape(data):
    print(f"This dataset has {data.shape[0]} rows and {data.shape[1]} columns")


def print_null_values(data):
    """ Prints the quantity of null/NaN values in columns containing null/NaN values

    Args:
        data (pd.DataFrame): Tabular MRI data
    """
    # Columns containing null/NaN values
    null_column_names = data.columns[data.isnull().any()].tolist()
    print(f"There are {len(null_column_names)} columns with null/NaN values")
    for column in null_column_names:
        print(f"{column}: {data[column].isna().sum()}")


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


def linear_regression(X, y):
    """Linear Regression model trained given mri data; prints accuracy results based on test sets

    Args:
        X ([type]): MRI dataset
        y ([type]): MRI dataset labels
    """
    print("-------------------\nLinear Regression\n-------------------")
    reg = linear_model.LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
    reg.fit(x_train, y_train)
    print(reg.score(x_test, y_test))


def main():
    # Loads access keys in from .env file
    load_dotenv()   
    access_key = os.getenv("ACCESS_KEY")
    secret_access_key = os.getenv("SECRET_ACCESS_KEY")

    client = boto3.client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_access_key)

    print("-------------------\nLoad Data\n-------------------")
    mri_data = get_tabular_data(client)
    # TODO: Get rid of to_csv line
    print_data_shape(mri_data)
    print("-------------------\nExploratory Data Analysis\n-------------------")
    # TODO: Plotly Dashboard of sorts which will be saved as HTML
    print(f"There are {len(pd.unique(mri_data['SID']))} patients.")
    first_only = False
    if first_only == True:
        # Filters data to only include first row instance of each unique SID
        mri_data = mri_data.groupby('SID').first()
    # TODO: Average each patients' MRI scans data to a single row?
    # NOTE: (This is unlikely the best method but it's one I'm doing)
    # names = mri_data['SID'].unique()
    # print(len(names))
    # average = True
    # if average == True:
    #     means = []
    #     for name in names:
    #         means.append(mri_data.loc[mri_data.SID == name, :].mean().tolist())
    #     print("done")
    #     df = pd.concat(means)
    #     df.to_csv("test.csv", index = False)
    #     print(df.head())

    print("-------------------\nNormalize Data\n-------------------")
    normalized_mri_data = normalize_tabular_data(mri_data)
    print(normalized_mri_data['Research Group'])
    print_null_values(normalized_mri_data)
    # NaN values occur because there are a bunch of columns filled with zeros, thus can't be normalized
    # normalized_mri_data = normalized_mri_data.fillna(0)
    # Drop columns
    normalized_mri_data=normalized_mri_data.dropna(axis=1,how='all')

    print("-------------------\nPreprocessing Data\n-------------------")
    # Filter data to remove last 6 columns (irrelevant)
    filtered_data = normalized_mri_data.iloc[:, :normalized_mri_data.shape[1]-6]
    # Checks if any values are NaN
    nan_values = filtered_data.isna()
    nan_columns = nan_values.any()
    columns_with_nan = filtered_data.columns[nan_columns].tolist()
    X = filtered_data.copy()
    y = normalized_mri_data['Research Group']
    y = y.replace("AD", 1)
    y = y.replace("CN", 0)

    linear_regression(X, y)
    print("-------------------\nLinear Regression (PCA)\n-------------------")
    pca = PCA(n_components=190, whiten = 'True')
    pca_x = pca.fit(X).transform(X)
    reg = linear_model.LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size = 0.2, random_state = 4)
    reg.fit(x_train, y_train)
    print(reg.score(x_test, y_test))
    print("-------------------\nLinear Regression (SVD)\n-------------------")
    svd = TruncatedSVD(n_components=190)
    x = svd.fit(X).transform(X)
    reg = linear_model.LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)
    reg.fit(x_train, y_train)
    print(reg.score(x_test, y_test))
    print("-------------------\nRandom Forest (PCA)\n-------------------")
    pca = PCA(n_components=190, whiten = 'True')
    pca_x = pca.fit(X).transform(X)
    rf = RandomForestClassifier(n_estimators=50)
    x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size = 0.2, random_state = 4)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    labels = y_test.values
    count = 0
    for i in range(len(pred)):
        if pred[i]==labels[i]:
            count = count +1
    print(count / float(len(pred)))

if __name__ == "__main__":
    main()