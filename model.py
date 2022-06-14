from os.path import exists

import pandas as pd
from imblearn.over_sampling import SMOTE
from plotly import express as px
from plotly import graph_objs as go
from sklearn import metrics
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class Model:
    """Class Encapsulating classifier functions"""

    def __init__(self, preprocessing, x_train, x_test, y_train, y_test, model,
                 name):
        """Initialisation function

        Args:
            x_train: training features
            x_test: testing features
            y_train: training targets
            y_test:  testing targets
            model: classifier to be trained
            name: string representing model name
        """

        # Initialise environment variable settings
        self.PREPROCESSING = preprocessing

        # Training/Test sets
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = model
        self.name = name
        self.file_name()

    def initialise_optimal_parameters(self, grid=None):
        # TODO: If there aren't pre-existing optimal parameter files
        if exists(f"optimal_parms/{self.f_name}_parms.csv"):
            self.parms = pd.read_csv(
                f"optimal_parms/{self.f_name}_parms.csv").to_dict("index")[0]
            self.model = self.model.set_params(**self.parms)
            return
        # NOTE: Executes cross validation to find optimal parameters for model
        # Only apply RandomSearch if pass as relevant argument
        if grid is not None:
            self.model = RandomizedSearchCV(estimator=self.model,
                                            param_distributions=grid,
                                            n_iter=100,
                                            cv=5,
                                            verbose=2,
                                            random_state=42,
                                            n_jobs=-1)
        else:
            self.model = GridSearchCV(estimator=self.model,
                                      param_distributions=grid,
                                      n_iter=100,
                                      cv=5,
                                      verbose=2,
                                      random_state=42,
                                      n_jobs=-1)

        self.parms = self.model.fit(self.x_train, self.y_train).best_params_
        for key in self.parms.keys():
            if self.parms[key] is None:
                self.parms[key] = "auto"
        # Save optimal parameters to file
        parms = pd.DataFrame.from_dict(self.parms, orient="index").transpose()
        parms.to_csv(
            f"optimal_parms/{self.f_name}_parms.csv", index=False)
        print(self.parms)
        # Save optimal model to file
        # self.model.best_estimator_.save(
        #     f"models/{self.f_name}_model.sav")

    def balance_dataset(self):
        """Balances dataset labels"""
        smote = SMOTE(sampling_strategy="minority", n_jobs=-1)
        self.x_train, self.y_train = smote.fit_resample(
            self.x_train, self.y_train)

    def train_predict(self):
        """Trains classification model and predicts labels

        Args:
            grid (dict, optional): _description_. Defaults to None.
        """

        # Fit features to targets
        self.model.fit(self.x_train, self.y_train)

        # save predictions when applying trained model to test data
        self.predictions = self.model.predict(self.x_test)

        # Accuracy
        self.accuracy = round(
            accuracy_score(self.y_test, self.predictions) * 100, 2)
        self.roc_auc = round(roc_auc_score(self.y_test, self.predictions), 2)
        self.log_loss = round(log_loss(self.y_test, self.predictions), 2)

    def print_metrics(self):
        """Function to display multiple metrics after the model had been trained"""
        print(self.name)
        print(f"[*]\t{self.accuracy}%\t{self.roc_auc}\t{self.log_loss}")

    def save_metrics(self):
        """Saves metrics to file"""
        all_metrics = [self.name, self.accuracy, self.roc_auc, self.log_loss]
        with open('model_metrics.csv', 'a') as file_object:
            my_string = ','.join(map(str, all_metrics))
            file_object.write(f"{my_string},")
            preprocessing = ','.join(map(str, self.PREPROCESSING))
            file_object.write(preprocessing)
            file_object.write('\n')

    def plot_precision_recall(self):
        """Function to display and create plot representing precision and recall of trained model

        Args:
            f_name (_type_, optional): optional file name if wanting to write plot to a file. Defaults to None.
        """

        PrecisionRecallDisplay.from_predictions(self.y_test, self.predictions)
        plt.savefig(self.f_name)
        plt.show()

    def plot_cm(self):
        """Plot confusion matrix

        Args:
            f_name (str, optional): file name. Defaults to None.
        """
        plot_title = f"{self.name} Confusion Matrix ({self.accuracy}%)"
        labels = ["Alzheimer's", "Healthy"]
        cm = metrics.confusion_matrix(self.y_test, self.predictions)

        data = go.Heatmap(z=cm,
                          y=labels,
                          x=labels,
                          hovertemplate="<br>".join([
                              "Predicted: <b>%{x}</b>",
                              "Actual: <b>%{y}</b>",
                              "Occurences %{z}",
                              "<extra></extra>",
                          ]),
                          colorscale="rdylgn",
                          showscale=False)
        annotations = []
        for i, row in enumerate(cm):
            annotations.extend({"x": labels[j], "y": labels[i], "text": str(value), "font": {"color": "black", "size": 20}, "xref": "x1", "yref": "y1", "showarrow": False,} for j, value in enumerate(row))

        layout = go.Layout(
            title=dict(text=plot_title, x=0.5),
            xaxis=dict(title="Predicted"),
            yaxis=dict(title="Actual"),
            annotations=annotations,
        )

        fig = go.Figure(data=data, layout=layout)
        print(f"[*]\tSaved as {self.f_name}.png")
        fig.write_image(f"plots/confusion_matrices/{self.f_name}.png")

    # MISC
    def file_name(self):
        """generate file name from name"""
        # name to lower case
        self.f_name = self.name.lower()
        # replace spaces with underscores
        self.f_name = self.f_name.replace(" ", "_")
