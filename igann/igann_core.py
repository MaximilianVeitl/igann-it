import time
import torch
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, LassoCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import abess.linear
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
# vei:
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from itertools import product, combinations
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator
from collections import defaultdict


warnings.simplefilter("once", UserWarning)


class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, dummy_columns):
        self.columns = None
        self.dummy_columns = dummy_columns

        self.cols_per_feat = (
            dict()
        )  # should be a list of len=2 for key = categorical_feature_name, val = [[remaining_column_names], dropped_column_name]
        self._cols_per_feat_needed = True

    def fit(self, X, y=None):
        for c in X.columns:
            for val in X[c].unique():
                if c not in self.cols_per_feat.keys():
                    self.cols_per_feat[c] = []
                self.cols_per_feat[c].append(str(c + "_" + str(val)))

        self.columns = pd.get_dummies(
            X, columns=self.dummy_columns, drop_first=True
        ).columns
        return self

    def transform(self, X):
        X_new = pd.get_dummies(X, columns=self.dummy_columns, drop_first=True)

        if self._cols_per_feat_needed:
            for key, val in self.cols_per_feat.items():
                curr = [[], ""]
                for v in val:
                    if v not in X_new.columns:
                        curr[1] = v
                    else:
                        curr[0].append(v)
                self.cols_per_feat[key] = curr
            self._cols_per_feat_needed = False
        return X_new.reindex(columns=self.columns, fill_value=0)


class torch_Ridge:
    def __init__(self, alpha, device):
        self.coef_ = None
        self.alpha = alpha
        self.device = device

    def fit(self, X, y):
        self.coef_ = torch.linalg.solve(
            X.T @ X + self.alpha * torch.eye(X.shape[1]).to(self.device), X.T @ y
        )

    def predict(self, X):
        return X.to(self.device) @ self.coef_


class ELM_Regressor:
    """
    This class represents one single hidden layer neural network for a regression task.
    Trainable parameters are only the parameters from the output layer. The parameters
    of the hidden layer are sampled from a normal distribution. This increases the training
    performance significantly as it reduces the training task to a regularized linear
    regression (Ridge Regression), see "Extreme Learning Machines" for more details.
    """

    def __init__(
        self,
        n_input,
        n_categorical_cols,
        n_hid,
        seed=0,
        elm_scale=10,
        elm_alpha=0.0001,
        act="elu",
        device="cpu",
    ):
        """
        Input parameters:
        - n_input: number of inputs/features (should be X.shape[1])
        - n_cat_cols: number of categorically encoded features
        - n_hid: number of hidden neurons for the base functions
        - seed: This number sets the seed for generating the random weights. It should
                be different for each regressor
        - elm_scale: the scale which is used to initialize the weights in the hidden layer of the
                 model. These weights are not changed throughout the optimization.
        - elm_alpha: the regularization of the ridge regression.
        - act: the activation function in the model. can be 'elu', 'relu' or a torch activation function.
        - device: the device on which the regressor should train. can be 'cpu' or 'cuda'.
        """
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.n_numerical_cols = n_input - n_categorical_cols
        self.n_categorical_cols = n_categorical_cols
        # The following are the random weights in the model which are not optimized.
        self.hidden_list = torch.normal(
            mean=torch.zeros(self.n_numerical_cols, self.n_numerical_cols * n_hid),
            std=elm_scale,
        ).to(device)

        mask = torch.block_diag(*[torch.ones(n_hid)] * self.n_numerical_cols).to(device)
        self.hidden_mat = self.hidden_list * mask
        self.output_model = None
        self.n_input = n_input

        self.n_hid = n_hid
        self.elm_scale = elm_scale
        self.elm_alpha = elm_alpha
        if act == "elu":
            self.act = torch.nn.ELU()
        elif act == "relu":
            self.act = torch.nn.ReLU()
        else:
            self.act = act
        self.device = device

    def get_hidden_values(self, X):
        """
        This step computes the values in the hidden layer. For this, we iterate
        through the input features and multiply the feature values with the weights
        from hidden_list. After applying the activation function, we return the result
        in X_hid
        """
        X_hid = X[:, : self.n_numerical_cols] @ self.hidden_mat
        X_hid = self.act(X_hid)
        X_hid = torch.hstack((X_hid, X[:, self.n_numerical_cols :]))

        return X_hid

    def predict(self, X, hidden=False):
        """
        This function makes a full prediction with the model for a given input X.
        """
        if hidden:
            X_hid = X
        else:
            X_hid = self.get_hidden_values(X)

        # Now, we can use the values in the hidden layer to make the prediction with
        # our ridge regression
        out = X_hid @ self.output_model.coef_
        return out

    def predict_single(self, x, i):
        """
        This function computes the partial output of one base function for one feature.
        Note, that the bias term is not used for this prediction.
        Input parameters:
        x: a vector representing the values which are used for feature i
        i: the index of the feature that should be used for the prediction
        """

        # See self.predict for the description - it's almost equivalent.
        x_in = x.reshape(len(x), 1)
        if i < self.n_numerical_cols:
            # numerical feature
            x_in = x_in @ self.hidden_mat[
                i, i * self.n_hid : (i + 1) * self.n_hid
            ].unsqueeze(0)
            x_in = self.act(x_in)
            out = x_in @ self.output_model.coef_[
                i * self.n_hid : (i + 1) * self.n_hid
            ].unsqueeze(1)
        else:
            # categorical feature
            start_idx = self.n_numerical_cols * self.n_hid + (i - self.n_numerical_cols)
            out = x_in @ self.output_model.coef_[start_idx : start_idx + 1].unsqueeze(1)
        return out

    # vei: added this function to predict the output of a feature pair
    def predict_it(self, X, int_pair):
        num_idx = [int_pair[0] * self.n_hid + i for i in range(self.n_hid)]
        cat_idx = [i - self.n_numerical_cols + (self.n_numerical_cols * self.n_hid) for i in int_pair[1:]]
        coef_idx = num_idx + cat_idx
        X_hid = X[:, 0].reshape(len(X), 1) @ self.hidden_mat[int_pair[0], num_idx].reshape(1, self.n_hid)
        X_hid = self.act(X_hid)
        X_hid = torch.cat((X_hid, X[:, 1:]), dim=1)
        out = X_hid @ self.output_model.coef_[[coef_idx]]

        return out

    def fit(self, X, y, mult_coef):
        """
        This function fits the ELM on the training data (X, y).
        """
        X_hid = self.get_hidden_values(X)
        X_hid_mult = X_hid * mult_coef
        # Fit the ridge regression on the hidden values.
        m = torch_Ridge(alpha=self.elm_alpha, device=self.device)
        m.fit(X_hid_mult, y)
        self.output_model = m
        return X_hid

class ELM_IT_Regressor:
    def __init__(
        self, 
        best_combination, 
        n_hid, 
        seed=0,
        elm_scale=10,
        elm_alpha=0.0001,
        act="elu", 
        device="cpu",
    ):
        """
        hier kommt noch die Beschreibung rein
        """
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.best_combination = best_combination
        self.n_hid = n_hid
        self.device = device
        n_cat = len(self.best_combination) # number of categories in the categorical feature (actually -1 as num_feature is included, but + 1 as first category was dropped in encoding step)
        # random weights for hidden layer
        self.hidden_list = torch.normal(mean=torch.zeros(n_cat, n_cat * self.n_hid), std=elm_scale).to(device)

        mask = torch.block_diag(*[torch.ones(self.n_hid)] * n_cat).to(self.device) # mask to keep in random weight matrix only in first row weights for neurons for first category, in second row for second category, ...
        self.hidden_mat = self.hidden_list * mask # delete not needed random weights
        self.output_model = None
        self.elm_scale = elm_scale
        self.elm_alpha = elm_alpha
        if act == "elu":
            self.act = torch.nn.ELU()
        elif act == "relu":
            self.act = torch.nn.ReLU()
        else:
            self.act = act

    def get_hidden_values(self, X):
        X_num_feature = X[:, 0]
        X_cat_feature = torch.hstack((X[:, 1:], 1 ^ X[:, 1:].sum(dim=1).unsqueeze(1).int())) # add new column to restore first category (was dropped in encoding step)
        X_comb_feature = X_num_feature.unsqueeze(1) * X_cat_feature
        X_hid = X_comb_feature @ self.hidden_mat # create list of hidden values
        X_hid = self.act(X_hid) # apply activation function

        return X_hid
    
    def predict(self, X, hidden=False):
        """
        """
        if hidden:
            X_hid = X
        else:
            X_hid = self.get_hidden_values(X)

        out = X_hid @ self.output_model.coef_
        return out

    def fit(self, X, y, mult_coef):
        X_hid = self.get_hidden_values(X)
        X_hid_mult = X_hid * mult_coef
        m = torch_Ridge(alpha=self.elm_alpha, device=self.device)
        m.fit(X_hid_mult, y)
        self.output_model = m
        return X_hid

class IGANN:
    """
    This class represents the IGANN model. It can be used like a
    sklearn model (i.e., it includes .fit, .predict, .predict_proba, ...).
    The model can be used for a regression task or a binary classification task.
    For binary classification, the labels must be set to -1 and 1 (Note that labels with
    0 and 1 are transformed automatically). The model first fits a linear model and then
    subsequently adds ELMs according to a boosting framework.
    """

    def __init__(
        self,
        task="classification",
        n_hid=10,
        n_estimators=5000,
        boost_rate=0.1,
        init_reg=1,
        elm_scale=1,
        elm_alpha=1,
        sparse=0,
        act="elu",
        early_stopping=50,
        device="cpu",
        random_state=1,
        optimize_threshold=False,
        verbose=0,
        igann_it=True,
        interaction_detection_method='dt',
    ):
        """
        Initializes the model. Input parameters:
        task: defines the task, can be 'regression' or 'classification'
        n_hid: the number of hidden neurons for one feature
        n_estimators: the maximum number of estimators (ELMs) to be fitted.
        boost_rate: Boosting rate.
        init_reg: the initial regularization strength for the linear model.
        elm_scale: the scale of the random weights in the elm model.
        elm_alpha: the regularization strength for the ridge regression in the ELM model.
        sparse: Tells if IGANN should be sparse or not. Integer denotes the max number of used features
        act: the activation function in the ELM model. Can be 'elu', 'relu' or a torch activation function.
        early_stopping: we use early stopping which means that we don't continue training more ELM
        models, if there has been no improvements for 'early_stopping' number of iterations.
        device: the device on which the model is optimized. Can be 'cpu' or 'cuda'
        random_state: random seed.
        optimize_threshold: if True, the threshold for the classification is optimized using train data only and using the ROC curve. Otherwise, per default the raw logit value greater 0 means class 1 and less 0 means class -1.
        verbose: tells how much information should be printed when fitting. Can be 0 for (almost) no
        information, 1 for printing losses, and 2 for plotting shape functions in each iteration.
        interaction_detection_method: method to detect interactions. Can be 'rulefit' or 'dt' or 'FAST'.
        """
        self.task = task
        self.n_hid = n_hid
        self.elm_scale = elm_scale
        self.elm_alpha = elm_alpha
        self.init_reg = init_reg
        self.act = act
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.sparse = sparse
        self.device = device
        self.random_state = random_state
        self.optimize_threshold = optimize_threshold
        self.verbose = verbose
        self.igann_it = igann_it
        self.interaction_detection_method = interaction_detection_method
        ###### this is not needed since these are just for one run! ######
        # self.regressors = []
        # self.boosting_rates = []
        # self.train_scores = []
        # self.val_scores = []
        # self.train_losses = []
        # self.val_losses = []
        # self.test_losses = []
        # self.regressor_predictions = []
        self.boost_rate = boost_rate
        self.target_remapped_flag = False
        """Is set to true during the fit method if the target (y) is remapped to -1 and 1 instead of 0 and 1."""

        ###### moved this to fit function to be more in line with sklearn. I also think would be normal to have to classes for reg and class######
        # if task == 'classification':
        #     # todo: torch
        #     self.init_classifier = LogisticRegression(penalty='l1', solver='liblinear', C=1 / self.init_reg,
        #                                              random_state=random_state)
        #     #self.init_classifier = LogisticRegression(penalty='none', solver='lbfgs',
        #     #                                          max_iter=2000, random_state=1337)
        #     self.criterion = lambda prediction, target: torch.nn.BCEWithLogitsLoss()(prediction,
        #                                                                              torch.nn.ReLU()(target))
        # elif task == 'regression':
        #     # todo: torch
        #     self.init_classifier = Lasso(alpha=self.init_reg)
        #     self.criterion = torch.nn.MSELoss()
        # else:
        #     warnings.warn('Task not implemented. Can be classification or regression')

    def _clip_p(self, p):
        if torch.max(p) > 100 or torch.min(p) < -100:
            warnings.warn(
                "Cutting prediction to [-100, 100]. Did you forget to scale y? Consider higher regularization elm_alpha."
            )
            return torch.clip(p, -100, 100)
        else:
            return p

    def _clip_p_numpy(self, p):
        if np.max(p) > 100 or np.min(p) < -100:
            warnings.warn(
                "Cutting prediction to [-100, 100]. Did you forget to scale y? Consider higher regularization elm_alpha."
            )
            return np.clip(p, -100, 100)
        else:
            return p

    def _loss_sqrt_hessian(self, y, p):
        """
        This function computes the square root of the hessians of the log loss or the mean squared error.
        """
        if self.task == "classification":
            return 0.5 / torch.cosh(0.5 * y * p)
        else:
            return torch.sqrt(torch.tensor([2.0]).to(self.device))

    def _get_y_tilde(self, y, p):
        if self.task == "classification":
            return y / torch.exp(0.5 * y * p)
        else:
            return torch.sqrt(torch.tensor(2.0).to(self.device)) * (y - p)

    def _reset_state(self):
        self.regressors = []
        self.boosting_rates = []
        self.train_scores = []
        self.val_scores = []
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.regressor_predictions = []
        # vei: store regressors_it separately
        self.regressors_it = []
        # vei: store boosting rates for regressors_it
        self.boosting_rates_it = []
        # vei: store train losses for regressors_it
        self.train_losses_it = []
        # vei: store val losses for regressors_it
        self.val_losses_it = []
        # vei: store regressor_it predictions separately
        self.regressor_predictions_it = []

    def _preprocess_feature_matrix(self, X, fit_dummies=False):
        if type(X) != pd.DataFrame:
            warnings.warn(
                "Please provide a pandas dataframe as input for X, as IGANN derives the categorical/numerical variables from the datatypes. We stop here for now."
            )
            return

        X.columns = [str(c) for c in X.columns]
        X = X.reindex(sorted(X.columns), axis=1)
        categorical_cols = sorted(
            X.select_dtypes(include=["category", "object"]).columns.tolist()
        )
        numerical_cols = sorted(list(set(X.columns) - set(categorical_cols)))

        if len(numerical_cols) > 0:
            X_num = torch.from_numpy(X[numerical_cols].values).float()
            self.n_numerical_cols = X_num.shape[1]
        else:
            self.n_numerical_cols = 0
        
        #vei: store dropped categories
        self.dropped_categories = {}

        if len(categorical_cols) > 0:
            if fit_dummies:
                self.get_dummies = GetDummies(categorical_cols)
                self.get_dummies.fit(X[categorical_cols])
            one_hot_encoded = self.get_dummies.transform(X[categorical_cols])
            encoded_list = [
                [c for c in one_hot_encoded.columns if c.startswith(f)]
                for f in categorical_cols
            ]
            original_list = [
                [categorical_cols[i]] * len(encoded_list[i])
                for i in range(len(encoded_list))
            ]

            self.dummy_encodings = dict(
                zip(self._flatten(encoded_list), self._flatten(original_list))
            )
            
            # vei: store dropped categories
            for col in categorical_cols:
                all_categories = [str(val) for val in X[col].unique()]
                dummy_cols = [c.split("_", 1)[-1] for c in one_hot_encoded.columns if c.startswith(col + "_")]
                dropped_category = list(set(all_categories) - set(dummy_cols))
                if dropped_category:
                    self.dropped_categories[col] = dropped_category[0] 

            X_cat = torch.from_numpy(one_hot_encoded.values.astype(float)).float()
            self.n_categorical_cols = X_cat.shape[1]
            self.feature_names = numerical_cols + list(one_hot_encoded.columns)
        else:
            self.n_categorical_cols = 0
            self.feature_names = numerical_cols

        if self.sparse > self.n_numerical_cols + self.n_categorical_cols:
            warnings.warn("The parameter sparse is higher than the number of features")
            self.sparse = self.n_numerical_cols + self.n_categorical_cols

        if self.n_numerical_cols > 0 and self.n_categorical_cols > 0:
            X = torch.hstack((X_num, X_cat))
        elif self.n_numerical_cols > 0:
            X = X_num
        else:
            X = X_cat

        # vei:
        if len(categorical_cols) > 0:
            self.grouped_encoded_features = [
                [self.feature_names.index(c) for c in group]
                for group in encoded_list
            ]

        return X

    def fit(
        self,
        X,
        y,
        val_set=None,
        eval=None,
        plot_fixed_features=None,
        fitted_dummies=None,
    ):
        """
        This function fits the model on training data (X, y).
        Parameters:
        X: the feature matrix
        y: the targets
        val_set: can be tuple (X_val, y_val) for a defined validation set. If not set,
        it will be split from the training set randomly.
        eval: can be tuple (X_test, y_test) for additional evaluation during training
        plot_fixed_features: Per default the most important features are plotted for verbose=2.
        This can be changed here to keep track of the same feature throughout training.
        """
        if self.task == "classification":
            # todo: torch
            self.init_classifier = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1 / self.init_reg,
                random_state=self.random_state,
            )
            # self.init_classifier = LogisticRegression(penalty='none', solver='lbfgs',
            #                                          max_iter=2000, random_state=1337)
            self.criterion = lambda prediction, target: torch.nn.BCEWithLogitsLoss()(
                prediction, torch.nn.ReLU()(target)
            )
        elif self.task == "regression":
            # todo: torch
            self.init_classifier = Lasso(alpha=self.init_reg)
            self.criterion = torch.nn.MSELoss()
        else:
            warnings.warn("Task not implemented. Can be classification or regression")

        self._reset_state()
        if fitted_dummies != None:
            self.get_dummies = fitted_dummies
            X = self._preprocess_feature_matrix(X, fit_dummies=False)

        else:
            X = self._preprocess_feature_matrix(X, fit_dummies=True)

        if type(y) == pd.Series or type(y) == pd.DataFrame:
            y = y.values

        y = torch.from_numpy(y.squeeze()).float()

        if self.task == "classification":
            # In the case of targets in {0,1}, transform them to {-1,1} for optimization purposes
            if torch.min(y) != -1:
                self.target_remapped_flag = True
                y = 2 * y - 1
        if self.sparse > 0:
            feature_indizes = self._select_features(X, y)
            self.feature_names = np.array(
                [f for e, f in enumerate(self.feature_names) if e in feature_indizes]
            )
            X = X[:, feature_indizes]
            self.feature_indizes = feature_indizes
        else:
            self.feature_indizes = np.arange(X.shape[1])

        # Fit the linear model on all data
        self.init_classifier.fit(X, y)

        # Split the data into train and validation data and compute the prediction of the
        # linear model. For regression this is straightforward, for classification, we
        # work with the logits and not the probabilities. That's why we multiply X with
        # the coefficients and don't use the predict_proba function.
        if self.task == "classification":
            if val_set == None:
                if self.verbose >= 1:
                    print("Splitting data")
                X, X_val, y, y_val = train_test_split(
                    X, y, test_size=0.15, stratify=y, random_state=self.random_state
                )
            else:
                X_val = val_set[0]
                y_val = val_set[1]

            y_hat = torch.squeeze(
                torch.from_numpy(self.init_classifier.coef_.astype(np.float32))
                @ torch.transpose(X, 0, 1)
            ) + float(self.init_classifier.intercept_)
            y_hat_val = torch.squeeze(
                torch.from_numpy(self.init_classifier.coef_.astype(np.float32))
                @ torch.transpose(X_val, 0, 1)
            ) + float(self.init_classifier.intercept_)

        else:
            if val_set == None:
                if self.verbose >= 1:
                    print("Splitting data")
                X, X_val, y, y_val = train_test_split(
                    X, y, test_size=0.15, random_state=self.random_state
                )
            else:
                X_val = val_set[0]
                y_val = val_set[1].squeeze()

            y_hat = torch.from_numpy(
                self.init_classifier.predict(X).squeeze().astype(np.float32)
            )
            y_hat_val = torch.from_numpy(
                self.init_classifier.predict(X_val).squeeze().astype(np.float32)
            )

        # Store some information about the dataset which we later use for plotting.
        self.X_min = list(X.min(axis=0))
        self.X_max = list(X.max(axis=0))
        self.unique = [torch.unique(X[:, i]) for i in range(X.shape[1])]
        self.hist = [torch.histogram(X[:, i]) for i in range(X.shape[1])]

        if self.verbose >= 1:
            print("Training shape: {}".format(X.shape))
            print("Validation shape: {}".format(X_val.shape))
            print("Regularization: {}".format(self.init_reg))

        train_loss_init = self.criterion(y_hat, y)
        val_loss_init = self.criterion(y_hat_val, y_val)

        if self.verbose >= 1:
            print(
                "Train: {:.4f} Val: {:.4f} {}".format(
                    train_loss_init, val_loss_init, "init"
                )
            )

        X, y, y_hat, X_val, y_val, y_hat_val = (
            X.to(self.device),
            y.to(self.device),
            y_hat.to(self.device),
            X_val.to(self.device),
            y_val.to(self.device),
            y_hat_val.to(self.device),
        )

        self._run_optimization(
            X,
            y,
            y_hat,
            X_val,
            y_val,
            y_hat_val,
            eval,
            val_loss_init,
            plot_fixed_features,
        )

        if self.task == "classification" and self.optimize_threshold:
            self.best_threshold = self._optimize_classification_threshold(X, y)

        self._get_feature_importance(first_call=True)
        return

    def get_params(self, deep=True):
        return {
            "task": self.task,
            "n_hid": self.n_hid,
            "elm_scale": self.elm_scale,
            "elm_alpha": self.elm_alpha,
            "init_reg": self.init_reg,
            "act": self.act,
            "n_estimators": self.n_estimators,
            "early_stopping": self.early_stopping,
            "sparse": self.sparse,
            "device": self.device,
            "random_state": self.random_state,
            "optimize_threshold": self.optimize_threshold,
            "verbose": self.verbose,
            "boost_rate": self.boost_rate,
            # "target_remapped_flag": self.target_remapped_flag,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if not hasattr(self, parameter):
                raise ValueError(
                    "Invalid parameter %s for estimator %s. Check the list of available parameters with `estimator.get_params().keys()`."
                    % (parameter, self)
                )
            setattr(self, parameter, value)
        return self

    def score(self, X, y, metric=None):
        predictions = self.predict(X)
        if self.task == "regression":
            # if these is no metric specified use default regression metric "mse"
            if metric is None:
                metric = "mse"
            metric_dict = {"mse": mean_squared_error, "r_2": r2_score}
            return metric_dict[metric](y, predictions)
        else:
            if metric is None:
                # if there is no metric specified use default classification metric "accuracy"
                metric = "accuracy"
            metric_dict = {
                "accuracy": accuracy_score,
                "precision": precision_score,
                "recall": recall_score,
                "f1": f1_score,
            }
            return metric_dict[metric](y, predictions)

    def _run_optimization(
        self, X, y, y_hat, X_val, y_val, y_hat_val, eval, best_loss, plot_fixed_features
    ):
        """
        This function runs the optimization for ELMs with single features. This function should not be called from outside.
        Parameters:
        X: the training feature matrix
        y: the training targets
        y_hat: the current prediction for y
        X_val: the validation feature matrix
        y_val: the validation targets
        y_hat_val: the current prediction for y_val
        eval: can be tuple (X_test, y_test) for additional evaluation during training
        best_loss: best previous loss achieved. This is to keep track of the overall best sequence of ELMs.
        plot_fixed_features: Per default the most important features are plotted for verbose=2.
        This can be changed here to keep track of the same feature throughout training.
        """

        counter_no_progress = 0
        best_iter = 0
        best_y_hat = y_hat  # vei: store best y_hat for ELM_IT
        best_y_hat_val = y_hat_val

        # Sequentially fit one ELM after the other. Max number is stored in self.n_estimators.
        for counter in range(self.n_estimators):
            hessian_train_sqrt = self._loss_sqrt_hessian(y, y_hat)
            y_tilde = torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(
                y, y_hat
            )

            # Init ELM
            regressor = ELM_Regressor(
                n_input=X.shape[1],
                n_categorical_cols=self.n_categorical_cols,
                n_hid=self.n_hid,
                seed=counter,
                elm_scale=self.elm_scale,
                elm_alpha=self.elm_alpha,
                act=self.act,
                device=self.device,
            )

            # Fit ELM regressor
            X_hid = regressor.fit(
                X,
                y_tilde,
                torch.sqrt(torch.tensor(0.5).to(self.device))
                * self.boost_rate
                * hessian_train_sqrt[:, None],
            )

            # Make a prediction of the ELM for the update of y_hat
            train_regressor_pred = regressor.predict(X_hid, hidden=True).squeeze()
            val_regressor_pred = regressor.predict(X_val).squeeze()

            self.regressor_predictions.append(train_regressor_pred)

            # Update the prediction for training and validation data
            y_hat += self.boost_rate * train_regressor_pred
            y_hat_val += self.boost_rate * val_regressor_pred

            y_hat = self._clip_p(y_hat)
            y_hat_val = self._clip_p(y_hat_val)

            train_loss = self.criterion(y_hat, y)
            val_loss = self.criterion(y_hat_val, y_val)

            # Keep the ELM, the boosting rate and losses in lists, so
            # we can later use them again.
            self.regressors.append(regressor)
            self.boosting_rates.append(self.boost_rate)
            self.train_losses.append(train_loss.cpu())
            self.val_losses.append(val_loss.cpu())

            # This is the early stopping mechanism. If there was no improvement on the
            # validation set, we increase a counter by 1. If there was an improvement,
            # we set it back to 0.
            counter_no_progress += 1
            if val_loss < best_loss:
                best_iter = counter + 1
                best_loss = val_loss
                counter_no_progress = 0
                best_y_hat = y_hat  # vei: store best y_hat for ELM_IT
                best_y_hat_val = y_hat_val

            if self.verbose >= 1:
                self._print_results(
                    counter,
                    counter_no_progress,
                    eval,
                    self.boost_rate,
                    train_loss,
                    val_loss,
                )

            # Stop training if the counter for early stopping is greater than the parameter we passed.
            if counter_no_progress > self.early_stopping and self.early_stopping > 0:
                break

            if self.verbose >= 2:
                if counter % 5 == 0:
                    if plot_fixed_features != None:
                        self.plot_single(plot_by_list=plot_fixed_features)
                    else:
                        self.plot_single()

        if self.early_stopping > 0:
            # We remove the ELMs that did not improve the performance. Most likely best_iter equals self.early_stopping.
            if self.verbose > 0:
                print(f"Cutting at {best_iter}")
            self.regressors = self.regressors[:best_iter]
            self.boosting_rates = self.boosting_rates[:best_iter]

            # vei: AB HIER ELM_IT
        if self.igann_it == True:
            if self.n_numerical_cols > 0 and self.n_categorical_cols > 0: # interaction term only possible if both numerical and categorical features are present
                # update y_hat to prediction of best ELM iteration (or of initial model if no ELM was fitted)
                y_hat = best_y_hat
                y_hat_val = best_y_hat_val
                y_tilde = torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(
                        y, y_hat
                    )
                # execute method to get the best feature combination
                self.best_combination = self._find_interactions(X, y_tilde, self.interaction_detection_method)

                if self.igann_it == True:
                    # X dataset only for cat features
                    X_it = X[:, self.best_combination]
                    X_val_it = X_val[:, self.best_combination]

                    # vei: store unique values of interaction features
                    self.unique_it = torch.unique(X_it, dim=0)

                    counter_no_progress = 0
                    best_iter = 0

                    # Sequentially fit one ELM for the Interaction pair after another. Max number is stored in self.n_estimators.
                    for counter in range(self.n_estimators):
                        hessian_train_sqrt = self._loss_sqrt_hessian(y, y_hat)
                        y_tilde = torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(
                            y, y_hat
                        )

                        # init ELM_IT Regressor
                        regressor = ELM_IT_Regressor(
                            best_combination = self.best_combination, 
                            n_hid = self.n_hid,
                            seed=counter,
                            elm_scale=self.elm_scale,
                            elm_alpha=self.elm_alpha,
                            act=self.act, 
                            device=self.device,
                            )

                        # Fit ELM regressor
                        X_hid = regressor.fit(
                            X_it, 
                            y_tilde,
                            torch.sqrt(torch.tensor(0.5).to(self.device))
                            * self.boost_rate
                            * hessian_train_sqrt[:, None],
                            )

                        # Make a prediction of the ELM for the update of y_hat
                        train_regressor_pred = regressor.predict(X_hid, hidden=True).squeeze()
                        val_regressor_pred = regressor.predict(X_val_it).squeeze()

                        self.regressor_predictions_it.append(train_regressor_pred)

                        # Update the prediction for training and validation data
                        y_hat += self.boost_rate * train_regressor_pred
                        y_hat_val += self.boost_rate * val_regressor_pred

                        y_hat = self._clip_p(y_hat)
                        y_hat_val = self._clip_p(y_hat_val)

                        train_loss = self.criterion(y_hat, y)
                        val_loss = self.criterion(y_hat_val, y_val)

                        # Keep the ELM, the boosting rate and losses in lists, so
                        # we can later use them again.
                        self.regressors_it.append(regressor)
                        self.boosting_rates_it.append(self.boost_rate)
                        self.train_losses_it.append(train_loss.cpu())
                        self.val_losses_it.append(val_loss.cpu())

                        # early stopping
                        counter_no_progress += 1
                        if val_loss < best_loss:
                            best_iter = counter + 1
                            best_loss = val_loss
                            counter_no_progress = 0

                        # Stop training if the counter for early stopping is greater than the parameter we passed.
                        if counter_no_progress > self.early_stopping and self.early_stopping > 0:
                            break

                if self.early_stopping > 0:
                    # We remove the ELMs that did not improve the performance. Most likely best_iter equals self.early_stopping.
                    if self.verbose > 0:
                        print(f"Cutting at {best_iter}")
                    self.regressors_it = self.regressors_it[:best_iter]
                    self.boosting_rates_it = self.boosting_rates_it[:best_iter]

        return best_loss

    # vei:
    def _constraint_dt(self, X, y_tilde):
        """
        This function fits a decision tree to determine the most important combination of one 
        categorical and one numberical feature.
        """
        
        numerical_features = list(range(self.n_numerical_cols))
        categorical_features = self.grouped_encoded_features

        best_score = np.inf if self.task == 'regression' else 0
        best_combination = None

        for num_var, cat_var in product(numerical_features, categorical_features):
            # Trainiere Modell mit den Variablen num_var und cat_var
            X_dt = X[:, [num_var] + cat_var]
            if self.task == 'regression':
                tree = DecisionTreeRegressor(random_state=42, max_depth=3, criterion='squared_error')
            else: 
                tree = DecisionTreeClassifier(random_state=42, max_depth=3, criterion='gini')
            tree.fit(X_dt, y_tilde)
            y_pred = tree.predict(X_dt)

            if self.task == 'regression':
                score = mean_squared_error(y_tilde, y_pred)
                is_better = score < best_score
            else:
                score = accuracy_score(y_tilde, y_pred)
                is_better = score > best_score

            if len(set(tree.tree_.feature).difference([-2])) > 1: # only consider decistion tree with more than one feature
                if is_better:
                    best_score = score
                    best_combination = [num_var] + cat_var

        if best_combination == None:
            print('No feature combination found. All decision trees only use one feature. Model does not capture interactions.') 

        return best_combination
    
    # vei:
    def _FAST_detection(self, X, y_tilde, n_bins):
        # andere funktionen einbauen
        # X_complete_decoded evtl löschen
        """
        FAST - Interaction Detection

        This module exposes a method called FAST [1] to measure and rank the strengths
        of the interaction of all pairs of features in a dataset.

        [1] https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf

        Args:
            X (torch.Tensor): Feature-Matrix.
            y_tilde (torch.Tensor): Residuals of univariate model.
            n_bins (int): Number of bins in which the numerical features are split. Defaults to 20.
            Returns:
            dict: Ein Dictionary mit den Ergebnissen.
        """
        # Split X in numerical and categorical features
        # X_num
        X_num = X[:, :self.n_numerical_cols]
        # split X_cat and decode it
        X_cat = X[:, self.n_numerical_cols:]
        for indices in self.grouped_encoded_features:
            cat = 0
            new_col = torch.zeros(X_cat.shape[0])
            for i in indices:
                cat += 1
                new_col += cat * X_cat[:, i-self.n_numerical_cols]
            X_cat = torch.hstack((X_cat, new_col.reshape(-1, 1)))
        X_cat = X_cat[:, self.n_categorical_cols:]

        # Marginal und cumulative histograms for numerical features
        histo_mar_num = {}
        histo_cum_num = {}
        for feature_idx in range(X_num.shape[1]):
            bins = np.percentile(X_num[:, feature_idx], np.linspace(0, 100, n_bins + 1))
            bins[-1] += 1e-6
            bin_indices = np.digitize(X_num[:, feature_idx], bins) - 1
            histo_mar_num[feature_idx] = {
                "bin_edges": bins,
                "H_t": np.zeros(n_bins),
                "H_w": np.zeros(n_bins),
            }
            histo_cum_num[feature_idx] = {
                "bin_edges": bins,
                "H_t": np.zeros(n_bins),
                "H_w": np.zeros(n_bins),
            }
            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                H_t = y_tilde[mask].sum()
                H_w = mask.sum()
                histo_mar_num[feature_idx]["H_t"][bin_idx] = H_t
                histo_mar_num[feature_idx]["H_w"][bin_idx] = H_w
                histo_cum_num[feature_idx]["H_t"][bin_idx] = histo_cum_num[feature_idx]["H_t"][bin_idx-1] + H_t
                histo_cum_num[feature_idx]["H_w"][bin_idx] = histo_cum_num[feature_idx]["H_w"][bin_idx-1] + H_w

        # Histograms for categorical features
        histo_cat = {}
        for feature_idx in range(X_cat.shape[1]):
            unique_values = np.unique(X_cat[:, feature_idx]).astype(int).tolist()
            # Marginal histograms for single categories
            marg_histo = {
                "unique_values": unique_values,
                "H_t": np.zeros(len(unique_values)),
                "H_w": np.zeros(len(unique_values)),
            }
            for unique_value_idx, unique_value in enumerate(unique_values):
                mask = X_cat[:, feature_idx] == unique_value
                mask = mask.numpy()
                marg_histo["H_t"][unique_value_idx] = y_tilde[mask].sum()
                marg_histo["H_w"][unique_value_idx] = mask.sum()
            # Generate all possible divisions of categories if splitting into two groups
            partitions = []
            right_partitions = []
            if len(unique_values) > 6:
                warnings.warn("The " + str((feature_idx+1)) + "-th categorical feature has k = " + str(len(unique_values)) + " unique values, resulting in 2**(k-1) - 1 possible paritions of the feature into two groups. Simulating the best split may take a while. Consider using an alternative method for the interaction detection.")
            for size in range(1, len(unique_values)):
                for left in combinations(unique_values, size):
                    left_set = set(left)
                    right_set = set(unique_values) - left_set
                    if left_set not in right_partitions:
                        partitions.append(left_set)
                        right_partitions.append(right_set)
            histo_cat[feature_idx] = {
                "partition": partitions,
                "H_t": np.zeros(len(partitions)),
                "H_w": np.zeros(len(partitions)),
            }
            # Histograms for combinations of categories within partitions
            for partition_idx, partition in enumerate(partitions):
                if len(partition) == 1:
                    histo_cat[feature_idx]["H_t"][partition_idx] = marg_histo["H_t"][unique_values.index(list(partition)[0])]
                    histo_cat[feature_idx]["H_w"][partition_idx] = marg_histo["H_w"][unique_values.index(list(partition)[0])]
                else:
                    H_t = 0
                    H_w = 0
                    for value in partition:
                        value_idx = marg_histo["unique_values"].index(value)
                        H_t += marg_histo["H_t"][value_idx]
                        H_w += marg_histo["H_w"][value_idx]
                    histo_cat[feature_idx]["H_t"][partition_idx] = H_t
                    histo_cat[feature_idx]["H_w"][partition_idx] = H_w 

        # Digitize X_num in bins
        X_digitized = X_num.clone()
        for column in range(X_num.shape[1]):
            bins = histo_cum_num[column]['bin_edges']
            bin_indices = np.digitize(X_num[:, column], bins) - 1
            X_digitized[:, column] = torch.tensor(bin_indices, dtype=torch.int64)
        # Generate all possible feature pairs
        numerical_features = list(range(self.n_numerical_cols))
        categorical_features = [x + self.n_numerical_cols for x in list(range(X_cat.shape[1]))]
        feature_pairs = list(product(numerical_features, categorical_features))

        RSS = {
            "pair": feature_pairs,
            "RSS": np.zeros(len(feature_pairs)),
            "split_num": np.array([""] * len(feature_pairs)),
            "split_cat": np.array([""] * len(feature_pairs)),
            }
        pair_idx = 0
        for pair in feature_pairs:
            num_feat = pair[0]
            cat_feat = pair[1] - self.n_numerical_cols
            n_unique_values = X_cat[:, cat_feat].unique().shape[0]

            # create lookup tables to store the sum of residuals, the number of samples, and the interaction predictor for each quadrant of the split
            columns = []
            for i in histo_cat[cat_feat]["partition"]:
                columns.append(str(i))
            index = [f"< {value:.2f}" for value in histo_cum_num[num_feat]['bin_edges'][1:]]
            lookup_t_a = pd.DataFrame(columns=columns, index=index)
            lookup_w_a = pd.DataFrame(columns=columns, index=index)
            lookup_t_b = pd.DataFrame(columns=columns, index=index)
            lookup_w_b = pd.DataFrame(columns=columns, index=index)
            lookup_T_b = pd.DataFrame(columns=columns, index=index)
            lookup_t_c = pd.DataFrame(columns=columns, index=index)
            lookup_w_c = pd.DataFrame(columns=columns, index=index)
            lookup_T_c = pd.DataFrame(columns=columns, index=index)
            lookup_t_d = pd.DataFrame(columns=columns, index=index)
            lookup_w_d = pd.DataFrame(columns=columns, index=index)
            lookup_T_d = pd.DataFrame(columns=columns, index=index)
    
            # fill lookup table with sum of resuduals and number of samples for quadrant a
            for col in lookup_t_a.columns[:n_unique_values]:
                partition = set(map(int, col.strip("{}").split(", ")))
                bin = 0
                h_t = 0
                h_w = 0
                for index in lookup_t_a.index:
                    mask = X_digitized[:, num_feat] == bin
                    mask2 = torch.isin(X_cat[:, (cat_feat)], torch.tensor(list(partition)))
                    mask3 = mask & mask2
                    h_t = h_t + y_tilde[mask3].sum()
                    h_w = h_w + mask3.sum()
                    lookup_t_a.loc[index, col] = h_t
                    lookup_w_a.loc[index, col] = h_w.item()
                    bin += 1
            # for partitions with more than one category in each split group
            for col in lookup_t_a.columns[n_unique_values:]:
                cols = [f"{{{value.strip()}}}" for value in col.strip("{}").split(",")]
                lookup_t_a.loc[:, col] = lookup_t_a.loc[:, cols].sum(axis=1)
                lookup_w_a.loc[:, col] = lookup_w_a.loc[:, cols].sum(axis=1)
            # calculate interaction predictor for quadrant a
            lookup_T_a = lookup_t_a.div(lookup_w_a)
            # calculate sum of residuals, the number of samples, and the interaction predictor for quadrant b
            h_t_values = pd.Series(histo_cum_num[num_feat]['H_t'], index=lookup_t_a.index)
            h_t_x_i_frame = pd.DataFrame({col: h_t_values for col in lookup_t_a.columns})
            lookup_t_b = h_t_x_i_frame.sub(lookup_t_a, axis=0)
            h_w_values = pd.Series(histo_cum_num[num_feat]['H_w'], index=lookup_w_a.index)
            h_w_x_i_frame = pd.DataFrame({col: h_w_values for col in lookup_w_a.columns})
            lookup_w_b = h_w_x_i_frame.sub(lookup_w_a, axis=0)
            lookup_T_b = lookup_t_b.div(lookup_w_b)
            # calculate sum of residuals, the number of samples, and the interaction predictor for quadrant c
            h_t_x_j_frame = pd.DataFrame([histo_cat[cat_feat]['H_t']] * len(lookup_t_a.index), index=lookup_t_a.index, columns=lookup_t_a.columns)
            lookup_t_c = h_t_x_j_frame.sub(lookup_t_a, axis=0)
            h_w_x_j_frame = pd.DataFrame([histo_cat[cat_feat]['H_w']] * len(lookup_w_a.index), index=lookup_w_a.index, columns=lookup_w_a.columns)
            lookup_w_c = h_w_x_j_frame.sub(lookup_w_a, axis=0)
            lookup_T_c = lookup_t_c.div(lookup_w_c)
            # calculate sum of residuals, the number of samples, and the interaction predictor for quadrant d
            ch__x_d_t = histo_cum_num[num_feat]['H_t'][-1] - h_t_x_i_frame
            lookup_t_d = ch__x_d_t.sub(lookup_t_c, axis=0)
            ch__x_d_w = histo_cum_num[num_feat]['H_w'][-1] - h_w_x_i_frame
            lookup_w_d = ch__x_d_w.sub(lookup_w_c, axis=0)
            lookup_T_d = lookup_t_d.div(lookup_w_d)
            # calculate RSS for all possible splits
            term_1 = (lookup_T_a**2) * lookup_w_a + (lookup_T_b**2) * lookup_w_b + (lookup_T_c**2) * lookup_w_c + (lookup_T_d**2) * lookup_w_d
            term_2 = 2 * (lookup_T_a * lookup_t_a + lookup_T_b * lookup_t_b + lookup_T_c * lookup_t_c + lookup_T_d * lookup_t_d)
            rss = - term_1 - term_2
            # find best split for pair in this loop
            min_RSS = rss.min().min()
            RSS["RSS"][pair_idx] = min_RSS
            position = (rss == min_RSS).stack().idxmax()
            RSS["split_num"][pair_idx] = position[0]
            RSS["split_cat"][pair_idx] = position[1]

            pair_idx += 1

        # interaction pair with lowest RSS
        best_pair = RSS["pair"][RSS["RSS"].argmin()]
        best_combination = [best_pair[0]] + self.grouped_encoded_features[best_pair[1]-self.n_numerical_cols]

        return best_combination
    
    # # vei:
    # def _extract_rules_from_tree(self, tree):
    #     rules = []

    #     def _traverse(node_id=0, conditions=None):
    #         if conditions is None:
    #             conditions = []

    #         # Check if the node is a leaf
    #         if tree.children_left[node_id] == tree.children_right[node_id]:
    #             support = tree.n_node_samples[node_id] / tree.n_node_samples[0]
    #             if conditions:
    #                 rules.append({'conditions': conditions, 'support': support})
    #             return

    #         # Get the feature and threshold for the current node
    #         feature = tree.feature[node_id]
    #         threshold = tree.threshold[node_id]

    #         _traverse(tree.children_left[node_id], conditions + [(feature, '<=', threshold)])
    #         _traverse(tree.children_right[node_id], conditions + [(feature, '>', threshold)])
        
    #     _traverse()
    #     return rules

    # # vei:
    # def filter_valid_rules(self, rules):
    #     """
    #     Filters rules to only keep those that involve exactly:
    #     - One numerical feature
    #     - One categorical feature group (which can map to multiple one-hot features)
    #     """
    #     filtered_rules = []
    #     categorical_features = self.grouped_encoded_features
    #     cat_feat2group = {
    #         feat: gid
    #         for gid, group in enumerate(categorical_features)
    #         for feat in group
    #     }

    #     for rule in rules:
    #         num_feature = None
    #         cat_group = None
    #         valid = True

    #         for feat, _, _ in rule['conditions']:
    #             if feat < self.n_numerical_cols:
    #                 if num_feature is None:
    #                     num_feature = feat
    #                 else:
    #                     valid = False
    #                     break
    #             else:
    #                 g = cat_feat2group.get(feat)
    #                 if g is None:
    #                     valid = False
    #                     break
    #                 if cat_group is None:
    #                     cat_group = g
    #                 elif cat_group != g:
    #                     valid = False
    #                     break

    #         if valid and num_feature is not None and cat_group is not None:
    #             filtered_rules.append(rule)

    #     return filtered_rules

    
    # # vei:
    # def _rule_activity_matrix(self, rules, X):
    #     """
    #     Binary Matrix indicating which rules are active for each observation in X.
    #     """
    #     n, m = X.shape[0], len(rules)
    #     A = torch.ones((n, m), dtype=torch.bool, device=X.device)

    #     for i, rule in enumerate(rules):
    #         for feature, op, threshold in rule['conditions']:
    #             if op == '<=':
    #                 A[:, i] &= X[:, feature] <= threshold
    #             else:
    #                 A[:, i] &= X[:, feature] > threshold

    #     A = A.cpu().numpy()
        
    #     return A.astype(float)
    
    # # vei:
    # def _top_rulefit_interaction(
    #         self, X, y_tilde,
    #         tree_leaves=5,
    #         n_trees=100,):
    #     """
    #     This function applies the RuleFit algorithm (Friedman and Popescu, 2008) to determine the most important combination of one 
    #     categorical and one numberical feature.
    #     """
    #     # Fit ensemble of decision trees
    #     if self.task == 'regression':
    #         base_tree = DecisionTreeRegressor(random_state=self.random_state, max_leaf_nodes=tree_leaves)

    #         bagger = BaggingRegressor(
    #             estimator=base_tree,
    #             n_estimators=n_trees,
    #             max_samples=0.5,
    #             bootstrap=True,
    #             random_state=self.random_state,
    #         )

    #     else:
    #         base_tree = DecisionTreeClassifier(random_state=self.random_state, max_leaf_nodes=tree_leaves)

    #         bagger = BaggingClassifier(
    #             estimator=base_tree,
    #             n_estimators=n_trees,
    #             max_samples=0.5,
    #             bootstrap=True,
    #             random_state=self.random_state,
    #         )

    #     bagger.fit(X, y_tilde)

    #     # Extract rules from the decision trees
    #     rules = []
    #     for est in bagger.estimators_:
    #         tree_rules =  self._extract_rules_from_tree(est.tree_)
    #         rules.extend(tree_rules)

    #     filtered_rules = self.filter_valid_rules(rules)
    #     A = self._rule_activity_matrix(filtered_rules, X)
    #     supports = np.array([r["support"] for r in filtered_rules])

    #     # Create a binary matrix indicating which rules are active for each observation in X
    #     #A = self._rule_activity_matrix(rules, X)
    #     #supports = np.array([r["support"] for r in rules])

    #     scaler = StandardScaler(with_mean=False)
    #     A_std = scaler.fit_transform(A)
    #     # L1-Regression to find the most important rules
    #     if self.task == 'regression':
    #         lasso = LassoCV(cv=5, random_state=self.random_state, n_jobs=-1, max_iter=10000)
    #         lasso.fit(A_std, y_tilde)
    #         coefs = lasso.coef_
    #     else:
    #         logreg = LogisticRegressionCV(penalty="l1", solver="liblinear", Cs=np.logspace(-4, 2, 15), cv=5, scoring='neg_log_loss', n_jobs=-1, max_iter=5000, random_state=self.random_state)
    #         logreg.fit(A_std, y_tilde)
    #         coefs = logreg.coef_.ravel_

    #     # Rule Importance
    #     importances = np.abs(coefs) * np.sqrt(supports * (1 - supports))
    #     active_idx   = np.nonzero(importances)[0]  

    #     # # filter rules with only one numerical feature and one categorical feature
    #     # filtered_rules = defaultdict(float)
    #     # categorical_features = self.grouped_encoded_features
    #     # cat_feat2group = {
    #     #     feat: gid
    #     #     for gid, group in enumerate(self.grouped_encoded_features)
    #     #     for feat in group
    #     # }

    #     # for i in active_idx:
    #     #     rule = rules[i]
    #     #     imp = importances[i]
    #     #     num_feature = None
    #     #     cat_group = None
    #     #     valid = True

    #     #     # extract features from the rule
    #     #     for feat, _, _ in rule['conditions']:
    #     #         if feat < self.n_numerical_cols:
    #     #             if num_feature is None:
    #     #                 num_feature = feat
    #     #             else:
    #     #                 valid = False
    #     #                 break
    #     #         else:
    #     #             g = cat_feat2group.get(feat)
    #     #             if cat_group is None:
    #     #                 cat_group = g
    #     #             elif cat_group != g:
    #     #                 valid = False
    #     #                 break

    #     #     if valid and num_feature is not None and cat_group is not None:
    #     #         filtered_rules[(num_feature, cat_group)] += imp

    #     # if not filtered_rules:
    #     #     print('No feature combination found. Model does not capture interactions. Try different feature interaction detection method.')
    #     #     self.igann_it = False

    #     #     return None
        
    #     # final_importances = defaultdict(float)
    #     # for key, value in filtered_rules.items():
    #     #     final_importances[key] += value

    #     # best_key = max(final_importances, key=final_importances.get)
    #     # num_var, cat_var = best_key

    #     # best_combination = [num_var] + categorical_features[cat_var]

    #     # return best_combination
    #     # Aufbau: Mapping von (num_feat, cat_group) → Importance

    #     final_importances = defaultdict(float)
    #     cat_feat2group = {
    #         feat: gid
    #         for gid, group in enumerate(self.grouped_encoded_features)
    #         for feat in group
    #     }

    #     for i in active_idx:
    #         rule = filtered_rules[i]
    #         imp = importances[i]
    #         num_feature = None
    #         cat_group = None
    #         valid = True

    #         for feat, _, _ in rule['conditions']:
    #             if feat < self.n_numerical_cols:
    #                 if num_feature is None:
    #                     num_feature = feat
    #                 else:
    #                     valid = False
    #                     break
    #             else:
    #                 g = cat_feat2group.get(feat)
    #                 if cat_group is None:
    #                     cat_group = g
    #                 elif cat_group != g:
    #                     valid = False
    #                     break

    #         if valid and num_feature is not None and cat_group is not None:
    #             final_importances[(num_feature, cat_group)] += imp

    #     # Best Combination bestimmen
    #     best_key = max(final_importances, key=final_importances.get)
    #     num_var, cat_group = best_key
    #     best_combination = [num_var] + self.grouped_encoded_features[cat_group]

    #     return best_combination


    # ---------------------------------------------------------------------------
    # 1) Regeln aus Entscheidungsbaum extrahieren
    # ---------------------------------------------------------------------------
    def _extract_rules_from_tree(self, tree):
        """Tiefensuche durch den Baum und Sammeln aller Blatt-Regeln."""
        rules = []

        def _traverse(node_id=0, conditions=None):
            if conditions is None:
                conditions = []

            # Blatt?
            if tree.children_left[node_id] == tree.children_right[node_id]:
                support = tree.n_node_samples[node_id] / tree.n_node_samples[0]
                if conditions:                   # leere Regel ignorieren
                    rules.append({'conditions': conditions, 'support': support})
                return

            feature   = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            _traverse(tree.children_left[node_id],
                    conditions + [(feature, '<=', threshold)])
            _traverse(tree.children_right[node_id],
                    conditions + [(feature, '>',  threshold)])

        _traverse()
        return rules
    # ---------------------------------------------------------------------------


    # ---------------------------------------------------------------------------
    # 2) Gültige Regeln (1 num + 1 cat-Gruppe) filtern
    # ---------------------------------------------------------------------------
    def filter_valid_rules(self, rules):
        """
        Behalte nur Regeln, die genau
        • ein numerisches Feature und
        • eine (ggf. mehrere one-hot-Spalten umfassende) kategoriale Gruppe
        enthalten.
        """
        filtered = []
        cat_feat2group = {
            feat: gid
            for gid, group in enumerate(self.grouped_encoded_features)
            for feat in group
        }

        for rule in rules:
            num_feature = None
            cat_group   = None
            valid       = True

            for feat, _, _ in rule['conditions']:
                if feat < self.n_numerical_cols:          # numerisch
                    if num_feature is None:
                        num_feature = feat
                    else:                                 # zwei numerische ⇒ raus
                        valid = False
                        break
                else:                                     # kategoriale OHE-Spalte
                    g = cat_feat2group.get(feat)
                    if g is None:
                        valid = False
                        break
                    if cat_group is None:
                        cat_group = g
                    elif cat_group != g:                  # zwei Gruppen ⇒ raus
                        valid = False
                        break

            if valid and num_feature is not None and cat_group is not None:
                filtered.append(rule)

        return filtered
    # ---------------------------------------------------------------------------


    # ---------------------------------------------------------------------------
    # 3) Aktivitätsmatrix
    # ---------------------------------------------------------------------------
    def _rule_activity_matrix(self, rules, X):
        """
        Gibt eine (n × m)-Matrix zurück (float), in der A[i,j]=1, falls
        Regel j für Beobachtung i aktiv ist. Bei leerer Regelliste wird
        eine Matrix mit 0 Spalten zurückgegeben.
        """
        n = X.shape[0]
        m = len(rules)
        if m == 0:
            return np.empty((n, 0), dtype=float)

        A = torch.ones((n, m), dtype=torch.bool, device=X.device)
        for j, rule in enumerate(rules):
            for feature, op, thresh in rule['conditions']:
                if op == '<=':
                    A[:, j] &= X[:, feature] <= thresh
                else:
                    A[:, j] &= X[:, feature] >  thresh
        return A.cpu().numpy().astype(float)
    # ---------------------------------------------------------------------------


    # ---------------------------------------------------------------------------
    # 4) Wichtigste 1×1-Interaktion via RuleFit-Importances
    # ---------------------------------------------------------------------------
    def _top_rulefit_interaction(
            self, X, y_tilde,
            tree_leaves=5,
            n_trees=100):
        """
        Liefert die feature-Indices der wichtigsten Interaktion
        [num_feature] + list(categorical_ohe_indices)
        oder None, falls keine gültige Regel vorhanden ist.
        """

        # --------------------------------------------------
        # 4.1 Ensemble trainieren
        # --------------------------------------------------
        #if self.task == 'regression':
        base_tree = DecisionTreeRegressor(
            random_state=self.random_state,
            max_leaf_nodes=tree_leaves)
        bagger = BaggingRegressor(
            estimator=base_tree,
            n_estimators=n_trees,
            max_samples=0.5,
            bootstrap=True,
            random_state=self.random_state)
        # else:
        #     base_tree = DecisionTreeClassifier(
        #         random_state=self.random_state,
        #         max_leaf_nodes=tree_leaves)
        #     bagger = BaggingClassifier(
        #         estimator=base_tree,
        #         n_estimators=n_trees,
        #         max_samples=0.5,
        #         bootstrap=True,
        #         random_state=self.random_state)

        bagger.fit(X, y_tilde)

        # --------------------------------------------------
        # 4.2 Regeln extrahieren & filtern
        # --------------------------------------------------
        rules = []
        for est in bagger.estimators_:
            rules.extend(self._extract_rules_from_tree(est.tree_))

        filtered_rules = self.filter_valid_rules(rules)
        if len(filtered_rules) == 0:                 # ← kein Kandidat
            self.igann_it = False
            return None

        # --------------------------------------------------
        # 4.3 Aktivitätsmatrix & Support
        # --------------------------------------------------
        A         = self._rule_activity_matrix(filtered_rules, X)
        supports  = np.array([r['support'] for r in filtered_rules])

        # --------------------------------------------------
        # 4.4 L1-Modell (Lasso / L1-LogReg)
        # --------------------------------------------------
        scaler = StandardScaler(with_mean=False)
        A_std  = scaler.fit_transform(A)

        if A_std.shape[1] == 0:                     # sollte hier nie passieren,
            self.igann_it = False                   # da len(filtered_rules) > 0,
            return None                             # aber zur Sicherheit.

        #if self.task == 'regression':
        lasso = LassoCV(cv=5,
                        random_state=self.random_state,
                        n_jobs=-1,
                        max_iter=10_000)
        lasso.fit(A_std, y_tilde)
        coefs = lasso.coef_

        # else:
        #     logreg = LogisticRegressionCV(
        #         penalty='l1',
        #         solver='liblinear',
        #         Cs=np.logspace(-4, 2, 15),
        #         cv=5,
        #         scoring='neg_log_loss',
        #         n_jobs=-1,
        #         max_iter=5_000,
        #         random_state=self.random_state)
        #     logreg.fit(A_std, y_tilde)
        #     coefs = logreg.coef_.ravel_

        # --------------------------------------------------
        # 4.5 Rule-Importances & Mapping auf (num, cat_group)
        # --------------------------------------------------
        importances = np.abs(coefs) * np.sqrt(supports * (1 - supports))
        active_idx  = np.nonzero(importances)[0]

        final_imps  = defaultdict(float)
        cat_feat2group = {
            feat: gid
            for gid, grp in enumerate(self.grouped_encoded_features)
            for feat in grp
        }

        for i in active_idx:
            rule = filtered_rules[i]
            imp  = importances[i]

            num_feature = None
            cat_group   = None
            valid       = True

            for feat, _, _ in rule['conditions']:
                if feat < self.n_numerical_cols:
                    if num_feature is None:
                        num_feature = feat
                    else:
                        valid = False
                        break
                else:
                    g = cat_feat2group.get(feat)
                    if cat_group is None:
                        cat_group = g
                    elif cat_group != g:
                        valid = False
                        break

            if valid and num_feature is not None and cat_group is not None:
                final_imps[(num_feature, cat_group)] += imp

        # --------------------------------------------------
        # 4.6 Beste Interaktion bestimmen
        # --------------------------------------------------
        if len(final_imps) == 0:
            print('No feature combination found. Model does not capture interactions. Try different feature interaction detection method.')
            self.igann_it = False
            return None

        best_key        = max(final_imps, key=final_imps.get)
        num_var, cat_g  = best_key
        best_comb       = [num_var] + self.grouped_encoded_features[cat_g]

        return best_comb

    
    # vei:
    def _find_interactions(self, X, y_tilde, method="dt"):
        """
        This function finds the most important interaction pair between a numerical and a categorical feature.
        X: the feature matrix for the training data
        y: the residuals of the trained univariate model
        method: the method to find the most important interaction pair. 
                - 'dt' for decision tree
                - 'FAST' for FAST algorithm.
        """
        if method == "dt":
            best_pair = self._constraint_dt(X, y_tilde)
        elif method == "FAST":
            best_pair = self._FAST_detection(X, y_tilde, n_bins=20)
        elif method == "rulefit":
            best_pair = self._top_rulefit_interaction(X, y_tilde)
        else:
            warnings.warn("Method not implemented. Use 'dt' or 'FAST' or 'rulefit.")

        return best_pair

    def _select_features(self, X, y):
        regressor = ELM_Regressor(
            X.shape[1],
            self.n_categorical_cols,
            self.n_hid,
            seed=0,
            elm_scale=self.elm_scale,
            act=self.act,
            device="cpu",
        )
        X_tilde = regressor.get_hidden_values(X)
        groups = self._flatten(
            [list(np.ones(self.n_hid) * i + 1) for i in range(self.n_numerical_cols)]
        )
        groups.extend(list(range(self.n_numerical_cols, X.shape[1])))

        if self.task == "classification":
            m = abess.linear.LogisticRegression(
                path_type="gs", cv=3, s_min=1, s_max=self.sparse, thread=0
            )
            m.fit(X_tilde.numpy(), np.where(y.numpy() == -1, 0, 1), group=groups)
        else:
            m = abess.linear.LinearRegression(
                path_type="gs", cv=3, s_min=1, s_max=self.sparse, thread=0
            )
            m.fit(X_tilde.numpy(), y, group=groups)

        active_num_features = np.where(
            np.sum(
                m.coef_[: self.n_numerical_cols * self.n_hid].reshape(-1, self.n_hid),
                axis=1,
            )
            != 0
        )[0]

        active_cat_features = (
            np.where(m.coef_[self.n_numerical_cols * self.n_hid :] != 0)[0]
            + self.n_numerical_cols
        )

        self.n_numerical_cols = len(active_num_features)
        self.n_categorical_cols = len(active_cat_features)

        active_features = list(active_num_features) + list(active_cat_features)

        if self.verbose > 0:
            print(f"Found features {active_features}")

        return active_features

    def _print_results(
        self, counter, counter_no_progress, eval, boost_rate, train_loss, val_loss
    ):
        """
        This function plots our results.
        """
        if counter_no_progress == 0:
            new_best_symb = "*"
        else:
            new_best_symb = ""
        if eval:
            test_pred = self.predict_raw(eval[0])
            test_loss = self.criterion(test_pred, eval[1])
            self.test_losses.append(test_loss)
            print(
                "{}{}: BoostRate: {:.3f}, Train loss: {:.5f} Val loss: {:.5f} Test loss: {:.5f}".format(
                    new_best_symb, counter, boost_rate, train_loss, val_loss, test_loss
                )
            )
        else:
            print(
                "{}{}: BoostRate: {:.3f}, Train loss: {:.5f} Val loss: {:.5f}".format(
                    new_best_symb, counter, boost_rate, train_loss, val_loss
                )
            )

    def _optimize_classification_threshold(self, X_train, y_train):
        """
        This function optimizes the classification threshold for the training set for later predictions.
        The use of the function is triggered by setting the parameter optimize_threshold to True.
        This is one method which does the job. However, we noticed that it is not always the best method and hence it
        defaults to no threshold optimization.
        """

        y_proba = self.predict_raw(X_train)

        # detach and numpy
        y_proba = y_proba.detach().cpu().numpy()
        y_train = y_train.detach().cpu().numpy()
        fpr, tpr, trs = roc_curve(y_train, y_proba)

        roc_scores = []
        thresholds = []
        for thres in trs:
            thresholds.append(thres)
            y_pred = np.where(y_proba > thres, 1, -1)
            # Apply desired utility function to y_preds, for example accuracy.
            roc_scores.append(roc_auc_score(y_train.squeeze(), y_pred.squeeze()))
        # convert roc_scores to numpy array
        roc_scores = np.array(roc_scores)
        # get the index of the best threshold
        ix = np.argmax(roc_scores)
        # get the best threshold
        return thresholds[ix]

    def predict_proba(self, X):
        """
        Similarly to sklearn, this function returns a matrix of the same length as X and two columns.
        The first column denotes the probability of class -1, and the second column denotes the
        probability of class 1.
        """
        if self.task == "regression":
            warnings.warn(
                "The call of predict_proba for a regression task was probably incorrect."
            )

        pred = self.predict_raw(X)
        pred = self._clip_p_numpy(pred)
        pred = 1 / (1 + np.exp(-pred))

        ret = np.zeros((len(X), 2), dtype=np.float32)
        ret[:, 1] = pred
        ret[:, 0] = 1 - pred

        return ret

    def predict(self, X):
        """
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the binary target values in a 1-d np.array, it can hold -1 and 1.
        If optimize_threshold is True for a classification task, the threshold is optimized on the training data.
        """
        if self.task == "regression":
            return self.predict_raw(X)
        else:
            pred_raw = self.predict_raw(X)
            # detach and numpy pred_raw
            if self.optimize_threshold:
                threshold = self.best_threshold
            else:
                threshold = 0
            pred = np.where(
                pred_raw < threshold,
                np.ones_like(pred_raw) * -1,
                np.ones_like(pred_raw),
            ).squeeze()

            if self.target_remapped_flag:
                pred = np.where(pred == -1, 0, 1)

            return pred

    def predict_raw(self, X):
        """
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the raw logit values.
        """
        X = self._preprocess_feature_matrix(X, fit_dummies=False).to(self.device)
        X = X[:, self.feature_indizes]

        pred_nn = torch.zeros(len(X), dtype=torch.float32).to(self.device)
        for boost_rate, regressor in zip(self.boosting_rates, self.regressors):
            pred_nn += boost_rate * regressor.predict(X).squeeze()
        # vei:
        if self.igann_it == True:
            for boost_rate, regressor in zip(self.boosting_rates_it, self.regressors_it):
                pred_nn += boost_rate * regressor.predict(X[:, self.best_combination]).squeeze()

        pred_nn = pred_nn.detach().cpu().numpy()
        X = X.detach().cpu().numpy()
        pred = (
            pred_nn
            + (self.init_classifier.coef_.astype(np.float32) @ X.transpose()).squeeze()
            + self.init_classifier.intercept_
        )

        return pred

    def _flatten(self, l):
        return [item for sublist in l for item in sublist]

    def _split_long_titles(self, l):
        return "\n".join(l[p : p + 22] for p in range(0, len(l), 22))

    def _get_pred_of_i(self, i, x_values=None):
        if x_values == None:
            feat_values = self.unique[i]
        else:
            feat_values = x_values[i]
        if self.task == "classification":
            pred = self.init_classifier.coef_[0, i] * feat_values
        else:
            pred = self.init_classifier.coef_[i] * feat_values
        feat_values = feat_values.to(self.device)
        for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
            pred += (
                boost_rate
                * regressor.predict_single(feat_values.reshape(-1, 1), i).squeeze()
            ).cpu()
        return feat_values, pred
    
    # vei:
    def _get_pred_it(self):
        feat_values = self.unique_it

        # create synthetic X values for the interaction pair to predict and plot the interaction effect
        first_col_min = feat_values[:, 0].min().item()
        first_col_max = feat_values[:, 0].max().item()
        num_feat_values = torch.linspace(first_col_min, first_col_max, steps=300) # 300 values between min and max of the numerical feature
        combinations = torch.cat([torch.zeros(1, len(self.best_combination)-1), torch.eye(len(self.best_combination)-1)], dim=0) # matrix with all values of the categorical feature
        first_col_repeated = num_feat_values.repeat_interleave(len(self.best_combination)).unsqueeze(1) # repeat each value of num feature for each value of the cat feature
        combinations_repeated = combinations.repeat(len(num_feat_values), 1) # repeat the cat feature values for each num feature value
        feat_values = torch.cat([first_col_repeated, combinations_repeated], dim=1) # combine the num feature values with the cat feature values

        feat_values = feat_values.to(self.device)
        pred_it = torch.zeros(len(feat_values), dtype=torch.float32).to(self.device)

        for regressor, boost_rate in zip(self.regressors_it, self.boosting_rates_it):
            pred_it += (
                boost_rate
                * regressor.predict(feat_values).squeeze() # hier ist eventuell ein Fehler
            ).cpu()

        pred = pred_it.clone()

        if self.task == "classification":
            # to - do
            print('classification not implemented yet')
        else:
            pred += (torch.from_numpy(self.init_classifier.coef_[self.best_combination].astype(np.float32)) @ feat_values.transpose(0,1))
        
        for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
            pred += (boost_rate * regressor.predict_it(feat_values, self.best_combination).squeeze()).cpu()

        return feat_values, pred_it, pred

    def _compress_shape_functions_dict(self, shape_functions):
        shape_functions_compressed = {}
        for sf in shape_functions:
            if sf["name"] in shape_functions_compressed.keys():
                shape_functions_compressed[sf["name"]]["x"].extend(sf["x"])
                shape_functions_compressed[sf["name"]]["y"].extend(sf["y"])
                shape_functions_compressed[sf["name"]]["avg_effect"] += sf["avg_effect"]
                shape_functions_compressed[sf["name"]]["hist"][0].append(sf["hist"][0])
                shape_functions_compressed[sf["name"]]["hist"][1].append(
                    shape_functions_compressed[sf["name"]]["hist"][1][-1] + 1
                )
            else:
                shape_functions_compressed[sf["name"]] = deepcopy(sf)

        return [v for k, v in shape_functions_compressed.items()]

    def get_shape_functions_as_dict(self, x_values=None):
        shape_functions = []
        for i, feat_name in enumerate(self.feature_names):
            datatype = "numerical" if i < self.n_numerical_cols else "categorical"
            feat_values, pred = self._get_pred_of_i(i, x_values)
            if datatype == "numerical":
                shape_functions.append(
                    {
                        "name": feat_name,
                        "datatype": datatype,
                        "x": feat_values.cpu().numpy(),
                        "y": pred.numpy(),
                        "avg_effect": float(torch.mean(torch.abs(pred))),
                        "hist": self.hist[i],
                    }
                )
            else:
                shape_functions.append(
                    {
                        "name": self.dummy_encodings[feat_name],
                        "datatype": datatype,
                        "x": ["".join(feat_name.split("_")[1:])],
                        "y": [pred.numpy()[1]],
                        "avg_effect": float(torch.mean(torch.abs(pred))),
                        "hist": [[self.hist[i][0][-1]], [0]],
                    }
                )

        if sum(1 for elem in shape_functions if elem["datatype"] == "categorical") > 0:
            if shape_functions[0]["datatype"] == "numerical":
                len_of_num_hist = np.sum(np.array(shape_functions[0]["hist"][0]))
                for key, val in self.get_dummies.cols_per_feat.items():
                    get_avg = []
                    len_of_other_hists = 0
                    for sf in shape_functions:
                        if key == sf["name"]:
                            get_avg.append(sf["avg_effect"])
                            len_of_other_hists += sf["hist"][0][0].item()
                    shape_functions.append(
                        {
                            "name": key,
                            "datatype": "categorical",
                            "x": ["".join(val[1].split("_")[1:])],
                            "y": [0],  # constant zero value
                            "avg_effect": float(np.mean(get_avg)),  # maybe change to 0?
                            "hist": [
                                [torch.tensor(len_of_num_hist - len_of_other_hists)],
                                [0],
                            ],
                        }
                    )

        overall_effect = np.sum([d["avg_effect"] for d in shape_functions])
        for d in shape_functions:
            if overall_effect != 0:
                d["avg_effect"] = d["avg_effect"] / overall_effect * 100
            else:
                d["avg_effect"] = 0
        shape_functions = self._compress_shape_functions_dict(shape_functions)

        return shape_functions
    
    # vei:
    def get_it_shape_functions_as_dict(self, scaler, y_mean, y_std):
        # predict for each unique combination of the best interaction pair. unique combinations are stored in self.unique_it and get return by self._get_pred_it()
        feat_values, pred_it, pred = self._get_pred_it()

        # scale values back to original values
        x_mean = scaler.mean_[self.best_combination[0]]
        x_std = scaler.scale_[self.best_combination[0]]
        feat_values_unsc = feat_values.clone() # cloned to avoid changing the original tensor
        feat_values_unsc[:, 0] = feat_values_unsc[:, 0] * x_std + x_mean
        pred_it_unsc = pred_it * y_std + y_mean # cloned to avoid changing the original tensor
        pred_unsc = pred * y_std + y_mean # cloned to avoid changing the original tensor

        x_values = {f"category {i}": [] for i in range(1, feat_values_unsc.shape[1]+1)}
        y_values = {f"category {i}": [] for i in range(1, feat_values_unsc.shape[1]+1)}
        y_incl_univ = {f"category {i}": [] for i in range(1, feat_values_unsc.shape[1]+1)}

        # Iterate through rows and assign values to the corresponding column list
        for row_idx, row in enumerate(feat_values_unsc):
            if torch.all(row[1:] == 0):
                x_values["category 1"].append(row[0].item())
                y_values["category 1"].append(pred_it_unsc[row_idx].item())
                y_incl_univ["category 1"].append(pred_unsc[row_idx].item())
            else:
                for col_index in range(1, feat_values_unsc.shape[1]):  # Start checking from column 2
                    if row[col_index] == 1:
                        x_values[f"category {col_index + 1}"].append(row[0].item())
                        y_values[f"category {col_index + 1}"].append(pred_it_unsc[row_idx].item())
                        y_incl_univ[f"category {col_index + 1}"].append(pred_unsc[row_idx].item())
                        break  # Stop after assigning the row to one column

        shape_functions = [
            {"name": col, "x": x_values[col], "y": y_values[col], "y_incl_univ": y_incl_univ[col]}
            for col in x_values.keys() if x_values[col]
        ]
        return shape_functions

    def plot_single(
        self, plot_by_list=None, show_n=5, scaler_dict=None, max_cat_plotted=4
    ):
        """
        This function plots the most important shape functions.
        Parameters:
        show_n: the number of shape functions that should be plotted.
        scaler_dict: dictionary that maps every numerical feature to the respective (sklearn) scaler.
                     scaler_dict[num_feature_name].inverse_transform(...) is called if scaler_dict is not None
        """
        shape_functions = self.get_shape_functions_as_dict()
        if plot_by_list is None:
            top_k = [
                d
                for d in sorted(
                    shape_functions, reverse=True, key=lambda x: x["avg_effect"]
                )
            ][:show_n]
            show_n = min(show_n, len(top_k))
        else:
            top_k = [
                d
                for d in sorted(
                    shape_functions, reverse=True, key=lambda x: x["avg_effect"]
                )
            ]
            show_n = len(plot_by_list)

        plt.close(fig="Shape functions")
        fig, axs = plt.subplots(
            2,
            show_n,
            figsize=(14, 4),
            gridspec_kw={"height_ratios": [5, 1]},
            num="Shape functions",
        )
        plt.subplots_adjust(wspace=0.4)

        i = 0
        for d in top_k:
            if plot_by_list is not None and d["name"] not in plot_by_list:
                continue
            if scaler_dict:
                d["x"] = (
                    scaler_dict[d["name"]]
                    .inverse_transform(d["x"].reshape(-1, 1))
                    .squeeze()
                )
            if d["datatype"] == "categorical":
                if show_n == 1:
                    d["y"] = np.array(d["y"])
                    d["x"] = np.array(d["x"])
                    hist_items = [d["hist"][0][0].item()]
                    hist_items.extend(his[0].item() for his in d["hist"][0][1:])

                    idxs_to_plot = np.argpartition(
                        np.abs(d["y"]),
                        -(len(d["y"]) - 1)
                        if len(d["y"]) <= (max_cat_plotted - 1)
                        else -(max_cat_plotted - 1),
                    )[-(max_cat_plotted - 1) :]
                    y_to_plot = d["y"][idxs_to_plot]
                    x_to_plot = d["x"][idxs_to_plot].tolist()
                    hist_items_to_plot = [hist_items[i] for i in idxs_to_plot]
                    if len(d["x"]) > max_cat_plotted - 1:
                        # other classes:
                        if "others" in x_to_plot:
                            x_to_plot.append(
                                "others_" + str(np.random.randint(0, 999))
                            )  # others or else seem like plausible variable names
                        else:
                            x_to_plot.append("others")
                        y_to_plot = np.append(y_to_plot.flatten(), [[0]]).reshape(
                            max_cat_plotted,
                        )
                        hist_items_to_plot.append(
                            np.sum(
                                [
                                    hist_items[i]
                                    for i in range(len(hist_items))
                                    if i not in idxs_to_plot
                                ]
                            )
                        )

                    axs[0].bar(
                        x=x_to_plot, height=y_to_plot, width=0.5, color="darkblue"
                    )
                    axs[1].bar(
                        x=x_to_plot,
                        height=hist_items_to_plot,
                        width=1,
                        color="darkblue",
                    )

                    axs[0].set_title(
                        "{}:\n{:.2f}%".format(
                            self._split_long_titles(d["name"]), d["avg_effect"]
                        )
                    )
                    axs[0].grid()
                else:
                    d["y"] = np.array(d["y"])
                    d["x"] = np.array(d["x"])
                    hist_items = [d["hist"][0][0].item()]
                    hist_items.extend(his[0].item() for his in d["hist"][0][1:])

                    idxs_to_plot = np.argpartition(
                        np.abs(d["y"]),
                        -(len(d["y"]) - 1)
                        if len(d["y"]) <= (max_cat_plotted - 1)
                        else -(max_cat_plotted - 1),
                    )[-(max_cat_plotted - 1) :]
                    y_to_plot = d["y"][idxs_to_plot]
                    x_to_plot = d["x"][idxs_to_plot].tolist()
                    hist_items_to_plot = [hist_items[i] for i in idxs_to_plot]
                    if len(d["x"]) > max_cat_plotted - 1:
                        # other classes:
                        if "others" in x_to_plot:
                            x_to_plot.append(
                                "others_" + str(np.random.randint(0, 999))
                            )  # others or else seem like plausible variable names
                        else:
                            x_to_plot.append("others")
                        y_to_plot = np.append(y_to_plot.flatten(), [[0]]).reshape(
                            max_cat_plotted,
                        )
                        hist_items_to_plot.append(
                            np.sum(
                                [
                                    hist_items[i]
                                    for i in range(len(hist_items))
                                    if i not in idxs_to_plot
                                ]
                            )
                        )

                    axs[0][i].bar(
                        x=x_to_plot, height=y_to_plot, width=0.5, color="darkblue"
                    )
                    axs[1][i].bar(
                        x=x_to_plot,
                        height=hist_items_to_plot,
                        width=1,
                        color="darkblue",
                    )

                    axs[0][i].set_title(
                        "{}:\n{:.2f}%".format(
                            self._split_long_titles(d["name"]), d["avg_effect"]
                        )
                    )
                    axs[0][i].grid()

            else:
                if show_n == 1:
                    g = sns.lineplot(
                        x=d["x"], y=d["y"], ax=axs[0], linewidth=2, color="darkblue"
                    )
                    g.axhline(y=0, color="grey", linestyle="--")
                    axs[1].bar(
                        d["hist"][1][:-1], d["hist"][0], width=1, color="darkblue"
                    )
                    axs[0].set_title(
                        "{}:\n{:.2f}%".format(
                            self._split_long_titles(d["name"]), d["avg_effect"]
                        )
                    )
                    axs[0].grid()
                else:
                    g = sns.lineplot(
                        x=d["x"], y=d["y"], ax=axs[0][i], linewidth=2, color="darkblue"
                    )
                    g.axhline(y=0, color="grey", linestyle="--")
                    axs[1][i].bar(
                        d["hist"][1][:-1], d["hist"][0], width=1, color="darkblue"
                    )
                    axs[0][i].set_title(
                        "{}:\n{:.2f}%".format(
                            self._split_long_titles(d["name"]), d["avg_effect"]
                        )
                    )
                    axs[0][i].grid()

            i += 1

        if show_n == 1:
            axs[1].get_xaxis().set_visible(False)
            axs[1].get_yaxis().set_visible(False)
        else:
            for i in range(show_n):
                axs[1][i].get_xaxis().set_visible(False)
                axs[1][i].get_yaxis().set_visible(False)
        plt.show()

    # vei:
    def plot_it(self, scaler, y_mean, y_std, include_univariate=False):
        if include_univariate:
            y_values = "y_incl_univ"
        else:
            y_values = "y"

        shape_functions = self.get_it_shape_functions_as_dict(scaler, y_mean, y_std)
        plt.close(fig="Shape functions")

        # Seaborn-Theme nur für diesen Plot aktivieren
        with sns.axes_style("whitegrid",
                            {"grid.color": "#f0f0f0",
                             "axes.edgecolor": "0.3",
                             "axes.linewidth": 1}
                             ):
            
            sns.set_context("notebook", font_scale=1)
            
            #colors = sns.color_palette("rocket_r", n_colors=len(shape_functions))
            colors = ["#27AE60", "#7D3C98", "#E74C3C", "#2E86C1", "#F39C12"]

            fig, axs = plt.subplots(1, 1, figsize=(8, 7), num="Shape functions", constrained_layout=False, dpi=100) # dpi für Masterarbeit dann auf 2 bis 300 erhöhen
            
            fig.patch.set_alpha(0)  # Transparenter Hintergrund für Figur


            for i, d in enumerate(shape_functions):
                color = colors[i % len(colors)]
                if i == 0:
                    feature_name = self.dropped_categories[self.feature_names[self.best_combination[1]].split("_")[0]]
                else:
                    feature_name = self.feature_names[self.best_combination[i]].split("_")[1]

                axs.plot(d["x"], d[y_values], linewidth=2.5, color=color, alpha=0.8, 
                        label=f"{self.feature_names[self.best_combination[-1]].split('_')[0]}: {feature_name}")

                axs.axhline(y=0, color="#404040", linestyle="-", linewidth=0.8, alpha=0.6)

            #axs.set_xlabel(self.feature_names[self.best_combination[0]], fontsize=14, labelpad=10, fontweight='semibold')
            axs.set_xlabel("Mileage (km)", fontsize=16, labelpad=10, fontweight='semibold')
            axs.set_ylabel("Contribution to Prediction of Claim Cost           ", fontsize=16, labelpad=10, fontweight='semibold')
            axs.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}€"))
            axs.yaxis.grid(True, linestyle='-', alpha=0.8, color="gray")
            axs.xaxis.grid(True, linestyle='--', alpha=0.8, color="gray")
            if include_univariate:
                axs.yaxis.set_major_locator(MultipleLocator(100))
            else:
                axs.yaxis.set_major_locator(MultipleLocator(50))

            axs.xaxis.set_major_locator(MultipleLocator(25000))

            axs.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            #axs.set_title(f"Interaction Effect: Mileage x Car Type", fontsize=16, fontweight="bold", pad=15)

            axs.tick_params(axis='x', labelrotation=45, labelsize=16)
            axs.tick_params(axis='y', labelsize=16)
            axs.grid(visible=True, which='major', axis='both', linestyle='--', alpha=0.6)
            #axs.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=12, frameon=True, shadow=False, fancybox=True, borderpad=1)
            axs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=16, frameon=True)

            axs.spines["bottom"].set_color("#404040")  # Untere Linie dunkelgrau
            axs.spines["left"].set_color("#404040")    # Linke Linie dunkelgrau
            axs.spines["bottom"].set_linewidth(0.8)  # Untere Linie dicker machen
            axs.spines["left"].set_linewidth(0.8)    # Linke Linie dicker machen
            sns.despine(left=False, bottom=False)
            plt.tight_layout(rect=[0, 0.1, 1, 1])
            #plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Platz für Legende lassen
            #plt.subplots_adjust(right=0.8)  # Rechts mehr Platz für Legende
            #plt.gca().set_facecolor('none')  # Hintergrund des Achsenbereichs transparent machen
            plt.subplots_adjust(bottom=0.4)
            plt.show()


    # vei: interaktiver Plot
    # def plot_it(self):
    #     shape_functions = self.get_it_shape_functions_as_dict()

    #     colors = px.colors.sequential.Viridis[::-1]  # Ähnliche Farbpalette wie "rocket_r"
        
    #     fig = go.Figure()

    #     # Erstellen der Daten für beide Y-Werte
    #     traces_y_incl_univ = []
    #     traces_y = []

    #     for i, d in enumerate(shape_functions):
    #         color = colors[i % len(colors)]
            
    #         if i == 0:
    #             feature_name = self.dropped_categories[self.feature_names[self.best_combination[1]].split("_")[0]]
    #         else:
    #             feature_name = self.feature_names[self.best_combination[i]].split("_")[1]

    #         # Linien für y_incl_univ
    #         traces_y_incl_univ.append(go.Scatter(
    #             x=d["x"],
    #             y=d["y_incl_univ"],
    #             mode="lines",
    #             line=dict(color=color, width=2.5),
    #             name=f"{self.feature_names[self.best_combination[-1]].split('_')[0]}: {feature_name}",
    #             visible=True  # Standardmäßig sichtbar
    #         ))

    #         # Linien für y (die alternative Option)
    #         traces_y.append(go.Scatter(
    #             x=d["x"],
    #             y=d["y"],
    #             mode="lines",
    #             line=dict(color=color, width=2.5, dash="dot"),  # Gepunktete Linien als Unterschied
    #             name=f"{self.feature_names[self.best_combination[-1]].split('_')[0]}: {feature_name}",
    #             visible=False  # Standardmäßig unsichtbar
    #         ))

    #     # Beide Trace-Gruppen zur Figur hinzufügen
    #     for trace in traces_y_incl_univ:
    #         fig.add_trace(trace)
    #     for trace in traces_y:
    #         fig.add_trace(trace)

    #     # Layout & Styling
    #     fig.update_layout(
    #         title_text=f"Feature Interaction: Contribution of {self.feature_names[self.best_combination[0]]} and "
    #                 f"{self.feature_names[self.best_combination[-1]].split('_')[0]} to the Prediction",
    #         title_font=dict(size=16, family="Arial", color="black"),
    #         xaxis=dict(title=self.feature_names[self.best_combination[0]], tickangle=45, tickfont=dict(size=12)),
    #         yaxis=dict(title="Contribution to Prediction $\\hat{y}$", title_font=dict(size=14)),
    #         plot_bgcolor="#f5f5f5",
    #         legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.5)", bordercolor="black", borderwidth=1),
    #         margin=dict(r=200, t=80, b=50, l=60),  # Mehr Platz für die Legende
    #         updatemenus=[
    #             {
    #                 "buttons": [
    #                     {
    #                         "label": "y_incl_univ",
    #                         "method": "update",
    #                         "args": [{"visible": [True] * len(traces_y_incl_univ) + [False] * len(traces_y)}]
    #                     },
    #                     {
    #                         "label": "y",
    #                         "method": "update",
    #                         "args": [{"visible": [False] * len(traces_y_incl_univ) + [True] * len(traces_y)}]
    #                     }
    #                 ],
    #                 "direction": "down",
    #                 "showactive": True,
    #                 "x": 0.17,
    #                 "xanchor": "left",
    #                 "y": 1.15,
    #                 "yanchor": "top"
    #             }
    #         ]
    #     )

    #     fig.show()


    def plot_learning(self):
        """
        Plot the training and the validation losses over time (i.e., for the sequence of learned
        ELMs)
        """
        fig, axs = plt.subplots(1, 1, figsize=(16, 8))
        fig.axes[0].plot(
            np.arange(len(self.train_losses)), self.train_losses, label="Train"
        )
        fig.axes[0].plot(np.arange(len(self.val_losses)), self.val_losses, label="Val")
        if len(self.test_losses) > 0:
            fig.axes[0].plot(
                np.arange(len(self.test_losses)), self.test_losses, label="Test"
            )
        plt.legend()
        plt.show()

    def _get_feature_importance(self, first_call=False):
        if first_call:
            self.feature_importances_ = self._get_feature_importance(first_call=False)
            # should not slow down fit if feature importance is never used
        else:
            shape_functions = self.get_shape_functions_as_dict()
            self.feature_importances_ = np.zeros(shape=(len(shape_functions),))
            for i, sf in enumerate(shape_functions):
                self.feature_importances_[i] = sf['avg_effect']
            total = np.sum(self.feature_importances_)
            self.feature_importances_ /= total
            self.feature_importances_ *= 100
        return self.feature_importances_


class IGANN_Bagged:
    def __init__(
        self,
        task="classification",
        n_hid=10,
        n_estimators=5000,
        boost_rate=0.1,
        init_reg=1,
        n_bags=3,
        elm_scale=1,
        elm_alpha=1,
        sparse=0,
        act="elu",
        early_stopping=50,
        device="cpu",
        random_state=1,
        optimize_threshold=False,
        verbose=0,
    ):
        2
        self.n_bags = n_bags
        self.sparse = sparse
        self.random_state = random_state
        self.bags = [
            IGANN(
                task,
                n_hid,
                n_estimators,
                boost_rate,
                init_reg,
                elm_scale,
                elm_alpha,
                sparse,
                act,
                early_stopping,
                device,
                random_state + i,
                optimize_threshold,
                verbose=verbose,
            )
            for i in range(n_bags)
        ]

    def fit(self, X, y, val_set=None, eval=None, plot_fixed_features=None):
        X.columns = [str(c) for c in X.columns]
        X = X.reindex(sorted(X.columns), axis=1)
        categorical_cols = sorted(
            X.select_dtypes(include=["category", "object"]).columns.tolist()
        )
        numerical_cols = sorted(list(set(X.columns) - set(categorical_cols)))

        if len(numerical_cols) > 0:
            X_num = torch.from_numpy(X[numerical_cols].values).float()
            self.n_numerical_cols = X_num.shape[1]
        else:
            self.n_numerical_cols = 0

        if len(categorical_cols) > 0:
            get_dummies = GetDummies(categorical_cols)
            get_dummies.fit(X[categorical_cols])
        else:
            get_dummies = None

        ctr = 0
        for b in self.bags:
            print("#")
            random_generator = np.random.Generator(
                np.random.PCG64(self.random_state + ctr)
            )
            ctr += 1
            idx = random_generator.choice(np.arange(len(X)), len(X))
            b.fit(
                X.iloc[idx], np.array(y)[idx], val_set, eval, fitted_dummies=get_dummies
            )

    def predict(self, X):
        preds = []
        for b in self.bags:
            preds.append(b.predict(X))
        return np.array(preds).mean(0), np.array(preds).std(0)

    def predict_proba(self, X):
        preds = []
        for b in self.bags:
            preds.append(b.predict_proba(X))
        return np.array(preds).mean(0), np.array(preds).std(0)

    def plot_single(
        self, plot_by_list=None, show_n=5, scaler_dict=None, max_cat_plotted=4
    ):
        x_values = dict()
        for i, feat_name in enumerate(self.bags[0].feature_names):
            curr_min = 2147483647
            curr_max = -2147483646
            most_unique = 0
            for b in self.bags:
                if b.X_min[0][i] < curr_min:
                    curr_min = b.X_min[0][i]
                if b.X_max[0][i] > curr_max:
                    curr_max = b.X_max[0][i]
                if len(b.unique[i]) > most_unique:
                    most_unique = len(b.unique[i])
            x_values[i] = torch.from_numpy(
                np.arange(curr_min, curr_max, 1.0 / most_unique)
            ).float()

        shape_functions = [
            b.get_shape_functions_as_dict(x_values=x_values) for b in self.bags
        ]

        avg_effects = {}
        for sf in shape_functions:
            for feat_d in sf:
                if feat_d["name"] in avg_effects:
                    avg_effects[feat_d["name"]].append(feat_d["avg_effect"])
                else:
                    avg_effects[feat_d["name"]] = [feat_d["avg_effect"]]

        for k, v in avg_effects.items():
            avg_effects[k] = np.mean(v)

        for sf in shape_functions:
            for feat_d in sf:
                feat_d["avg_effect"] = avg_effects[feat_d["name"]]

        if plot_by_list is None:
            top_k = [
                d
                for d in sorted(
                    shape_functions[0], reverse=True, key=lambda x: x["avg_effect"]
                )
            ][:show_n]
            show_n = min(show_n, len(top_k))
        else:
            top_k = [
                d
                for d in sorted(
                    shape_functions[0], reverse=True, key=lambda x: x["avg_effect"]
                )
            ]
            show_n = len(plot_by_list)

        plt.close(fig="Shape functions")
        fig, axs = plt.subplots(
            2,
            show_n,
            figsize=(14, 4),
            gridspec_kw={"height_ratios": [5, 1]},
            num="Shape functions",
        )
        plt.subplots_adjust(wspace=0.4)

        axs_i = 0
        for d in top_k:
            if plot_by_list is not None and d["name"] not in plot_by_list:
                continue
            if scaler_dict:
                for sf in shape_functions:
                    d["x"] = (
                        scaler_dict[d["name"]]
                        .inverse_transform(d["x"].reshape(-1, 1))
                        .squeeze()
                    )
            y_l = []
            for sf in shape_functions:
                for feat in sf:
                    if d["name"] == feat["name"]:
                        y_l.append(np.array(feat["y"]))
                    else:
                        continue
            y_mean = np.mean(y_l, axis=0)
            y_std = np.std(y_l, axis=0)
            y_mean_and_std = np.column_stack((y_mean, y_std))
            if show_n == 1:
                if d["datatype"] == "categorical":
                    hist_items = [d["hist"][0][0].item()]
                    hist_items.extend(his[0].item() for his in d["hist"][0][1:])

                    idxs_to_plot = np.argpartition(
                        np.abs(y_mean_and_std[:, 0]),
                        -(len(y_mean_and_std) - 1)
                        if len(y_mean_and_std) <= (max_cat_plotted - 1)
                        else -(max_cat_plotted - 1),
                    )[-(max_cat_plotted - 1) :]
                    d_X = [d["x"][i] for i in idxs_to_plot]
                    y_mean_and_std_to_plot = y_mean_and_std[:, :][idxs_to_plot]
                    hist_items_to_plot = [hist_items[i] for i in idxs_to_plot]
                    if len(y_mean_and_std) > max_cat_plotted - 1:
                        # other classes:
                        if "others" in d_X:
                            d_X.append(
                                "others_" + str(np.random.randint(0, 999))
                            )  # others or else seem like plausible variable names
                        else:
                            d_X.append("others")
                        y_mean_and_std_to_plot = np.append(
                            y_mean_and_std_to_plot.flatten(), [[0, 0]]
                        ).reshape(max_cat_plotted, 2)
                        hist_items_to_plot.append(
                            np.sum(
                                [
                                    hist_items[i]
                                    for i in range(len(hist_items))
                                    if i not in idxs_to_plot
                                ]
                            )
                        )
                    axs[0].bar(
                        x=d_X,
                        height=y_mean_and_std_to_plot[:, 0],
                        width=0.5,
                        color="darkblue",
                    )
                    axs[0].errorbar(
                        x=d_X,
                        y=y_mean_and_std_to_plot[:, 0],
                        yerr=y_mean_and_std_to_plot[:, 1],
                        fmt="none",
                        color="black",
                        capsize=5,
                    )
                    axs[1].bar(
                        x=d_X, height=hist_items_to_plot, width=1, color="darkblue"
                    )
                else:
                    g = sns.lineplot(
                        x=d["x"],
                        y=y_mean_and_std[:, 0],
                        ax=axs[0],
                        linewidth=2,
                        color="darkblue",
                    )
                    g = axs[0].fill_between(
                        x=d["x"],
                        y1=y_mean_and_std[:, 0] - y_mean_and_std[:, 1],
                        y2=y_mean_and_std[:, 0] + y_mean_and_std[:, 1],
                        color="aqua",
                    )
                    axs[1].bar(
                        d["hist"][1][:-1], d["hist"][0], width=1, color="darkblue"
                    )
                axs[0].axhline(y=0, color="grey", linestyle="--")
                axs[0].set_title(
                    "{}:\n{:.2f}%".format(
                        self.bags[0]._split_long_titles(d["name"]), d["avg_effect"]
                    )
                )
                axs[0].grid()
            else:
                if d["datatype"] == "categorical":
                    hist_items = [d["hist"][0][0].item()]
                    hist_items.extend(his[0].item() for his in d["hist"][0][1:])

                    idxs_to_plot = np.argpartition(
                        np.abs(y_mean_and_std[:, 0]),
                        -(len(y_mean_and_std) - 1)
                        if len(y_mean_and_std) <= (max_cat_plotted - 1)
                        else -(max_cat_plotted - 1),
                    )[-(max_cat_plotted - 1) :]
                    d_X = [d["x"][i] for i in idxs_to_plot]
                    y_mean_and_std_to_plot = y_mean_and_std[:, :][idxs_to_plot]
                    hist_items_to_plot = [hist_items[i] for i in idxs_to_plot]
                    if len(y_mean_and_std) > max_cat_plotted - 1:
                        # other classes:
                        if "others" in d_X:
                            d_X.append(
                                "others_" + str(np.random.randint(0, 999))
                            )  # others or else seem like plausible variable names
                        else:
                            d_X.append("others")
                        y_mean_and_std_to_plot = np.append(
                            y_mean_and_std_to_plot.flatten(), [[0, 0]]
                        ).reshape(max_cat_plotted, 2)
                        hist_items_to_plot.append(
                            np.sum(
                                [
                                    hist_items[i]
                                    for i in range(len(hist_items))
                                    if i not in idxs_to_plot
                                ]
                            )
                        )
                    axs[0][axs_i].bar(
                        x=d_X,
                        height=y_mean_and_std_to_plot[:, 0],
                        width=0.5,
                        color="darkblue",
                    )
                    axs[0][axs_i].errorbar(
                        x=d_X,
                        y=y_mean_and_std_to_plot[:, 0],
                        yerr=y_mean_and_std_to_plot[:, 1],
                        fmt="none",
                        color="black",
                        capsize=5,
                    )
                    axs[1][axs_i].bar(
                        x=d_X, height=hist_items_to_plot, width=1, color="darkblue"
                    )
                else:
                    g = sns.lineplot(
                        x=d["x"],
                        y=y_mean_and_std[:, 0],
                        ax=axs[0][axs_i],
                        linewidth=2,
                        color="darkblue",
                    )
                    g = axs[0][axs_i].fill_between(
                        x=d["x"],
                        y1=y_mean_and_std[:, 0] - y_mean_and_std[:, 1],
                        y2=y_mean_and_std[:, 0] + y_mean_and_std[:, 1],
                        color="aqua",
                    )
                    axs[1][axs_i].bar(
                        d["hist"][1][:-1], d["hist"][0], width=1, color="darkblue"
                    )
                axs[0][axs_i].axhline(y=0, color="grey", linestyle="--")
                axs[0][axs_i].set_title(
                    "{}:\n{:.2f}%".format(
                        self.bags[0]._split_long_titles(d["name"]), d["avg_effect"]
                    )
                )
                axs[0][axs_i].grid()
            axs_i += 1

        if show_n == 1:
            axs[1].get_xaxis().set_visible(False)
            axs[1].get_yaxis().set_visible(False)
        else:
            for i in range(show_n):
                axs[1][i].get_xaxis().set_visible(False)
                axs[1][i].get_yaxis().set_visible(False)
        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_circles, make_regression

    """
    X_small, y_small = make_circles(n_samples=(250, 500), random_state=3, noise=0.04, factor=0.3)
    X_large, y_large = make_circles(n_samples=(250, 500), random_state=3, noise=0.04, factor=0.7)

    y_small[y_small == 1] = 0

    df = pd.DataFrame(np.vstack([X_small, X_large]), columns=['x1', 'x2'])
    df['label'] = np.hstack([y_small, y_large])
    df.label = 2 * df.label - 1

    sns.scatterplot(data=df, x='x1', y='x2', hue='label')
    df['x1'] = 1 * (df.x1 > 0)

    m = IGANN(n_estimators=100, n_hid=10, elm_alpha=5, boost_rate=1, sparse=0, verbose=2)
    start = time.time()

    inputs = df[['x1', 'x2']]
    targets = df.label

    m.fit(inputs, targets)
    end = time.time()
    print(end - start)

    m.plot_learning()
    m.plot_single(show_n=7)

    m.predict(inputs)

    ######
    """
    X, y = make_regression(1000, 4, n_informative=4, random_state=42)
    X = pd.DataFrame(X)
    X["cat_test"] = np.random.choice(
        ["A", "B", "C", "D"], X.shape[0], p=[0.2, 0.2, 0.1, 0.5]
    )
    X["cat_test_2"] = np.random.choice(
        ["E", "F", "G", "H"], X.shape[0], p=[0.2, 0.2, 0.1, 0.5]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    start = time.time()
    # m = IGANN_Bagged(
    #     task="regression", n_estimators=100, verbose=0, n_bags=5
    # )  # , device='cuda'
    m = IGANN(task='regression', n_estimators=100, verbose=0)
    m.fit(pd.DataFrame(X_train), y_train)
    print(m.feature_importances_)
    end = time.time()
    print(end - start)
    m.plot_single(show_n=6, max_cat_plotted=4)

    #####
    """
    X, y = make_regression(10000, 2, n_informative=2)
    y = (y - y.mean()) / y.std()
    X = pd.DataFrame(X)
    X['categorical'] = np.random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], size=len(X))
    X['categorical2'] = np.random.choice(['q', 'b', 'c', 'd'], size=len(X), p=[0.1, 0.2, 0.5, 0.2])
    print(X.dtypes)
    m = IGANN(task='regression', n_estimators=100, sparse=0, verbose=2)
    m.fit(X, y)

    from Benchmark import FiveFoldBenchmark
    m = IGANN(n_estimators=0, n_hid=10, elm_alpha=5, boost_rate=1.0, sparse=0, verbose=2)
    #m = LogisticRegression()
    benchmark = FiveFoldBenchmark(model=m)
    folds_auroc = benchmark.run_model_on_dataset(dataset_id=1)
    print(np.mean(folds_auroc))
    """
