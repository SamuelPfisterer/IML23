# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline

def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("public/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("public/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("public/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("public/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("public/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.

        # New Code:
        self.fc1 = nn.Linear(1000, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.

        # New Code:
        x = F.relu(self.fc1(x))
        x = self.out(x)

        return x
    
def make_feature_extractor(x, y, batch_size=256, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # model declaration
    model = Net()
    model.train()
    
    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.

    # New Code:
    # weight_decays = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    weight_decays = [0.01]
    n_epochs = 200
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)
    criterion = nn.MSELoss()
    losses = []
    for weight_decay in weight_decays:
        print("\n\n\n")
        print("----------WEIGHT DECAY = " + str(weight_decay) + "----------")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
        for i in range(n_epochs):
            y_pred = model.forward(x_tr)
            #print("This is the shape of x_tr:", x_tr.shape)
            #print("This is the shape of y_pred:", y_pred.shape)
            #print("This is the shape of y_tr:", y_tr.shape)

            y_pred = y_pred.squeeze(-1)
            loss = criterion(y_pred, y_tr)
            losses.append(loss)
            # print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Validation loss: at " + str(i) + ":" + str(criterion(model.forward(x_val).squeeze(-1), y_val).item()))

    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        x = torch.tensor(x, dtype=torch.float)
        # New Code
        model.out = torch.nn.Sequential()
        x = model(x).detach().numpy()

        return x

    return make_features

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    # model = None

    # New Code:
    model = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])

    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    print(type(x_train))
    print(type(x_test))
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor = make_feature_extractor(x_pretrain, y_pretrain)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # regression model
    regression_model = get_regression_model()

    y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.

    # New Code:
    pipe = Pipeline(steps = [('extractor', PretrainedFeatureClass(feature_extractor='pretrain')), ('final', regression_model)])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    #x_train_embedding = feature_extractor(x_train)
    #x_test_embedding = feature_extractor(x_test)
    #regression_model.fit(x_test_embedding, y_train)
    #y_pred = regression_model.predict(x_test)


    assert y_pred.shape == (x_test.shape[0],)
    x_test = pd.DataFrame(x_test)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index+50100)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")