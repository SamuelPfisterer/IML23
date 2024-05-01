# Solar Energy Materials Discovery through Transfer Learning

## Project Overview
This project, part of the Introduction to Machine Learning course at ETH Zurich, focuses on predicting the HOMO-LUMO gap of molecules to identify efficient materials for organic solar cells. The goal is to leverage transfer learning to enhance predictions using a larger dataset to inform models applied to a smaller, targeted dataset.

## Dataset
- **Training Dataset (`train.csv`)**: Contains HOMO-LUMO gap values for 100 molecules, which is the primary dataset for training the final model.
- **Pre-training Dataset (`pretrain.csv`)**: Comprises 50,000 molecules with LUMO energy levels, used for the pre-training phase to capture generalizable features.
- **Test Dataset (`test.csv`)**: Includes 10,000 molecules without known HOMO-LUMO gaps, used for evaluating the model's performance.

## Methodology
The approach incorporates transfer learning from a larger dataset to enhance the predictive accuracy on a smaller dataset:
1. **Data Loading**: CSV files are loaded into NumPy arrays to prepare the data for model training and evaluation.
2. **Model Training**:
   - **Pre-training Phase**: A model is initially trained on the large dataset (`pretrain.csv`) to predict LUMO energy levels. This phase focuses on capturing broad, applicable features across a wide range of molecules.
   - **Transfer Learning Phase**: Features learned during the pre-training are transferred to a new model focused on the smaller dataset (`train.csv`). This model is fine-tuned to predict the HOMO-LUMO gap, which is the key performance indicator for solar cell efficiency.
3. **Model Evaluation**: Performance is assessed using a test set (`test.csv`), where the model predicts HOMO-LUMO gaps. Evaluation metrics include Root Mean Square Error (RMSE) against values computed by density functional theory.

## Implemented Solution
Utilizing Python and libraries like Pandas, NumPy, and PyTorch, the solution includes:
- **Custom Load Function**: Efficiently imports data from provided CSV files into the working environment.
- **Machine Learning Pipeline**:
  - 
  - Pretraining a "feature" extractor based on the energy level data. This is implemted via a PyTorch neural network. 
  - 
  - "RidgeCV" Scikit-Learn is used to finetune a last "layer", that predicts the HOMO-LUMO gaps based on the embedding from the pretrained model.
