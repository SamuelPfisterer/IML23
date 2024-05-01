# Food Taste Similarity Prediction

## Project Overview
This project, part of an open-ended task in a machine learning course, aims to predict food taste similarity based on images and human judgments. It involves analyzing a dataset of images of 10,000 dishes and a set of triplets indicating human taste preferences.

## Dataset
- **Images**: 10,000 images of different dishes.
- **Training Triplets (`train_triplets.txt`)**: Human annotated triplets indicating which two dishes are more similar in taste to each other.
- **Test Triplets (`test_triplets.txt`)**: Triplets for which predictions of taste similarity need to be made.
- **Sample Submission (`sample.txt`)**: A template showing the format for submission.

## Methodology
The task requires predicting for each triplet (A, B, C) whether dish A is more similar in taste to dish B or C based on their images. This involves:
1. **Embeddings**: We first create embeddings for every image such that we have usable data for our models. 
2. **Modeling**: Using pretrained vision models and fine-tuning them on the given dataset with a torch neural network. 
3. **Prediction Format**: Generating predictions as '0' or '1', where '1' indicates A is closer in taste to B, and '0' indicates A is closer in taste to C.




