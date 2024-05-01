# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    # Original Line:
    # train_transforms = transforms.Compose([transforms.ToTensor()])

    # New Code:
    train_transforms = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    batchsize=16
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batchsize,
                              shuffle=False,
                              pin_memory=True, num_workers=4)

    # TODO: define a model for extraction of the embeddings
    # Original Line:
    # model = nn.Module()

    # New Code:
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.eval()
    model.fc = torch.nn.Sequential()

    embeddings = []
    embedding_size = 2048 
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))
    
    counter = 0
    with torch.no_grad():
        for input, _ in train_loader:
            output = model(input)
            for i in range(batchsize):
                embeddings[counter*batchsize + i] = output.numpy()[i]
            counter += 1
            print(counter)
                
    np.save('dataset/embeddings.npy', embeddings)

# Output: X, y
# X contains, for each triplet of images, the embedding of the three images
# y contains, for each triplet of images, 1
# if train==True, X also contains for each triplet of images, the embeddings with 
# images 1 and 2 inverted and y contains a corresponding 0
def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)
    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings.npy')
    # Normalize the embeddings across the dataset

    file_to_embedding = {}
    embeddings_max = embeddings.max()
    embeddings_min = embeddings.min()
    
    for i in range(len(filenames)):
        # Original Code:
        # file_to_embedding[filenames[i]] = embeddings[i]
        
        # New Code:
        file_to_embedding[filenames[i]] = (embeddings[i]-embeddings_min)/(embeddings_max-embeddings_min)

    # file_to_embedding = {'00000': [0.3, 0.6, ..., 0.19], ...}

    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        # emb = [embedding of t[0], embedding of t[1], embedding of t[2]]
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)

    return X, y

# Adjusting the batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        
        self.f1 = nn.Linear(6144, 768)
        self.f2 = nn.Linear(768, 96)
        self.out = nn.Linear(96, 1)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """

        # New Code 3:
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.out(x)
        #x = torch.sigmoid(x)

        return x

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 10
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part 
    # of the training data as a validation split. After each epoch, compute the loss on the 
    # validation split and print it out. This enables you to see how your model is performing 
    # on the validation data before submitting the results on the server. After choosing the 
    # best model, train it on the whole training data.

    # New Code:
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for epoch in range(n_epochs):
        
       
        for i, [X, y] in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model.forward(X)
            y = y.unsqueeze(1)
            y = y.to(torch.float32)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss)


    return model

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)

    # define a model and train it
    model = train_model(train_loader)
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")  

    # generate_embeddings()
    # X, y = get_data(TRAIN_TRIPLETS)
