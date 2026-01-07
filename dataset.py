# To define how your data is loaded, preprocessed, and served to the model. This is where you'll use torchaudio. You'll create a custom PyTorch Dataset class
# get rid or replace missing data
# determine what structure data should be in as input to model
# detemrine how certain features should be represented in nurmerical form (if needed)
# get rid of useless data

import torch
import torchaudio
import torchaudio.transforms as transforms # transform audio to make them digestible to pytorch model 
from torch.utils.data import Dataset

# transforming out data into a tensor and then normalizing values to have mean of 0 and std of 0.5
# most activation functions have their gradients around x = 0 so centering data there can speed learning

# we can create instances of a dataset from torchaudio.datasets

class SpeechDataset(Dataset):
    def __init__(self, data_path):
        # ... logic to find all audio files and transcripts
        pass
    def __len__(self):
        # ... return the total number of audio samples
        pass
    def __getitem__(self, idx):
        # ... load one audio file and its label
        # ... apply transformations (e.g., create a MelSpectrogram)
        # ... return the processed data and label as tensors
        pass
