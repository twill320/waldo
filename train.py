# train.py

import torch
import torch.optim as optim

import torchaudio
import torchaudio.transforms as transforms

# 1. Import your model's blueprint and other components
from models.model import SpeechRecognitionModel
from dataset import SpeechDataset
from config import LEARNING_RATE, BATCH_SIZE, EPOCHS

# we want to loop over dataset multiple times
# when looping over dataset we get inputs, zero the parameter gradients -> foward + backward + optimize -> get loss
# need to prevent overfitting where model is simply memorizing the data so when met with new data it isn't able to properly adjust and make accurate predictions
# why we use test dataset to make sure model is at least consistent

# --- SETUP ---
# Create an instance of your model from the blueprint
# This creates a 'blank slate' model with random initial weights.
model = SpeechRecognitionModel(input_size=..., hidden_size=..., num_classes=...)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- THE LOADING LOGIC ---
# Define the path to the checkpoint you want to load
CHECKPOINT_PATH = "models/checkpoint_epoch_10.pth"

# Check if a checkpoint exists and load it
if torch.cuda.is_available(): # Or whatever your device check is
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ Checkpoint loaded. Resuming training from epoch {start_epoch}.")
else:
    start_epoch = 0
    print("ℹ️ No checkpoint found. Starting training from scratch.")


# --- THE TRAINING LOOP ---
# The loop now starts from 'start_epoch' instead of 0
for epoch in range(start_epoch, EPOCHS):
    # ... your training logic for one epoch ...
    print(f"Epoch {epoch} complete.")
    # ... logic to save the next checkpoint ...
    # The path where you want to save the model
    save_path = f'models/checkpoint_epoch_{epoch}.pth'

    print(f"Saving checkpoint to {save_path}...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,  # Assuming 'loss' is your loss variable
    }, save_path)