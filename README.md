# Ai-Ml-Code-Project

➡️ Load audio files → Convert them to mel-spectrograms → Train a CNN → Evaluate → Predict → Export CSV.

🔧 Step 0 — Setup

You define where the training and test audio are stored, and where the final CSV output will be saved.
You also set a small configuration (like how much to augment audio).

🎲 Step 1 — Import Libraries & Fix Randomness

You import all the tools you need (PyTorch, NumPy, librosa, etc.).
You also set a seed (42) so the results are reproducible and stable every run.

🎛 Step 2 — Audio Augmentation

You create a function that randomly modifies audio to make the model more robust.

The audio can be:

unchanged

made noisy

pitch shifted

speed changed

time-shifted (shifted left/right)

given a fake reverb

This effectively doubles your dataset and prevents overfitting.

🎵 Step 3 — Convert Audio to Mel-Spectrograms

You load each .wav file, trim silence, optionally augment it, make sure it's 5 seconds long, and then convert it to a mel-spectrogram (image-like representation that CNNs understand).

📦 Step 4 — Build Training & Test Feature Sets

You scan your train/ folder where each subfolder is a class.

For every audio file:

Extract mel-spectrogram.

Save feature + label.

Optionally create an augmented version too.

For test audio files, you just extract features (no labels).

🏷 Step 5 — Encode Labels & Split Data

Since the model works with numbers, not text, you convert class names into integers.

Then you split the dataset:

80% training

20% validation

📚 Step 6 — Torch Dataset/Dataloader

You wrap your spectrograms into PyTorch Datasets so they can be fed into the model in batches.

🧠 Step 7 — Build a CNN Model

You define a convolutional neural network that:

Takes a spectrogram image

Passes it through several Conv → ReLU → Pool layers

Compresses it

Sends it through fully connected layers

Outputs class scores

This is your classifier.

⚙️ Step 8 — Set Loss Function, Optimizer, Scheduler

CrossEntropy with label smoothing

AdamW optimizer

Learning rate scheduler that gradually reduces LR

Training for up to 80 epochs

🔥 Step 9 — Training Loop (with MixUp!)

For each training batch:

Sometimes apply MixUp (blending two audio samples + their labels).

Feed data to model.

Compute loss.

Backpropagate and update model.

Clip gradients for stability.

After each epoch:

Evaluate on validation set.

If the model improved → save checkpoint.

If no improvement for too long → early stopping.

🚀 Step 10 — Inference & Create Submission File

Once training is complete:

Load the saved best model.

Run predictions on test spectrograms.

Convert numeric predictions back to class names.

Save them into submission.csv.
