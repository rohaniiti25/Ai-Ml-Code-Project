# About-Projects-1

➡️ Load audio files → Convert them to mel-spectrograms → Train a CNN → Evaluate → Predict → Export CSV.

## 🔧 Step 0 — Setup

You define where the training and test audio are stored, and where the final CSV output will be saved.
You also set a small configuration (like how much to augment audio).

## 🎲 Step 1 — Import Libraries & Fix Randomness

You import all the tools you need (PyTorch, NumPy, librosa, etc.).
You also set a seed (42) so the results are reproducible and stable every run.

## 🎛 Step 2 — Audio Augmentation

You create a function that randomly modifies audio to make the model more robust.

The audio can be:

unchanged

made noisy

pitch shifted

speed changed

time-shifted (shifted left/right)

given a fake reverb

This effectively doubles your dataset and prevents overfitting.

## 🎵 Step 3 — Convert Audio to Mel-Spectrograms

You load each .wav file, trim silence, optionally augment it, make sure it's 5 seconds long, and then convert it to a mel-spectrogram (image-like representation that CNNs understand).

## 📦 Step 4 — Build Training & Test Feature Sets

You scan your train/ folder where each subfolder is a class.

For every audio file:

Extract mel-spectrogram.

Save feature + label.

Optionally create an augmented version too.

For test audio files, you just extract features (no labels).

## 🏷 Step 5 — Encode Labels & Split Data

Since the model works with numbers, not text, you convert class names into integers.

Then you split the dataset:

80% training

20% validation

## 📚 Step 6 — Torch Dataset/Dataloader

You wrap your spectrograms into PyTorch Datasets so they can be fed into the model in batches.

## 🧠 Step 7 — Build a CNN Model

You define a convolutional neural network that:

Takes a spectrogram image

Passes it through several Conv → ReLU → Pool layers

Compresses it

Sends it through fully connected layers

Outputs class scores

This is your classifier.

## ⚙️ Step 8 — Set Loss Function, Optimizer, Scheduler

CrossEntropy with label smoothing

AdamW optimizer

Learning rate scheduler that gradually reduces LR

Training for up to 80 epochs

## 🔥 Step 9 — Training Loop (with MixUp!)

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

# 🚀 Step 10 — Inference & Create Submission File

Once training is complete:

Load the saved best model.

Run predictions on test spectrograms.

Convert numeric predictions back to class names.

Save them into submission.csv.


#

# About-Project-2

## 1. Get Ready
🛠️ Load all the computer tools we need

Make sure we can read sound files and build AI brains

## 2. Organize Your Sounds 📁
Look for folders with sounds (like "dog/", "car/")

Take each sound and turn it into a "sound picture" (spectrogram)

Make all pictures the same size

Remember which category each sound belongs to

## 3. Build Two AI Brains 🧠
🎨 The Creator:

Takes random noise + a category (like "dog")

"Draws" new sound pictures

Tries to make them look real

🔍 The Detective:

Looks at real and fake sound pictures

Guesses which ones are real

Learns to spot fakes

## 4. The Training Game 🎯
Round 1 - Train Detective:

Show real pictures → "This is real!"

Show fake pictures → "This is fake!"

Detective learns the difference

Round 2 - Train Creator:

Make new fake pictures

Try to fool the Detective

Creator learns to make better fakes

Repeat - Both get smarter each round!

## 5. Make New Sounds 🔊
Tell Creator what category we want

It makes a new sound picture

Turn that picture back into real sound

Save as .wav file you can play

6. Show Progress 📊
Save new sound files every few rounds

Save pictures to see improvement

Watch the AI get better over time
