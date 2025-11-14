What This Does
AI that creates new sounds from your audio examples.

Quick Start
Put audio in folders:

text
train/
├── dog/
│   └── barks.wav
├── car/
│   └── horns.wav
Run:

bash
python gan_audio.py --data train/ --epochs 100
Get new sounds in gan_generated_audio/ folder!

How It Works
Step 1: Turn Sound into Pictures
Convert .wav files to spectrograms (sound images)

All images are same size: 128x512

Step 2: Train Two AIs
🎨 Generator AI - Creates fake sound pictures
🔍 Discriminator AI - Spots fake vs real pictures

They play a game: Generator tries to fool Discriminator, both get better!

Step 3: Make New Sounds
Generator creates new spectrograms

Convert pictures back to .wav files

Save new audio you can listen to

Command Options
--data folder/ - Where your audio is

--epochs 100 - Training time (more = better quality)

--batch_size 32 - How many files to learn from at once

--play - Try to play generated audio

What You Get
text
📁 gan_generated_audio/
   ├── dog_ep050.wav  ← New dog sounds!
   └── car_ep050.wav  ← New car sounds!
