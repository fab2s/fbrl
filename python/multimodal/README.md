# Multimodal Training Pipeline
This project provides a Python script (multimodal_training.py) for training a bidirectional multimodal model on video inputs containing text (e.g., letters) and associated sounds. The model learns to map visuals to audio and vice versa in a self-supervised manner, using cycle consistency for robustness. It's designed as a proof-of-concept (POC) for child-like language acquisition simulation, starting with letters and extensible to syllables/words.

The script supports:
- Generating clean videos (text + sound) using gTTS for realistic audio or synthetic fallbacks.
- Adding noise to create training datasets.
- Training a single bidirectional model with resumable checkpoints.
 -Testing on videos, detecting missing modalities (silent video or black screen) and generating the missing part.

## Requirements

- Python 3.8+
- Required packages: torch, torchaudio, numpy, matplotlib, pillow, gtts (for realistic TTS), moviepy (for video handling), librosa (for MFCC inversion to audio in testing).
- Install via: pip install torch torchaudio numpy matplotlib pillow gtts moviepy librosa
- Limited hardware friendly: Small MLPs, batch size 13.

Notes:

- gTTS requires internet.
- If MoviePy/Librosa unavailable, script falls back to separate image/audio files (less ideal for pure video mode).
- For video mode, ensure MoviePy is installed; otherwise, it uses fallback.

## Usage
The script is CLI-based with commands: generate_clean, generate, train, test.
Run: python multimodal_training.py <command> [options]
1. Generate Clean Videos (or Audio Fallback)
Creates clean .mp4 videos (or .mp3/.wav audio if no MoviePy) for each letter.
````bash
python multimodal_training.py generate_clean --letters A-Z --output_dir data/clean_video --use_real_tts
````

- `--letters`: Comma-separated or range (default: A-Z).
- `--output_dir`: Where to save (default: data/clean_video).
- `--use_real_tts`: Use gTTS for realistic speech (default: False, uses sine waves).

2. Generate Noisy Dataset
Adds noise to clean videos for training. Requires clean_video_dir.
````bash
python multimodal_training.py generate --letters A-Z --noise_level 0.01 --output_dir data/letters --clean_video_dir data/clean_video
````

- `--letters`: As above.
- `--noise_level`: Gaussian noise std (default: 0.01).
- `--output_dir`: Save noisy videos/metadata (default: data/letters).
- `--clean_video_dir`: Path to clean videos (required).

If no MoviePy, generates separate noisy images/audios.

3. Train the Model
Trains on the noisy dataset (videos extracted to image frames + MFCC audio).
````bash
python multimodal_training.py train --data_dir data/letters --epochs 50 --save_dir models --checkpoint_interval 10 --resume models/checkpoint_epoch_10.pth
````

- `--data_dir`: Path to noisy dataset (required).
- `--epochs`: Number of epochs (default: 50).
- `--save_dir`: Where to save models/checkpoints/graph (default: models).
- `--checkpoint_interval`: Save every N epochs (default: 10).
- `--resume`: Path to checkpoint to resume from (optional).

Outputs: `model_final.pth`, checkpoints, `training_losses.png` graph.

4. Test the Model
Tests on videos, detects missing modality (silent audio or black video), generates the missing one. Saves outputs and quality graph if GT available.

````bash
python multimodal_training.py test --model_dir models --test_data_dir test_videos --output_dir results
````

- `--model_dir`: Path to saved model (required).
- `--test_data_dir`: Path to test videos (.mp4; craft silent/black for partial tests) (required).
- `--output_dir`: Save generated files/graph (default: results).

Outputs: Generated audio (.wav or .npy), images (.png), `test_scores.png` bar graph of MSE scores.

## Model Architecture

- BidirectionalModel: Single model with shared encoders (visual/audio to latent) and decoders (latent to audio/visual).
- Inputs: Video frame (32x32 grayscale image) + MFCC audio features (13x50).
- Losses: Reconstruction (V2A + A2V) + cycle consistency for mismatch handling.
- Extensible: For videos with sequences, add RNN/Transformer; for mismatches, cycle loss promotes reconciliation.

## Extension Ideas

Syllables/Words: Modify generate_clean_video to use multi-letter text/audio.
Video Sequences: Update dataset to process frame sequences (e.g., add ConvLSTM).
Real Data: Replace synthetic with real child videos (e.g., via datasets like PAVSig).
Output Audio: Use Librosa to invert MFCC to wav in training/test for playable sounds.

## Limitations

POC scale: Simple MLPs; for real tasks, use deeper nets (e.g., ViT for visuals, Wav2Vec for audio).
Video Fallback: If no MoviePy, uses separate files—install for pure video mode.
Noise: Gaussian only; extend for realistic distortions.

For issues, check console for fallbacks/warnings.
