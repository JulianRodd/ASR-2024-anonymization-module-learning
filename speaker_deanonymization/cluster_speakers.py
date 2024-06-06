from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import os
import json
from datasets import load_dataset

from characterictics_model import split_dataset_spi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pedalboard import (
    Bitcrush,
    Chorus,
    Distortion,
    Gain,
    HighpassFilter,
    LowpassFilter,
    Pedalboard,
    Phaser,
    PitchShift,
    time_stretch,
)

def apply_audio_effects(audio, sample_rate, params):
        """
        Apply audio effects based on pedalboard library with given parameters.
        Convert PyTorch tensor to numpy array for compatibility with the pedalboard library.
        """

        is_torch = False
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
            is_torch = True

        board = Pedalboard(
            [
                Distortion(drive_db=params["distortion_drive_db"]),
                PitchShift(semitones=params["pitch_shift_semitones"]),
                HighpassFilter(cutoff_frequency_hz=params["highpass_cutoff"]),
                LowpassFilter(cutoff_frequency_hz=params["lowpass_cutoff"]),
                Bitcrush(bit_depth=params["bitcrush_bit_depth"]),
                Chorus(rate_hz=params["chorus_rate_hz"]),
                Phaser(rate_hz=params["phaser_rate_hz"]),
                Gain(gain_db=params["gain_db"]),
            ],
        )
        processed_audio = board(audio, sample_rate=int(sample_rate))
        processed_audio = time_stretch(
            processed_audio, sample_rate, params["time_stretch_factor"]
        )

        if is_torch:
            return torch.from_numpy(processed_audio)
            
        return processed_audio

class VCTKDataset(Dataset):
    def __init__(self, hf_dataset, processor, target_sampling_rate=16000, anonymize=False, best_params=None):
        self.dataset = hf_dataset
        self.processor = processor
        self.target_sampling_rate = target_sampling_rate

        self.anonymize = anonymize
        self.best_params = best_params

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        speech_array, original_sampling_rate = torchaudio.load(item['file'])

        if self.anonymize:
            speech_array = apply_audio_effects(speech_array, original_sampling_rate, self.best_params)

        resampler = torchaudio.transforms.Resample(original_sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze()
        inputs = self.processor(speech, sampling_rate=self.target_sampling_rate, return_tensors="pt", padding="longest")
        inputs = inputs['input_values'][0] # TODO: processor gives really bad audio, maybe don't include it
        
        return inputs

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)
model.eval()

# Load dataset
dataset = load_dataset(
    "vctk", split="train", cache_dir='cache'
)

with open('speaker_deanonymization/best_params.json', 'r') as f:
    best_params = json.load(f)

samples = 20
sampels_val = 5
n_speakers = 5
train_dataset, val_dataset = split_dataset_spi(dataset, num_train_samples_per_speaker=samples, num_val_samples_per_speaker=sampels_val, num_speakers=n_speakers)

# Create dataset and DataLoader for training
vctk_train_dataset = VCTKDataset(train_dataset, processor)
train_dataloader = DataLoader(vctk_train_dataset, batch_size=1, shuffle=False)

# Create dataset and DataLoader for validation
vctk_val_dataset = VCTKDataset(val_dataset, processor)
val_dataloader = DataLoader(vctk_val_dataset, batch_size=1, shuffle=False)

# Create dataset and DataLoader for validation
vctk_val_dataset_anon = VCTKDataset(val_dataset, processor, anonymize=True, best_params=best_params)
val_dataloader_anon = DataLoader(vctk_val_dataset, batch_size=1, shuffle=False)

# Define the cache directory and file paths
cache_dir = 'cache'
speaker_embeddings_dir = os.path.join(cache_dir, 'speaker_embeddings')
train_embeddings_file = os.path.join(speaker_embeddings_dir, f'normal_embeddings_train_subset_{samples}_speakers_{n_speakers}.npy')
val_embeddings_file = os.path.join(speaker_embeddings_dir, f'normal_embeddings_val_subset_{samples}_speakers_{n_speakers}.npy')
val_anon_embeddings_file = os.path.join(speaker_embeddings_dir, f'anonymized_embeddings_val_subset_{samples}_speakers_{n_speakers}.npy')

# Create the necessary directories if they don't exist
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(speaker_embeddings_dir, exist_ok=True)

# Function to extract embeddings
def extract_embeddings(dataloader, embeddings_file):
    num_samples = len(dataloader.dataset)
    embedding_dim = 1024  # The dimensionality of the wav2vec2 embeddings
    embeddings = np.zeros((num_samples, embedding_dim))
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
        print(f"Embeddings loaded from cache: {embeddings_file}")
    else:
        for i, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            with torch.no_grad():
                input_values = batch.to(device)
                outputs = model(input_values, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1].squeeze(0)  # Remove batch dimension
                mean_pooled = hidden_states.mean(dim=0).cpu().numpy()  # Mean pooling
                embeddings[i] = mean_pooled
        np.save(embeddings_file, embeddings)
        print(f"Embeddings saved to cache: {embeddings_file}")
    return embeddings

# Extract embeddings for training and validation datasets
train_embeddings = extract_embeddings(train_dataloader, train_embeddings_file)
val_embeddings = extract_embeddings(val_dataloader, val_embeddings_file)
val_anon_embeddings = extract_embeddings(val_dataloader_anon, val_anon_embeddings_file)

# Standardize the embeddings
scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)
val_embeddings_scaled = scaler.transform(val_embeddings)
val_anon_embeddings_scaled = scaler.transform(val_anon_embeddings)

# Assuming you have a function to extract speaker IDs from the dataset
train_speaker_ids = train_dataset['speaker_id']
val_speaker_ids = val_dataset['speaker_id']
    
# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_embeddings_scaled, train_speaker_ids)

# Predict on the validation set
val_predictions = knn.predict(val_embeddings_scaled)
val_anon_predictions = knn.predict(val_anon_embeddings_scaled)

# Evaluate the model
accuracy_norm = accuracy_score(val_speaker_ids, val_predictions)
accuracy_anon = accuracy_score(val_speaker_ids, val_anon_predictions)

f1_norm = f1_score(val_speaker_ids, val_predictions, average='weighted')
f1_anon = f1_score(val_speaker_ids, val_anon_predictions, average='weighted')

print(f"Validation Accuracy nomral: {accuracy_norm:.2f}, f1: {f1_norm:.2f}")
print(f"Validation Accuracy anonymized: {accuracy_anon:.2f}, f1: {f1_anon:.2f}")
print(classification_report(val_speaker_ids, val_predictions))
print(classification_report(val_speaker_ids, val_anon_predictions))
