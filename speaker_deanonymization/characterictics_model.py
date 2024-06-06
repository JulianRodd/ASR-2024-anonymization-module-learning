import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
import time
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

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

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
# from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    # Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from transformers import Wav2Vec2Model, Wav2Vec2Processor

random.seed(42)
np.random.seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True

import warnings

# Suppress the specific UserWarning about requires_grad
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")

from transformers import Wav2Vec2Config

class CustomWav2Vec2Config(Wav2Vec2Config):
    def __init__(self, num_labels=0, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels

class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size // 2, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config, num_labels):
        super().__init__(config)
        
        self.config = config
        self.num_labels = num_labels
        self.wav2vec2 = Wav2Vec2Model(config)

        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        self.head = ModelHead(config, self.num_labels)
        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, num_labels, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = CustomWav2Vec2Config.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels, **kwargs)
        return super(AgeGenderModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, config=config, num_labels=num_labels, **kwargs)

    def forward(
            self,
            input_values,
    ):
        with torch.no_grad():
            outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)

        logits = self.head(hidden_states)
        if self.num_labels > 1:
            logits = torch.softmax(logits, dim=1)  

        return hidden_states, logits
    
    def train_epoch(self, train_loader, loss_fn, optimizer):
        self.train()
        train_loss = 0.0
        all_labels = []
        all_preds = []
        for batch in tqdm(train_loader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device) 

            _, logits = self(inputs)
            if self.num_labels > 1:
                loss = loss_fn(logits, labels)
            else:
                loss = loss_fn(logits.squeeze(0), labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            if self.num_labels > 1:
                all_labels.extend(labels.cpu().numpy())
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
            else:
                all_labels.extend(labels.cpu().numpy() * 100)
                all_preds.extend(logits.squeeze(0).cpu().detach().numpy() * 100)
            
        avg_train_loss = train_loss / len(train_loader)
        
        scores = {}

        if self.num_labels > 1:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            scores['Accuracy'] = accuracy; scores['f1'] = f1
        else:
            mae = mean_absolute_error(all_labels, all_preds)
            mse = mean_squared_error(all_labels, all_preds)
            r2 = r2_score(all_labels, all_preds)
            scores['MAE'] = mae; scores['MSE'] = mse; scores['R2'] = r2

        return avg_train_loss, scores

    def val_epoch(self, val_loader, loss_fn=None):
        self.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                _, logits = self(inputs)
                
                if loss_fn:
                    if self.num_labels > 1:
                        loss = loss_fn(logits, labels)
                    else:
                        loss = loss_fn(logits.squeeze(0), labels)

                    val_loss += loss.item()
                
                if self.num_labels > 1:
                    all_labels.extend(labels.cpu().numpy())
                    _, preds = torch.max(logits, 1)
                    all_preds.extend(preds.cpu().numpy())
                else:
                    all_labels.extend(labels.cpu().numpy() * 100)
                    all_preds.extend(logits.squeeze(0).cpu().detach().numpy() * 100)

        scores = {}
        if self.num_labels > 1:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            scores['Accuracy'] = accuracy; scores['f1'] = f1
        else:         
            mae = mean_absolute_error(all_labels, all_preds)
            mse = mean_squared_error(all_labels, all_preds)
            r2 = r2_score(all_labels, all_preds)
            scores['MAE'] = mae; scores['MSE'] = mse; scores['R2'] = r2

        if loss_fn:
            avg_val_loss = val_loss / len(val_loader)
            return avg_val_loss, scores
        else:
            print(all_labels)
            print(all_preds)
        return scores

    def fit(self, train_loader, val_loader, num_epochs=3, lr=1e-4, name=''):
        self.to(device)
        
        optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)

        loss_train_epochs = []
        loss_val_epochs = []

        if self.num_labels > 1:
            loss_fn = torch.nn.CrossEntropyLoss()
            scores_train = {'Accuracy':[], 'f1':[]}
            scores_val = {'Accuracy':[], 'f1':[]}
        else:
            loss_fn = torch.nn.MSELoss()
            scores_train = {'MAE':[], 'MSE':[], 'R2':[]}
            scores_val = {'MAE':[], 'MSE':[], 'R2':[]}
        
        for epoch in range(num_epochs):
            
            avg_train_loss, scores = self.train_epoch(train_loader, loss_fn, optimizer)
            metrics_str = ", ".join([f"{key}: {val:.4f}" for key, val in scores.items()])

            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, {metrics_str}")

            for key in scores_train:
                if key in scores:
                    scores_train[key].append(scores[key])

            avg_val_loss, scores = self.val_epoch(val_loader, loss_fn)

            metrics_str = ", ".join([f"{key}: {val:.4f}" for key, val in scores.items()])
            
            print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}, {metrics_str}")

            for key in scores_val:
                if key in scores:
                    scores_val[key].append(scores[key])

            loss_train_epochs.append(avg_train_loss)
            loss_val_epochs.append(avg_val_loss)

        plot_losses(loss_train_epochs, loss_val_epochs, scores_train, scores_val,
                    num_epochs, lr, name)
        
        local_dir = f"checkpoints/deanonymization"
        os.makedirs(local_dir, exist_ok=True)
        if self.num_labels > 1:
            torch.save(
                self.head.state_dict(),
                f"{local_dir}/gender_classification_{name}_{num_epochs}_{lr}.pt",
            )
        else:
            torch.save(
                self.head.state_dict(),
                f"{local_dir}/age_regression_{name}_{num_epochs}_{lr}.pt",
            )

class SpeakerIdentificationModel(nn.Module):
    def __init__(self, model_name, num_speakers):
        super(SpeakerIdentificationModel, self).__init__()
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze Wav2Vec2Model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.model.config.hidden_size, num_speakers).to(device)
        
        print(f"Initialized model with {num_speakers} speakers for fine-tuning.")

    def forward(self, input_values):
        with torch.no_grad():
            outputs = self.model(input_values.to(device))
        embeddings = outputs.last_hidden_state.mean(dim=1).to(device)
        logits = self.classifier(embeddings)
        return logits
    
    def train_epoch(self, train_loader, loss_fn, optimizer):
        self.model.train()

        train_loss = 0.0
        all_labels = []
        all_preds = []

        for inputs, labels in tqdm(
            train_loader, total=len(train_loader), desc="Training"
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = self(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)

        scores = {}

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        scores['Accuracy'] = accuracy; scores['f1'] = f1

        return avg_train_loss, scores

    def val_epoch(self, val_loader, loss_fn=None):
        self.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                logits = self(inputs)
                
                if loss_fn:
                    loss = loss_fn(logits, labels)
                    val_loss += loss.item()
                
                all_labels.extend(labels.cpu().numpy())
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                

        scores = {}

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        scores['Accuracy'] = accuracy; scores['f1'] = f1

        if loss_fn:
            avg_val_loss = val_loss / len(val_loader)
            return avg_val_loss, scores
        else:
            print(all_labels)
            print(all_preds)
        return scores

    def fit(self, train_loader, val_loader, num_epochs=3, lr=1e-2, name=''):
        loss_train_epochs = []
        loss_val_epochs = []

        scores_train = {'Accuracy':[], 'f1':[]}
        scores_val = {'Accuracy':[], 'f1':[]}

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):

            avg_train_loss, scores = self.train_epoch(train_loader, criterion, optimizer)

            metrics_str = ", ".join([f"{key}: {val:.4f}" for key, val in scores.items()])

            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, {metrics_str}")

            for key in scores_train:
                if key in scores:
                    scores_train[key].append(scores[key])

            avg_val_loss, scores = self.val_epoch(val_loader, criterion)

            metrics_str = ", ".join([f"{key}: {val:.4f}" for key, val in scores.items()])
            
            print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}, {metrics_str}")

            for key in scores_val:
                if key in scores:
                    scores_val[key].append(scores[key])

            loss_train_epochs.append(avg_train_loss)
            loss_val_epochs.append(avg_val_loss)

        plot_losses(loss_train_epochs, loss_val_epochs, scores_train, scores_val,
                    num_epochs, lr, name)
        
        local_dir = f"checkpoints/deanonymization"
        os.makedirs(local_dir, exist_ok=True)
        torch.save(
            self.classifier.state_dict(),
            f"{local_dir}/spi_{name}_{num_epochs}_{lr}.pt",
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

# Define custom dataset
class VCTKDataset(Dataset):
    def __init__(self, hf_dataset, processor, target_sampling_rate=16000, task='gender', anonymize=False, best_params=None, accents=None, in_memory=True):
        self.dataset = hf_dataset
        self.processor = processor
        self.target_sampling_rate = target_sampling_rate
        self.task = task
        self.anonymize = anonymize
        self.best_params = best_params
        self.accent_mapping = {accent: idx for idx, accent in enumerate(accents)} if accents else {}
        print(self.accent_mapping)
        self.in_memory = in_memory
        self.resampler = torchaudio.transforms.Resample(48000, self.target_sampling_rate)

        if self.task == 'spi':
            unique_speaker_ids = sorted(set(hf_dataset['speaker_id']))
            self.speaker_mapping = {speaker: idx for idx, speaker in enumerate(unique_speaker_ids)}

        if self.in_memory:
            self.data = []
            for i in tqdm(range(len(self.dataset)), desc="Loading dataset: "):
                self.data.append(self._load_item(i))

    def __len__(self):
        return len(self.dataset)

    def _load_item(self, idx):
        item = self.dataset[idx]
        speech_array, original_sampling_rate = torchaudio.load(item['file'])

        if self.anonymize:
            speech_array = apply_audio_effects(speech_array, original_sampling_rate, self.best_params)

        speech = self.resampler(speech_array).squeeze()
        inputs = self.processor(speech, sampling_rate=self.target_sampling_rate, return_tensors="pt")
        inputs = inputs['input_values'][0]  # TODO: processor gives really bad audio, maybe don't include it

        if self.task == 'gender':
            target = 1 if item['gender'] == 'M' else 0
            target = torch.tensor(target, dtype=torch.long)
        elif self.task == 'age':
            target = float(item['age']) / 100
            target = torch.tensor(target, dtype=torch.float32)
        elif self.task == 'accent':
            target = self.accent_mapping.get(item['accent'], -1)
            target = torch.tensor(target, dtype=torch.long)
        elif self.task == 'spi':
            target = self.speaker_mapping[item['speaker_id']]
            target = torch.tensor(target, dtype=torch.long)

        return inputs, target

    def __getitem__(self, idx):
        if self.in_memory:
            return self.data[idx]
        else:
            return self._load_item(idx)

# Create train/validation split
def split_dataset_by_speaker(dataset, num_samples=10, balance_accents=False, test_size=0.2, top_n_accents=None):
    # Count occurrences of each accent
    accent_counts = Counter(dataset['accent'])

    # If top_n_accents is specified, select the top n occurring accents
    if top_n_accents is not None and balance_accents:
        top_accents = [accent for accent, _ in accent_counts.most_common(top_n_accents)]
        top_accents_indices = []
        for i, accent in enumerate(dataset['accent']):
            if accent in top_accents:
                top_accents_indices.append(i)
        dataset = dataset.select(top_accents_indices)
    
    speakers = list(set(dataset['speaker_id']))    
    accents = list(set(dataset['accent']))

    if balance_accents:
        speaker_accents = {}
        for speaker, accent in zip(dataset['speaker_id'], dataset['accent']):
            if speaker not in speaker_accents:
                speaker_accents[speaker] = accent

        accent_to_speakers = defaultdict(list)
        for speaker, accent in speaker_accents.items():
            accent_to_speakers[accent].append(speaker)

        train_speakers = []
        val_speakers = []
        for accent, speakers in accent_to_speakers.items():
            if len(speakers) == 1:
                train_speakers.append(speakers)
                val_speakers.append(speakers)
            else:
                train, val = train_test_split(speakers, test_size=test_size, random_state=42)
                train_speakers.extend(train)
                val_speakers.extend(val)
    else:
        train_speakers, val_speakers = train_test_split(speakers, test_size=test_size, random_state=42)

    train_indices = []
    val_indices = []

    indices_train = defaultdict(list)
    indices_val = defaultdict(list)
    for i, (speaker, name) in enumerate(zip(dataset['speaker_id'], 
                                            (dataset['accent'] if balance_accents else dataset['speaker_id']))):
        if speaker in train_speakers:
            indices_train[name].append(i)
        if speaker in val_speakers:
            indices_val[name].append(i)

    for name in (top_accents if balance_accents and top_n_accents else (accents if balance_accents else train_speakers)):
        indices = indices_train[name]
        if len(indices) > num_samples:
            indices = random.sample(indices, num_samples)
        train_indices.extend(indices)

    for name in (top_accents if balance_accents and top_n_accents else (accents if balance_accents else val_speakers)):
        indices = indices_val[name]
        if len(indices) > num_samples:
            indices = random.sample(indices, num_samples)
        val_indices.extend(indices)

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    return train_dataset, val_dataset

def split_dataset_spi(dataset, num_train_samples_per_speaker=10, num_val_samples_per_speaker=5, num_speakers=None, seed=42):
    # Set the random seed for reproducibility
    random.seed(seed)
    # Group indices by speaker
    speaker_indices = defaultdict(list)
    for idx, speaker in enumerate(dataset['speaker_id']):
        speaker_indices[speaker].append(idx)
    
    if num_speakers is not None:
        selected_speakers = random.sample(list(speaker_indices.keys()), num_speakers)
    else:
        selected_speakers = list(speaker_indices.keys())

    # Create lists to hold the selected indices
    train_indices = []
    val_indices = []

    for speaker in selected_speakers:
        indices = speaker_indices[speaker]

        # Shuffle indices for each speaker
        random.shuffle(indices)
        
        # Select train and validation indices
        if len(indices) >= num_train_samples_per_speaker + num_val_samples_per_speaker:
            train_indices.extend(indices[:num_train_samples_per_speaker])
            val_indices.extend(indices[num_train_samples_per_speaker:num_train_samples_per_speaker + num_val_samples_per_speaker])
        else:
            print(f"Speaker {speaker} does not have enough samples. Available: {len(indices)}, Required: {num_train_samples_per_speaker + num_val_samples_per_speaker}")

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    return train_dataset, val_dataset

def plot_losses(
        loss_per_epoch_train, loss_per_epoch_val, 
        scores_train, scores_val,
        num_epochs, learning_rate, name=''):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_per_epoch_train, marker="o", label='Training loss')
    plt.plot(range(1, num_epochs + 1), loss_per_epoch_val, marker="o", label='Val loss')
    for key, val in scores_train.items():
        plt.plot(range(1, num_epochs + 1), val, marker="o", label=f'Train {key}')
    for key, val in scores_val.items():
        plt.plot(range(1, num_epochs + 1), val, marker="o", label=f'Val {key}')
    plt.title("Training Mean Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    images_dir = f"images/speaker_deanonymization/"
    os.makedirs(images_dir, exist_ok=True)
    plot_path = f"{images_dir}/{name}_{num_epochs}_{learning_rate}_mean_losses_per_epoch.png"
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

class GenderAgeFromAudio:
    def __init__(self, num_labels, model_name, spi=False):
        # load model from hub
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        if spi: 
            self.model = SpeakerIdentificationModel(
                model_name=model_name, 
                num_speakers=num_labels
            ).to(device)
        else:
            self.model = AgeGenderModel.from_pretrained(
                pretrained_model_name_or_path=model_name,
                num_labels=num_labels
            ).to(device)

        # dummy signal
        self.sampling_rate = 16000
        self.signal = np.zeros((1, self.sampling_rate), dtype=np.float32)
        
    def process_func(
        self,
        x: np.ndarray,
        embeddings: bool = False,
    ) -> np.ndarray:
        r"""Predict age and gender or extract embeddings from raw audio signal."""

        # run through processor to normalize signal
        # always returns a batch, so we just get the first entry
        # then we put it on the device
        y = self.processor(x, sampling_rate=self.sampling_rate)
        y = y['input_values'][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(device)

        # run through model
        with torch.no_grad():
            y = self.model(y)
            if embeddings:
                y = y[0]
            else:
                y = torch.hstack([y[1], y[2]])

        # convert to numpy
        y = y.detach().cpu().numpy()

        return y

    def train(self, dataset, params=None):
        train_dataset, val_dataset = split_dataset_by_speaker(dataset, samples_per_speaker=1)

        train_dataset = VCTKDataset(train_dataset, self.processor)
        val_dataset = VCTKDataset(val_dataset, self.processor)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)

        self.model.fit(train_loader, val_loader, num_epochs=10, lr=1e-4, name='train_normal')

    def experiments(self, dataset, settings, params=None, task=None, accents=None):
        train_dataset, val_dataset = split_dataset_by_speaker(dataset, num_samples=settings['num_samples'], 
                                                              balance_accents=(True if task=='accent' else False), top_n_accents=settings['num_labels'])
        
        print(set(train_dataset['accent']))
        print(set(val_dataset['accent']))
        assert set(train_dataset['accent']) == set(val_dataset['accent'])
        train_dataset_norm = VCTKDataset(train_dataset, self.processor, task=task, accents=sorted(set(train_dataset['accent'])))
        val_dataset_norm = VCTKDataset(val_dataset, self.processor, task=task, accents=sorted(set(val_dataset['accent'])))

        train_loader_norm = DataLoader(train_dataset_norm, batch_size=1, shuffle=True)
        val_loader_norm = DataLoader(val_dataset_norm, batch_size=1)

        self.model.fit(train_loader_norm, val_loader_norm, num_epochs=settings['epochs'], lr=settings['lr'], name='train_normal')

        print(self.model.val_epoch(val_loader_norm))

        train_dataset_anon = VCTKDataset(train_dataset, self.processor, anonymize=True, task=task, best_params=params, accents=sorted(set(train_dataset['accent'])))
        val_dataset_anon = VCTKDataset(val_dataset, self.processor, anonymize=True, task=task, best_params=params, accents=sorted(set(val_dataset['accent'])))

        train_loader_anon = DataLoader(train_dataset_anon, batch_size=1, shuffle=True)
        val_loader_anon = DataLoader(val_dataset_anon, batch_size=1)

        print(self.model.val_epoch(val_loader_anon))
        
        self.model.fit(train_loader_anon, val_loader_anon, num_epochs=settings['epochs'], lr=settings['lr'], name='train_anonymized')

        print(self.model.val_epoch(val_loader_anon))

    def experiment_spi(self, dataset, settings, params=None):
        train_dataset, val_dataset = split_dataset_spi(dataset, num_train_samples_per_speaker=10, num_speakers=100)

        train_dataset_norm = VCTKDataset(train_dataset, self.processor, task='spi')
        val_dataset_norm = VCTKDataset(val_dataset, self.processor, task='spi')

        train_loader_norm = DataLoader(train_dataset_norm, batch_size=1, shuffle=True)
        val_loader_norm = DataLoader(val_dataset_norm, batch_size=1)

        self.model.fit(train_loader_norm, val_loader_norm, num_epochs=settings['epochs'], lr=settings['lr'], name='train_normal')

        print(self.model.val_epoch(val_loader_norm))

        train_dataset_anon = VCTKDataset(train_dataset, self.processor, anonymize=True, task='spi', best_params=params)
        val_dataset_anon = VCTKDataset(val_dataset, self.processor, anonymize=True, task='spi', best_params=params)

        train_loader_anon = DataLoader(train_dataset_anon, batch_size=1, shuffle=True)
        val_loader_anon = DataLoader(val_dataset_anon, batch_size=1)

        print(self.model.val_epoch(val_loader_anon))
        
        self.model.fit(train_loader_anon, val_loader_anon, num_epochs=settings['epochs'], lr=settings['lr'], name='train_anonymized')

        print(self.model.val_epoch(val_loader_anon))


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset(
        "vctk", split="train", cache_dir='cache'
    )

    accents = sorted(list(set(dataset['accent'])))
    speakers = sorted(list(set(dataset['speaker_id'])))

    settings = {
        'age': {'lr': 7e-7, 'epochs': 4, 'num_samples': 10, 'num_labels': 1, 'model_name': 'audeering/wav2vec2-large-robust-24-ft-age-gender'},
        'gender': {'lr': 5e-6, 'epochs': 1, 'num_samples': 1, 'num_labels': 3, 'model_name': 'audeering/wav2vec2-large-robust-24-ft-age-gender'},
        'accent': {'lr': 5e-3, 'epochs': 10, 'num_samples': 50, 'num_labels': 13, 'model_name': 'facebook/wav2vec2-base'},
        'spi': {'lr': 1e-2, 'epochs': 10, 'num_samples': 10, 'num_labels': len(speakers), 'model_name': 'facebook/wav2vec2-base'}
    }
    task = 'accent'

    with open('speaker_deanonymization/best_params.json', 'r') as f:
        best_params = json.load(f)

    model = GenderAgeFromAudio(num_labels=settings[task]['num_labels'], model_name=settings[task]['model_name'], spi=True)

    # model.experiment_spi(dataset, settings['spi'], params=best_params)
    model.experiments(dataset, settings[task], task=task, params=best_params, accents=accents)

    # print(type(dataset['audio']['array']))
    # max = 0
    # for i in range(len(dataset)):
    #     array = dataset[i]['audio']['array']
    #     if max < len(array):
    #         max = len(array)
    #         print(max)
    # print(max)
    # train_dataset, val_dataset = split_dataset_by_speaker(dataset)

    # model = GenderAgeFromAudio()
    # train_dataset = VCTKDataset(train_dataset, model.processor)
    # val_dataset = VCTKDataset(val_dataset, model.processor)

    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1)

    # for batch in train_loader:
    #     inputs, gender_labels, age_labels = batch
    #     inputs = inputs.to(device)
    #     gender_labels = gender_labels.to(device)
    #     age_labels = age_labels.to(device)

    #     print(inputs.shape)
    #     print(model.model(inputs))
    #     break
    
    # print(inputs.shape)
    # print(model.model(inputs))



