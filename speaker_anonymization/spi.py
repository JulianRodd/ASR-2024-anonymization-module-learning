import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.transforms as T
from torch import nn
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from speaker_anonymization.config import Config
from speaker_anonymization.data import get_audio_data_wavs
from speaker_anonymization.losses import speaker_verification_loss
from speaker_anonymization.utils import load_audio


class SpeakerIdentificationModel:
    def __init__(self, num_speakers, CONFIG):
        self.study_name = CONFIG.STUDY_NAME
        self.processor = Wav2Vec2Processor.from_pretrained(CONFIG.SPI_BACKBONE)
        self.model = Wav2Vec2Model.from_pretrained(CONFIG.SPI_BACKBONE)

        for param in self.model.parameters():
            param.requires_grad = False

        self.speaker_classifier = nn.Linear(self.model.config.hidden_size, num_speakers)
        self.age_classifier = nn.Linear(self.model.config.hidden_size, 100) #assuming max age is 100
        self.gender_classifier = nn.Linear(self.model.config.hidden_size, 2)
        self.accent_classifier = nn.Linear(self.model.config.hidden_size, num_speakers)
        self.region_classifier = nn.Linear(self.model.config.hidden_size, num_speakers)
        self.num_speakers = num_speakers
        self.resampler = T.Resample(orig_freq=48000, new_freq=16000)
        print(f"Initialized model with {num_speakers} speakers for fine-tuning.")

    def forward(self, input_values):
        with torch.no_grad():
            outputs = self.model(input_values)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        speaker_logits = self.speaker_classifier(embeddings)
        age_logits = self.age_classifier(embeddings)
        gender_logits = self.gender_classifier(embeddings)
        accent_logits = self.accent_classifier(embeddings)
        region_logits = self.region_classifier(embeddings)
        return speaker_logits, age_logits, gender_logits, accent_logits, region_logits

    def finetune_model(self, speaker_labels, age_labels, gender_labels, accent_labels, region_labels, files, n_epochs=10, learning_rate=1e-2):
        try:
            cached_model = torch.load(
                f"checkpoints/{self.study_name}/new_loss_speaker_verification_model_{self.num_speakers}_{n_epochs}_{learning_rate}.pt"
            )
        except FileNotFoundError:
            cached_model = None
        if cached_model:
            self.speaker_classifier.load_state_dict(cached_model['speaker_classifier'])
            self.age_classifier.load_state_dict(cached_model['age_classifier'])
            self.gender_classifier.load_state_dict(cached_model['gender_classifier'])
            self.accent_classifier.load_state_dict(cached_model['accent_classifier'])
            self.region_classifier.load_state_dict(cached_model['region_classifier'])
            print(
                f"Loaded cached model for {self.num_speakers} speakers and {n_epochs} epochs."
            )
            return
        self.model.train()
        optimizer = torch.optim.Adam(
            list(self.speaker_classifier.parameters()) + 
            list(self.age_classifier.parameters()) + 
            list(self.gender_classifier.parameters()) + 
            list(self.accent_classifier.parameters()) + 
            list(self.region_classifier.parameters()), 
            lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        mean_loss_per_epoch = []
        mean_speaker_loss_per_epoch = []
        mean_age_loss_per_epoch = []
        mean_gender_loss_per_epoch = []
        mean_accent_loss_per_epoch =[]
        mean_region_loss_per_epoch = []
        for epoch in range(n_epochs):
            print(f"Starting epoch {epoch + 1}/{n_epochs}")
            losses = []
            losses_speaker = []
            losses_age = []
            losses_gender = []
            losses_accent = []
            losses_region = []
            for file, speaker_label, age_label, gender_label, accent_label, region_label in tqdm(
                zip(files, speaker_labels, age_labels, gender_labels, accent_labels, region_labels), total=len(files), desc="Training"
            ):
                waveform, sample_rate = load_audio(file)

                waveform = waveform.unsqueeze(0)
                input_values = self.processor(
                    waveform, sampling_rate=sample_rate, return_tensors="pt"
                ).input_values.squeeze(0)

                speaker_logits, age_logits, gender_logits, accent_logits, region_logits = self.forward(input_values)
                loss_speaker = criterion(speaker_logits, torch.tensor([speaker_label]))
                loss_age = criterion(age_logits, torch.tensor([age_label], dtype=torch.long))
                loss_gender = criterion(gender_logits, torch.tensor([gender_label], dtype=torch.long))
                loss_accent = criterion(accent_logits, torch.tensor([accent_label], dtype=torch.long))
                loss_region = criterion(region_logits, torch.tensor([region_label], dtype=torch.long))
                loss = loss_speaker + loss_age + loss_gender + loss_accent + loss_region
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
                losses_speaker.append(loss_speaker.item())
                losses_age.append(loss_age.item())
                losses_gender.append(loss_gender.item())
                losses_accent.append(loss_accent.item())
                losses_region.append(loss_region.item())
            mean_loss_per_epoch.append(np.mean(losses))
            mean_speaker_loss_per_epoch.append(np.mean(losses_speaker))
            mean_age_loss_per_epoch.append(np.mean(losses_age))
            mean_gender_loss_per_epoch.append(np.mean(losses_gender))
            mean_accent_loss_per_epoch.append(np.mean(losses_accent))
            mean_region_loss_per_epoch.append(np.mean(losses_region))
            print("Mean loss:", np.mean(losses))

        local_dir = f"checkpoints/{self.study_name}"
        os.makedirs(local_dir, exist_ok=True)
        torch.save(
            self.speaker_classifier.state_dict(),
            f"{local_dir}/new_loss_speaker_verification_model_{self.num_speakers}_{n_epochs}_{learning_rate}.pt",
        )
        self.plot_losses(
            mean_loss_per_epoch, n_epochs, self.num_speakers, learning_rate, "combined"
        )
        self.plot_losses(
            mean_speaker_loss_per_epoch, n_epochs, self.num_speakers, learning_rate, "speaker"
        )
        self.plot_losses(
            mean_age_loss_per_epoch, n_epochs, self.num_speakers, learning_rate, "age"
        )
        self.plot_losses(
            mean_gender_loss_per_epoch, n_epochs, self.num_speakers, learning_rate, "gender"
        )
        self.plot_losses(
            mean_accent_loss_per_epoch, n_epochs, self.num_speakers, learning_rate, "accent"
        )
        self.plot_losses(
            mean_region_loss_per_epoch, n_epochs, self.num_speakers, learning_rate, "region"
        )

    def plot_losses(self, mean_loss_per_epoch, num_epochs, num_speakers, learning_rate, kind_of_loss):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch, marker="o")
        plt.title("Training Mean Loss per Epoch" + kind_of_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.grid(True)
        plt.tight_layout()

        images_dir = f"images/{self.study_name}/speaker_verification_training_{num_epochs}_{num_speakers}_{learning_rate}"
        os.makedirs(images_dir, exist_ok=True)
        plot_path = f"{images_dir}/mean_losses_per_epoch_{kind_of_loss}.png"
        plt.savefig(plot_path)
        print(f"Loss plot saved to {plot_path}")

    def get_speakers(self, files):
        self.model.eval()
        predicted_speakers = []
        predicted_ages = []
        predicted_genders = []
        predicted_accents = []
        predicted_regions = []
        for file in tqdm(files, desc="Predicting speakers"):
            waveform, sample_rate = load_audio(file)
            input_values = self.processor(
                waveform, sampling_rate=sample_rate, return_tensors="pt"
            ).input_values.squeeze(0)
            speaker_logits, age_logits, gender_logits, accent_logits, region_logits = self.forward(input_values.unsqueeze(0))
            predicted_speaker = torch.argmax(speaker_logits).item()
            predicted_age = torch.argmax(age_logits).item()
            predicted_gender = torch.argmax(gender_logits).item()
            predicted_accent = torch.argmax(accent_logits).item()
            predicted_region = torch.argmax(region_logits).item()
            predicted_speakers.append(predicted_speaker)
            predicted_ages.append(predicted_age)
            predicted_genders.append(predicted_gender)
            predicted_accents.append(predicted_accent)
            predicted_regions.append(predicted_region)
        return predicted_speakers, predicted_ages, predicted_genders, predicted_accents, predicted_regions

    def get_speakers_using_waveforms(self, waveforms, sample_rate=16000):
        predicted_speakers = []
        predicted_ages = []
        predicted_genders = []
        predicted_accents = []
        predicted_regions = []
        self.model.eval()
        for waveform in waveforms:
            input_values = self.processor(
                waveform, sampling_rate=sample_rate, return_tensors="pt"
            ).input_values.squeeze(0)
            speaker_logits, age_logits, gender_logits, accent_logits, region_logits = self.forward(input_values.unsqueeze(0))
            predicted_speaker = torch.argmax(speaker_logits).item()
            predicted_age = torch.argmax(age_logits).item()
            predicted_gender = torch.argmax(gender_logits).item()
            predicted_accent = torch.argmax(accent_logits).item()
            predicted_region = torch.argmax(region_logits).item()
            predicted_speakers.append(predicted_speaker)
            predicted_ages.append(predicted_age)
            predicted_genders.append(predicted_gender)
            predicted_accents.append(predicted_accent)
            predicted_regions.append(predicted_region)
        return predicted_speakers, predicted_ages, predicted_genders, predicted_accents, predicted_regions


if __name__ == "__main__":
    config = Config(n_speakers=10, n_samples_per_speaker=10)
    file_paths, _, speakers = get_audio_data_wavs(CONFIG=config)

    print(speakers)
    num_speakers = len(set(speakers))

    model = SpeakerIdentificationModel(num_speakers)
    model.finetune_model(
        speakers,
        file_paths,
        n_epochs=10,
    )
    predicted_speakers = model.get_speakers(
        file_paths,
    )
    print("Predicted Speakers:", predicted_speakers)
    accuracy = speaker_verification_loss(speakers, predicted_speakers)
    print("Accuracy:", accuracy)
