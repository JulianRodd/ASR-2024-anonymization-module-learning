import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from data import get_audio_data_wavs


class SpeakerVerificationModel:
    def __init__(self, num_speakers):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.model.config.hidden_size, num_speakers)
        self.num_speakers = num_speakers
        self.resampler = T.Resample(orig_freq=48000, new_freq=16000)  # Add a resampler
        print(f"Initialized model with {num_speakers} speakers for fine-tuning.")

    def forward(self, input_values):
        with torch.no_grad():
            outputs = self.model(input_values)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(embeddings)
        return logits

    def finetune_model(self, speaker_labels, files, n_epochs=10, learning_rate=1e-2):
        try:
            cached_model = torch.load(
                f"checkpoints/speaker_verification_model_{self.num_speakers}_{n_epochs}_{learning_rate}.pt"
            )
        except FileNotFoundError:
            cached_model = None
        if cached_model:
            self.classifier.load_state_dict(cached_model)
            print(
                f"Loaded cached model for {self.num_speakers} speakers and {n_epochs} epochs."
            )
            return
        self.model.train()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        mean_loss_per_epoch = []
        for epoch in range(n_epochs):
            print(f"Starting epoch {epoch + 1}/{n_epochs}")
            losses = []
            for file, label in tqdm(
                zip(files, speaker_labels), total=len(files), desc="Training"
            ):
                waveform, sample_rate = torchaudio.load(file)

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0)

                if sample_rate != 16000:
                    waveform = self.resampler(waveform)  # Resample to 16000 Hz

                waveform = waveform.unsqueeze(0)
                input_values = (
                    self.processor(waveform, sampling_rate=16000, return_tensors="pt")
                    .input_values.squeeze(0)
                    .squeeze(0)
                )

                logits = self.forward(input_values)
                loss = criterion(logits, torch.tensor([label]))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
            mean_loss_per_epoch.append(np.mean(losses))
            print("Mean loss:", np.mean(losses))

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            self.classifier.state_dict(),
            f"checkpoints/speaker_verification_model_{self.num_speakers}_{n_epochs}_{learning_rate}.pt",
        )
        self.plot_losses(mean_loss_per_epoch, n_epochs, self.num_speakers, learning_rate)

    def plot_losses(self, mean_loss_per_epoch, num_epochs, num_speakers, learning_rate):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch, marker="o")
        plt.title("Training Mean Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot in a specified directory
        images_dir = f"images/speaker_verification_training_{num_epochs}_{num_speakers}_{learning_rate}"
        os.makedirs(images_dir, exist_ok=True)
        plot_path = f"{images_dir}/mean_losses_per_epoch.png"
        plt.savefig(plot_path)
        print(f"Loss plot saved to {plot_path}")

    def get_speakers(self, files):
        self.model.eval()
        predicted_speakers = []
        for file in tqdm(files, desc="Predicting speakers"):
            waveform, sample_rate = torchaudio.load(file)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)

            if sample_rate != 16000:
                waveform = self.resampler(waveform)  # Resample to 16000 Hz

            waveform = waveform.unsqueeze(0)
            input_values = (
                self.processor(waveform, sampling_rate=16000, return_tensors="pt")
                .input_values.squeeze(0)
                .squeeze(0)
            )
            logits = self.forward(input_values)
            predicted_speaker = torch.argmax(logits).item()
            predicted_speakers.append(predicted_speaker)
        return predicted_speakers

    def get_speakers_using_waveforms(self, waveforms):
        predicted_speakers = []
        for waveform in waveforms:
            predicted_speaker = self.get_speaker_using_waveform(waveform)
            predicted_speakers.append(predicted_speaker)
        return predicted_speakers

    def get_speaker_using_waveform(self, waveform):
        self.model.eval()
        # Ensure waveform is a 1D tensor
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(
            0
        )  # Add channel dimension

        # Validate if resampling is needed
        if waveform.size(1) != 16000:
            waveform = self.resampler(waveform)

        # Process waveform through the processor
        input_values = self.processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_values

        # Predict using the model
        logits = self.forward(input_values.squeeze(0))
        predicted_speaker = torch.argmax(logits, dim=1).item()
        return predicted_speaker


def speaker_verification_loss(real_speakers, predicted_speakers):
    accuracy = accuracy_score(real_speakers, predicted_speakers)
    return accuracy


# Example of usage
if __name__ == "__main__":
    file_paths, _, speakers = get_audio_data_wavs()

    print(speakers)
    num_speakers = len(set(speakers))

    model = SpeakerVerificationModel(num_speakers)
    model.finetune_model(
        speakers,
        file_paths,
    )
    predicted_speakers = model.get_speakers(
        file_paths,
    )
    print("Predicted Speakers:", predicted_speakers)
    accuracy = speaker_verification_loss(speakers, predicted_speakers)
    print("Accuracy:", accuracy)
