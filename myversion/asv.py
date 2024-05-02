import numpy as np
import torch
import torchaudio
from sklearn.metrics import accuracy_score
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm

from data import get_audio_data_wavs

class SpeakerVerificationModel:
    def __init__(self, num_speakers):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze the parameters of Wav2Vec2Model to prevent them from being updated during training
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(self.model.config.hidden_size, num_speakers)
        self.num_speakers = num_speakers
        print("Initialized model with {} speakers for fine-tuning.")

    def forward(self, input_values):
        with torch.no_grad():  # Ensuring no grad is computed for the base model
            outputs = self.model(input_values)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(embeddings)
        return logits
      
    def finetune_model(self, speaker_labels, files, n_epochs=10, learning_rate=1e-2):
        self.model.train()  # Set the model to training mode
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(n_epochs):  # Number of epochs
            print("Starting epoch {}/10".format(epoch + 1))
            losses = []
            for file, label in tqdm(zip(files, speaker_labels), total=len(files), desc="Training"):
                waveform, sample_rate = torchaudio.load(file)

                if waveform.shape[0] > 1:  # Ensure the audio is mono
                    waveform = waveform.mean(dim=0)

                waveform = waveform.unsqueeze(0)  # Resulting in [1, num_samples]
                input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values.squeeze(0).squeeze(0)

                logits = self.forward(input_values)
                loss = criterion(logits, torch.tensor([label]))  # CrossEntropyLoss expects class indices
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
            print("Mean loss:", np.mean(losses))
                
    
    def get_speaker(self, files):
        self.model.eval()  # Evaluation mode
        predicted_speakers = []
        for file in tqdm(files, desc="Predicting speakers"):
            waveform, sample_rate = torchaudio.load(file)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.unsqueeze(0)
            input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values.squeeze(0).squeeze(0)
            logits = self.forward(input_values)
            predicted_speaker = torch.argmax(logits).item()
            predicted_speakers.append(predicted_speaker)
        return predicted_speakers

def speaker_verification_loss(real_speakers, predicted_speakers):
    accuracy = accuracy_score(real_speakers, predicted_speakers)
    return accuracy

# Example of usage
if __name__ == "__main__":
    file_paths, _ = get_audio_data_wavs()
    speakers = [int(file.split("/")[-1].split("_")[0][1:]) - 225 for file in file_paths]

    num_speakers = len(set(speakers))
    
    model = SpeakerVerificationModel(num_speakers)
    model.finetune_model(
        speakers,
        file_paths,
    )
    predicted_speakers = model.get_speaker(
        file_paths,
    )
    print("Predicted Speakers:", predicted_speakers)
    accuracy = speaker_verification_loss(speakers, predicted_speakers)
    print("Accuracy:", accuracy)
