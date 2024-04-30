import numpy as np
import torch
import torchaudio
from sklearn.metrics import accuracy_score
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class SpeakerVerificationModel:
    def __init__(self, num_speakers):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Linear(self.model.config.hidden_size, num_speakers)
        self.model.train()  # Set the model to training mode if fine-tuning
        self.num_speakers = num_speakers

    def finetune_model(self, speaker_labels, files):
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.classifier.parameters()), lr=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(10):
            for file, label in zip(files, speaker_labels):
                waveform, sample_rate = torchaudio.load(file)

                # Ensure the audio is mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0)

                waveform = waveform.unsqueeze(0)  # Should result in [1, num_samples]

                input_values = self.processor(
                    waveform, sampling_rate=sample_rate, return_tensors="pt"
                ).input_values
                input_values = input_values.squeeze(1).squeeze(0)

                outputs = self.model(input_values)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                logits = self.classifier(embeddings)
                target = np.zeros(self.num_speakers)
                target[label] = 1
                target = torch.tensor(target, dtype=torch.float64)
                loss = criterion(logits.squeeze(0), target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def get_speaker(self, files):
        self.model.eval()  # Set the model to evaluation mode
        predicted_speakers = []
        for file in files:
            waveform, sample_rate = torchaudio.load(file)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.unsqueeze(0)
            input_values = self.processor(
                waveform, sampling_rate=sample_rate, return_tensors="pt"
            ).input_values
            input_values = input_values.squeeze(1).squeeze(0)
            outputs = self.model(input_values)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            logits = self.classifier(embeddings)
            predicted_speaker = torch.argmax(logits).item()
            predicted_speakers.append(predicted_speaker)
        return predicted_speakers

    def speaker_verification_loss(self, real_speakers, predicted_speakers):
        """
        Computes the accuracy of speaker predictions.

        Args:
        real_speakers (list[int]): Actual speaker indices.
        predicted_speakers (list[int]): Predicted speaker indices by the model.

        Returns:
        float: Accuracy of the predictions.
        """
        accuracy = accuracy_score(real_speakers, predicted_speakers)
        return accuracy


# Example of usage
if __name__ == "__main__":
    num_speakers = 3  # Example for three different speakers
    model = SpeakerVerificationModel(num_speakers)

    # Example: Fine-tune and test with dummy data (you need actual data paths and labels)
    model.finetune_model(
        [0, 0, 1],  # Assuming labels are indices of the speakers
        ["data/vctk/p225_001.wav", "data/vctk/p225_002.wav", "data/vctk/p226_001.wav"],
    )
    predicted_speakers = model.get_speaker(
        ["data/vctk/p225_001.wav", "data/vctk/p225_002.wav", "data/vctk/p226_001.wav"]
    )
    print("Predicted Speakers:", predicted_speakers)
    
    # Calculate the accuracy of the model
    accuracy = model.speaker_verification_loss([0, 0, 1], predicted_speakers)
    print("Accuracy:", accuracy)
