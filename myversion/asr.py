import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import jiwer
from data import get_audio_data_wavs
class ASRModel:
    def __init__(self, num_labels):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        # Initialize new classifier
        self.classifier = torch.nn.Linear(32, num_labels)

        # Freeze the pre-trained model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Set the classifier to training mode initially
        self.classifier.train()
        print("Model, processor, and new classifier initialized for fine-tuning.")

    def forward(self, input_values):
        # Getting the output from the Wav2Vec2 model
        output = self.model(input_values.squeeze(1))
        logits = output.logits
        # Pass the model outputs through the classifier
        logits = self.classifier(logits)
        return logits 

    def transcribe_wav_file(self, file_path):
        self.classifier.eval()  # Ensure classifier is in evaluation mode
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        input_values = self.processor(
            waveform, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        logits = self.forward(input_values)
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        return transcription

    def finetune_model(self, transcriptions, files, n_epochs=10, learning_rate=1e-4):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        loss_func = torch.nn.CTCLoss(
            blank=self.processor.tokenizer.pad_token_id, zero_infinity=True
        )

        for epoch in range(n_epochs):
            total_loss = 0
            self.classifier.train()
            for transcription, file_path in tqdm(
                zip(transcriptions, files), total=len(files), desc="Training"
            ):
                self.classifier.zero_grad()
                waveform, sample_rate = torchaudio.load(file_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0)
                input_values = self.processor(
                    waveform, sampling_rate=sample_rate, return_tensors="pt"
                ).input_values
                labels = self.processor.tokenizer(
                    transcription, return_tensors="pt"
                ).input_ids

                logits = self.forward(input_values)
                input_lengths = torch.full(
                    (input_values.size(0),), logits.shape[1], dtype=torch.long
                )
                label_lengths = torch.tensor(
                    [labels.shape[1]] * input_values.shape[0], dtype=torch.long
                )

                loss = loss_func(
                    logits.log_softmax(2).permute(1, 0, 2),
                    labels,
                    input_lengths,
                    label_lengths,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.classifier.parameters(), 5.0
                )
                optimizer.step()
                total_loss += loss.item()
                tqdm.write(f"File: {file_path}, Loss: {loss.item()}")

            average_loss = total_loss / len(files)
            print(f"Epoch {epoch + 1}/{n_epochs}, Mean Loss: {average_loss}")

def asr_loss(actual_transcriptions, predicted_transcriptions):
    wers = [
        jiwer.wer(actual, predicted)
        for actual, predicted in zip(actual_transcriptions, predicted_transcriptions)
    ]
    average_wer = sum(wers) / len(wers)
    return average_wer

if __name__ == "__main__":
    file_paths, transcriptions = get_audio_data_wavs()
    num_labels = len(set(transcriptions))  # Assuming labels are the second tuple element
    asr = ASRModel(num_labels)
    files, transcriptions = get_audio_data_wavs()  # Assuming this returns correct data
    asr.finetune_model(transcriptions, files)

    # Testing the fine-tuned model
    transcription = asr.transcribe_wav_file(files[0])
    print("Transcription:", transcription)
    print("ASR Loss:", asr_loss([transcriptions[0]], [transcription]))
