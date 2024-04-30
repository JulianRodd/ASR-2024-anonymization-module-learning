import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import jiwer
import torchaudio

class ASRModel:
    def __init__(self):
        # Load the pre-trained model and processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.eval()  # Set the model to evaluation mode

    def transcribe_wav_file(self, file_path):
        """
        Transcribes a WAV file using a pre-trained Wav2Vec 2.0 model.
        
        Args:
        file_path (str): Path to the audio file to transcribe.

        Returns:
        str: The transcription of the audio file.
        """
        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Ensure the audio is mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure the waveform is correctly shaped: (batch_size, num_samples)
        waveform = waveform.squeeze(0)  # Remove the channel dimension if it's singular

        # Preprocess the audio
        input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values

        # Check input shape
        print(f"Input shape to the model: {input_values.shape}")

        # Perform inference
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode the model output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])

        return transcription

def asr_loss(actual_transcription, predicted_transcription):
    """
    Calculates the loss between the actual and predicted transcriptions using Word Error Rate (WER).
    
    Args:
    actual_transcription (str): The correct transcription of the audio.
    predicted_transcription (str): The transcription produced by the ASR model.

    Returns:
    float: The WER score representing the transcription error rate.
    """
    return jiwer.wer(actual_transcription, predicted_transcription)

# Example of usage
if __name__ == "__main__":
    asr = ASRModel()
    transcription = asr.transcribe_wav_file("data/vctk/p225_001.wav")
    print("Transcription:", transcription)
    print("ASR Loss:", asr_loss("PLEASE COOL STELLA", transcription))
