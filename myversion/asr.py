import logging

import jiwer
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
from data import get_audio_data_wavs

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_pretrained_model():
    """Load and return the pre-trained Wav2Vec2 model and processor."""
    # Use a well-known pre-trained model that includes a tokenizer and feature extractor.
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    model.eval()  # Set the model to evaluation mode, which deactivates dropout layers.
    logging.info("Loaded Wav2Vec2 large robust model and general processor successfully.")
    return processor, model


def
def preprocess_audio(file_path):
    """Load, resample, and convert audio to mono."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    logging.debug(f"Processed audio file {file_path} for transcription.")
    return waveform.squeeze(0)


def transcribe_audio(processor, model, waveform):
    """Transcribe audio using the loaded model and processor."""
    input_values = processor(
        waveform, sampling_rate=16000, return_tensors="pt"
    ).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    logging.debug("Transcription complete.")
    return transcription


def normalized_wer(actual, predicted):
    """
    Compute normalized WER between actual and predicted transcriptions.
    Handles cases where transformations might result in empty strings or lists.
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ])

    # Ensure the input is not empty after transformation
    if not actual.strip() or not predicted.strip():
        return 1.0  # Return the worst score if inputs are empty or only whitespace

    measures = jiwer.compute_measures(
        actual, predicted,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )

    wer = measures["wer"]
    return wer

def average_wer(actual_transcriptions, predicted_transcriptions):
    """
    Calculate the average Word Error Rate (WER) precisely, handling edge cases.
    """
    if len(actual_transcriptions) != len(predicted_transcriptions) or not actual_transcriptions:
        raise ValueError("Transcription lists are empty or of unequal length.")

    wers = [
        normalized_wer(actual, predicted) if actual and predicted else 1.0
        for actual, predicted in zip(actual_transcriptions, predicted_transcriptions)
    ]

    return sum(wers) / len(wers)


def main():
    processor, model = load_pretrained_model()
    files, ground_truths, _ = get_audio_data_wavs()

    transcriptions = []
    for file, truth in zip(files, ground_truths):
        waveform, sr = sf.read(file)
        transcription = transcribe_audio(processor, model, waveform)
        wer = normalized_wer(truth, transcription)
        logging.info(
            f"File: {file}, Truth: {truth}, Transcription: {transcription}, WER: {wer}"
        )
        transcriptions.append(transcription)

    overall_wer = average_wer(ground_truths, transcriptions)
    logging.info(f"Average Word Error Rate (WER) across all files: {overall_wer:.2f}")


if __name__ == "__main__":
    main()
