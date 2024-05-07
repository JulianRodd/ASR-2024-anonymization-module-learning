import logging

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from speaker_anonymization.data import get_audio_data_wavs
from speaker_anonymization.losses import average_wer, normalized_wer
from speaker_anonymization.utils import load_audio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_pretrained_model(CONFIG):
    """Load and return the pre-trained Wav2Vec2 model and processor."""

    processor = Wav2Vec2Processor.from_pretrained(CONFIG.ASR_BACKBONE)
    model = Wav2Vec2ForCTC.from_pretrained(CONFIG.ASR_BACKBONE)
    model.eval()
    logging.info(
        "Loaded Wav2Vec2 large robust model and general processor successfully."
    )
    return processor, model


def transcribe_audio(processor, model, waveform):
    """Transcribe audio using the loaded model and processor."""
    input_values = processor(
        waveform, sampling_rate=16000, return_tensors="pt", padding=True
    ).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    logging.debug("Transcription complete.")
    return transcription


def main():
    processor, model = load_pretrained_model()
    files, ground_truths, _ = get_audio_data_wavs()

    transcriptions = []
    for file, truth in zip(files, ground_truths):
        waveform, sr = load_audio(file)
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
