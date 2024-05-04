import logging
import os

import optuna
import soundfile as sf
import torch
from asr import load_audio, load_pretrained_model, transcribe_audio
from asv import SpeakerVerificationModel
from losses import calculate_combined_loss
from pedalboard import Distortion, HighpassFilter, LowpassFilter, Pedalboard, PitchShift
from utils import save_optimization_plots

from data import get_audio_data_wavs

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

images_dir = "images"
os.makedirs(images_dir, exist_ok=True)


def apply_audio_effects(audio, sample_rate, params):
    """
    Apply audio effects based on pedalboard library with given parameters.
    Convert PyTorch tensor to numpy array for compatibility with the pedalboard library.
    """
    # Convert PyTorch tensor to numpy array if necessary
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()

    board = Pedalboard(
        [
            Distortion(drive_db=params["distortion_drive_db"]),
            PitchShift(semitones=params["pitch_shift_semitones"]),
            HighpassFilter(cutoff_frequency_hz=params["highpass_cutoff"]),
            LowpassFilter(cutoff_frequency_hz=params["lowpass_cutoff"]),
        ],
    )
    processed_audio = board(audio, sample_rate=int(sample_rate))
    return processed_audio


def evaluate_asr_and_asv(audio_data, transcriptions, speakers):
    """Evaluate ASR and ASV models and log results using the modified audio."""
    predictions = []
    for waveform, sr in audio_data:
        #  waveform = preprocess_audio(audio, sr)

        prediction = transcribe_audio(processor, asr_model, waveform)
        predictions.append(prediction)

    predicted_speakers = speaker_verification.get_speakers_using_waveforms(
        [a for a, _ in audio_data]
    )  # Extract only audio from tuples
    combined_loss, avg_wer, accuracy = calculate_combined_loss(
        transcriptions, predictions, speakers, predicted_speakers
    )
    logging.info("Evaluation complete.\n")
    logging.info(f"Average WER: {avg_wer}\n")
    logging.info(f"Speaker Verification Accuracy: {accuracy}\n")

    logging.info(f"Combined Loss: {combined_loss}\n\n\n")

    return predictions, predicted_speakers, avg_wer, accuracy, combined_loss


def optimize_params(trial):
    """Define and apply hyperparameter optimization using normal distribution for parameters."""
    params = {
        "distortion_drive_db": trial.suggest_float(
            "distortion_drive_db",
            low=max(
                0, 25 - 2 * 12.5
            ),  # Assuming std deviation so that it covers a reasonable range
            high=25 + 2 * 12.5,
        ),
        "pitch_shift_semitones": trial.suggest_float(
            "pitch_shift_semitones",
            low=-5 - 2 * 2.5,  # Similar to above, adjust range
            high=min(0, -5 + 2 * 2.5),
        ),
        "highpass_cutoff": trial.suggest_float(
            "highpass_cutoff",
            low=max(0, 250 - 2 * 125),  # Ensure it does not go below 0 Hz
            high=250 + 2 * 125,
        ),
        "lowpass_cutoff": trial.suggest_float(
            "lowpass_cutoff",
            low=3500 - 2 * 750,  # Adjust the bounds as needed
            high=3500 + 2 * 750,
        ),
    }
    audio_data = []
    for file_path in file_paths:
        audio, sr = load_audio(file_path)
        processed_audio = apply_audio_effects(audio, float(sr), params)
        audio_data.append((processed_audio, sr))

    _, _, avg_wer, accuracy, combined_loss = evaluate_asr_and_asv(
        audio_data, transcriptions, speakers
    )

    trial.set_user_attr("avg_wer", avg_wer)
    trial.set_user_attr("speaker_accuracy", accuracy)
    trial.set_user_attr("combined_loss", combined_loss)
    return combined_loss


if __name__ == "__main__":
    logging.info("Starting audio effects optimization...\n")
    file_paths, transcriptions, speakers = get_audio_data_wavs(subset_size=100)
    num_speakers = len(set(speakers))
    processor, asr_model = load_pretrained_model()
    speaker_verification = SpeakerVerificationModel(num_speakers=num_speakers)
    speaker_verification.finetune_model(speakers, file_paths, n_epochs=20)
    logging.info("Speaker Verification model trained.\n\n")
    # Evaluate the initial model
    logging.info("Evaluating the initial model...\n")
    initial_audio_data = []
    for file_path in file_paths:
        audio, sr = load_audio(file_path)
        initial_audio_data.append((audio, sr))
    _, _, avg_wer, accuracy, combined_loss = evaluate_asr_and_asv(
        initial_audio_data, transcriptions, speakers
    )
    # Hyperparameter optimization
    study = optuna.create_study(
        direction="minimize", study_name="optimizing_audio_effects_for_anonymization"
    )
    study.optimize(optimize_params, n_trials=5)
    logging.info(
        f"Optimization complete. Best Parameters: {study.best_params}, Best Loss: {study.best_value}"
    )

    images_dir = (
        f"images/{study.study_name}_{str(num_speakers)}_speakers_{study.best_value:.2f}"
    )
    os.makedirs(images_dir, exist_ok=True)
    # Save optimization plots
    save_optimization_plots(study, images_dir)

    # Apply best parameters to audio files and store them
    best_params = study.best_params
    anon_folder = "anonymized_files"
    os.makedirs(anon_folder, exist_ok=True)

    for fp in file_paths:
        audio, sample_rate = load_audio(
            fp
        )  # Ensure to capture both audio and its sample rate
        processed_audio = apply_audio_effects(audio, sample_rate, best_params)
        path = os.path.join(anon_folder, os.path.basename(fp))
        sf.write(
            path, processed_audio, sample_rate
        )  # Use the original sample rate or a standardized one if needed

    print(f"All anonymized audio files stored in: {anon_folder}")
