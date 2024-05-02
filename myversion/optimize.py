import logging
import os

import optuna
import soundfile as sf
from asr import average_wer, load_pretrained_model, preprocess_audio, transcribe_audio
from asv import SpeakerVerificationModel, speaker_verification_loss
from mods import chorus, clipping, modspec_smoothing, resampling, vp_baseline2, vtln

from data import get_audio_data_wavs

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate_asr_and_asv(file_paths, transcriptions, speakers):
    """Evaluate ASR and ASV models and log results."""
    predictions = []
    for file in file_paths:
        waveform = preprocess_audio(file)
        prediction = transcribe_audio(processor, asr_model, waveform)
        predictions.append(prediction)

    predicted_speakers = speaker_verification.get_speaker(file_paths)
    combined_loss, avg_wer, accuracy  = calculate_combined_loss(
        transcriptions, predictions, speakers, predicted_speakers
    )
    logging.info("Evaluation complete.\n")

    logging.info(f"Average WER: {avg_wer}\n")
    logging.info(f"Actual Transcriptions: {transcriptions}")
    logging.info(f"Predicted Transcriptions: {predictions}\n\n")

    logging.info(f"Actual Speakers: {speakers}")
    logging.info(f"Predicted Speakers: {predicted_speakers}\n")
    logging.info(f"Speaker Verification Accuracy: {accuracy}\n\n")

    logging.info(f"Combined Loss: {combined_loss}\n\n\n")
    return predictions, predicted_speakers, avg_wer, accuracy, combined_loss


def calculate_combined_loss(
    transcriptions, predictions, speakers, pred_speakers, weight_asv=1.0
):
    """
    Calculate combined loss from ASR error and speaker verification accuracy.

    :param transcriptions: list of actual transcriptions
    :param predictions: list of predicted transcriptions
    :param speakers: list of actual speaker identities
    :param pred_speakers: list of predicted speaker identities
    :param weight_asv: weight assigned to the speaker verification loss
    :return: combined loss, maintaining high floating-point precision
    """
    # Calculate average Word Error Rate (ASR error)
    asr_error = average_wer(transcriptions, predictions)

    # Calculate speaker verification accuracy and apply the weighting factor
    speaker_accuracy = speaker_verification_loss(speakers, pred_speakers)

    weighted_speaker_accuracy = speaker_accuracy * weight_asv

    # Calculate combined loss ensuring arithmetic precision
    combined_loss = (asr_error + weighted_speaker_accuracy) / 2
    return combined_loss, asr_error, speaker_accuracy


def optimize_params(trial):
    """Define and apply hyperparameter optimization."""
    params = {
        "vtln": trial.suggest_float(
            "vtln", -0.01, 0.01
        ),  # Reduced range for Vocal Tract Length Normalization
        "resampling": trial.suggest_float(
            "resampling", 0.09, 0.11
        ),  # Closer to normal speed to avoid distorting the speech too much
        "vp_baseline2": trial.suggest_float(
            "vp_baseline2", 0.09, 0.11
        ),  # Subtle pitch and formant shifting
        "modspec": trial.suggest_float(
            "modspec", 0.005, 0.015
        ),  # Lesser modulation spectrum smoothing
        "clipping": trial.suggest_float(
            "clipping", 0.05, 0.07
        ),  # Avoid aggressive clipping to preserve dynamics
        "chorus": trial.suggest_float(
            "chorus", 0.0, 0.01
        ),  # Reduced effect depth to minimize voice 'doubling' which can confuse ASR
    }

    modified_audios = apply_audio_modifications(file_paths, params)

    _, _, _, _, combined_loss = evaluate_asr_and_asv(
        modified_audios, transcriptions, speakers
    )

    return combined_loss


def apply_audio_modifications(file_paths, params):
    """Apply audio effects based on parameters."""
    anonymized_audios = []
    for fp in file_paths:
        audio, _ = sf.read(fp)
        audio = vtln(audio, params["vtln"])
        audio = resampling(audio, params["resampling"])
        audio = vp_baseline2(audio, params["vp_baseline2"])
        audio = modspec_smoothing(audio, params["modspec"])
        audio = clipping(audio, params["clipping"])
        audio = chorus(audio, params["chorus"])
        anonymized_audios.append(audio)

    # Save anonymized audio files
    anon_folder = "working_anonymized_files"
    os.makedirs(anon_folder, exist_ok=True)
    for i, audio in enumerate(anonymized_audios):
        path = os.path.join(anon_folder, os.path.basename(file_paths[i]))
        sf.write(path, audio, 16000)
        anonymized_audios[i] = path

    return anonymized_audios


if __name__ == "__main__":
    file_paths, transcriptions, speakers = get_audio_data_wavs()
    processor, asr_model = load_pretrained_model()
    speaker_verification = SpeakerVerificationModel(num_speakers=len(set(speakers)))
    speaker_verification.finetune_model(speakers, file_paths, n_epochs=10)

    # Evaluate ASR and ASV models before hyperparameter optimization
    evaluate_asr_and_asv(file_paths, transcriptions, speakers)

    # Hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(optimize_params, n_trials=30)
    logging.info(
        f"Optimization complete. Best Parameters: {study.best_params}, Best Loss: {study.best_value}"
    )

    # Apply best parameters to audio files
    best_params = study.best_params
    modified_audios = apply_audio_modifications(file_paths, best_params)

    # store anonymized audio files
    anon_folder = "anonymized_files"
    os.makedirs(anon_folder, exist_ok=True)
    for file in modified_audios:
        audio, _ = sf.read(file)

        path = os.path.join(anon_folder, os.path.basename(file))
        sf.write(path, audio, 16000)
