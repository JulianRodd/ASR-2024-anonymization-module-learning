import logging
import os

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns
import soundfile as sf
from asr import average_wer, load_pretrained_model, transcribe_audio
from asv import SpeakerVerificationModel, speaker_verification_loss
from pedalboard import Distortion, HighpassFilter, LowpassFilter, Pedalboard, PitchShift

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
    """
    board = Pedalboard(
        [
            Distortion(drive_db=params["distortion_drive_db"]),
            PitchShift(semitones=params["pitch_shift_semitones"]),
            HighpassFilter(cutoff_frequency_hz=params["highpass_cutoff"]),
            LowpassFilter(cutoff_frequency_hz=params["lowpass_cutoff"]),
        ]
    )
    return board(audio, sample_rate=sample_rate)


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
    # logging.info(f"Actual Transcriptions: {transcriptions}")
    # logging.info(f"Predicted Transcriptions: {predictions}\n\n")

    logging.info(f"Actual Speakers: {speakers}")
    logging.info(f"Predicted Speakers: {predicted_speakers}\n")
    logging.info(f"Speaker Verification Accuracy: {accuracy}\n\n")

    logging.info(f"Combined Loss: {combined_loss}\n\n\n")

    return predictions, predicted_speakers, avg_wer, accuracy, combined_loss


def calculate_combined_loss(
    transcriptions, predictions, speakers, pred_speakers, wer_weight=0.7, asv_weight=0.3
):
    """
    Calculate a combined loss from ASR error (WER) and speaker verification accuracy designed to maximize WER and minimize ASV.

    Parameters:
        transcriptions (list): List of actual transcriptions.
        predictions (list): List of predicted transcriptions.
        speakers (list): List of actual speaker identities.
        pred_speakers (list): List of predicted speaker identities.
        wer_weight (float): Weight assigned to the ASR WER loss, aiming to maximize it.
        asv_weight (float): Weight assigned to the speaker verification loss, aiming to minimize it.

    Returns:
        combined_loss (float): The combined loss calculated to maximize WER and minimize ASV.
        asr_error (float): Average Word Error Rate.
        speaker_accuracy (float): Speaker verification accuracy.
    """
    # Calculate average Word Error Rate (WER)
    asr_error = average_wer(transcriptions, predictions)
    # Normalize and square the WER to increase the impact as it approaches 1
    asr_error = min(1.0, max(0.0, asr_error))

    # The closer the WER is to 1, the higher the negative impact
    scaled_wer = asr_error ** 2
    # Calculate speaker verification accuracy
    speaker_accuracy = speaker_verification_loss(speakers, pred_speakers)

    # Calculate combined loss:
    # Negative impact increases with higher WER due to squaring
    combined_loss = (wer_weight * scaled_wer) + (asv_weight * speaker_accuracy)

    return combined_loss, asr_error, speaker_accuracy


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
        audio, sr = sf.read(file_path)
        processed_audio = apply_audio_effects(audio, sr, params)
        audio_data.append((processed_audio, sr))

    _, _, avg_wer, accuracy, combined_loss = evaluate_asr_and_asv(
        audio_data, transcriptions, speakers
    )

    trial.set_user_attr("avg_wer", avg_wer)
    trial.set_user_attr("speaker_accuracy", accuracy)
    trial.set_user_attr("combined_loss", combined_loss)
    return combined_loss


def save_optimization_plots(study, images_dir):
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(images_dir, "optimization_history.png"))

    # Plot parameter importances
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(images_dir, "param_importances.png"))

    # Plot slice of each parameter
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(os.path.join(images_dir, "slice_plot.png"))

    # Optionally, save contour plot of interactions of each pair of parameters if you have many parameters
    if len(study.best_params) > 1:
        fig = optuna.visualization.plot_contour(study)
        fig.update_layout(
            autosize=False,
            width=1200,  # Width in pixels
            height=800,  # Height in pixels
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
            font=dict(size=10),
        )
        fig.write_image(os.path.join(images_dir, "contour_plot.png"))

    results = []
    for trial in study.trials:
        results.append(
            {
                "Trial": trial.number,
                "Average WER": trial.user_attrs["avg_wer"],
                "1 - Speaker Accuracy": 1 - trial.user_attrs["speaker_accuracy"],
                "Combined Loss": trial.user_attrs["combined_loss"],
            }
        )

    results_df = pd.DataFrame(results)

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results_df, x="Trial", y="Average WER", marker="o", label="Average WER"
    )
    sns.lineplot(
        data=results_df,
        x="Trial",
        y="1 - Speaker Accuracy",
        marker="o",
        label="1 - Speaker Accuracy",
    )
    sns.lineplot(
        data=results_df, x="Trial", y="Combined Loss", marker="o", label="Combined Loss"
    )
    plt.xlabel("Trial")
    plt.ylabel("Metric Value")
    plt.title("Effects of Audio Tweaks on Metrics")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(images_dir, "metrics_plot.png")
    plt.savefig(plot_path)

    print("Saved all optimization plots to:", images_dir)


if __name__ == "__main__":
    logging.info("Starting audio effects optimization...\n")
    file_paths, transcriptions, speakers = get_audio_data_wavs(subset_size= 100)
    num_speakers = len(set(speakers))
    processor, asr_model = load_pretrained_model()
    speaker_verification = SpeakerVerificationModel(num_speakers=num_speakers)
    speaker_verification.finetune_model(speakers, file_paths, n_epochs=1)
    logging.info("Speaker Verification model trained.\n\n")
    # Evaluate the initial model
    logging.info("Evaluating the initial model...\n")
    initial_audio_data = []
    for file_path in file_paths:
        audio, sr = sf.read(file_path)
        initial_audio_data.append((audio, sr))
    _, _, avg_wer, accuracy, combined_loss = evaluate_asr_and_asv(
        initial_audio_data, transcriptions, speakers
    )
    # Hyperparameter optimization
    study = optuna.create_study(
        direction="minimize", study_name="optimizing_audio_effects_for_anonymization"
    )
    study.optimize(optimize_params, n_trials=2)
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
        audio, sample_rate = sf.read(
            fp
        )  # Ensure to capture both audio and its sample rate
        processed_audio = apply_audio_effects(audio, sample_rate, best_params)
        path = os.path.join(anon_folder, os.path.basename(fp))
        sf.write(
            path, processed_audio, sample_rate
        )  # Use the original sample rate or a standardized one if needed

    print(f"All anonymized audio files stored in: {anon_folder}")
