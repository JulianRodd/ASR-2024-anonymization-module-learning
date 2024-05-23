import logging
import os

import optuna
import torch
from pedalboard import (
    Bitcrush,
    Chorus,
    Distortion,
    Gain,
    HighpassFilter,
    LowpassFilter,
    Pedalboard,
    Phaser,
    PitchShift,
    time_stretch,
)

from speaker_anonymization.asr import (
    load_audio,
    load_pretrained_model,
    transcribe_audio,
)
from speaker_anonymization.data import get_audio_data_wavs
from speaker_anonymization.losses import calculate_combined_loss
from speaker_anonymization.spi import SpeakerIdentificationModel
from speaker_anonymization.utils import save_audio_file, save_optimization_plots


def optimize_audio_effects(CONFIG):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    os.makedirs(CONFIG.IMAGES_DIR, exist_ok=True)

    def apply_audio_effects(audio, sample_rate, params):
        """
        Apply audio effects based on pedalboard library with given parameters.
        Convert PyTorch tensor to numpy array for compatibility with the pedalboard library.
        """

        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        board = Pedalboard(
            [
                Distortion(drive_db=params["distortion_drive_db"]),
                PitchShift(semitones=params["pitch_shift_semitones"]),
                HighpassFilter(cutoff_frequency_hz=params["highpass_cutoff"]),
                LowpassFilter(cutoff_frequency_hz=params["lowpass_cutoff"]),
                Bitcrush(bit_depth=params["bitcrush_bit_depth"]),
                Chorus(rate_hz=params["chorus_rate_hz"]),
                Phaser(rate_hz=params["phaser_rate_hz"]),
                Gain(gain_db=params["gain_db"]),
            ],
        )
        processed_audio = board(audio, sample_rate=int(sample_rate))
        processed_audio = time_stretch(
            processed_audio, sample_rate, params["time_stretch_factor"]
        )
        return processed_audio

    def evaluate_asr_and_asv(audio_data, transcriptions, speakers, ages, genders, accents, regions):
        """Evaluate ASR and ASV models and log results using the modified audio."""
        predictions = []
        for waveform, sr in audio_data:

            prediction = transcribe_audio(processor, asr_model, waveform)
            predictions.append(prediction)

        predicted_speakers, predicted_ages, predicted_genders, predicted_accents, predicted_regions = speaker_identification.get_speakers_using_waveforms(
            [a for a, _ in audio_data]
        )
        combined_loss, avg_wer, speaker_accuracy, age_accuracy, gender_accuracy, accent_accuracy, region_accuracy = calculate_combined_loss(
            transcriptions,
            predictions,
            speakers,
            predicted_speakers,
            ages,
            predicted_ages,
            genders,
            predicted_genders,
            accents,
            predicted_accents,
            regions,
            predicted_regions,
            wer_weight=CONFIG.WER_WEIGHT,
            spi_weight=CONFIG.SPI_WEIGHT,
            age_weight=CONFIG.AGE_WEIGHT,
            gender_weight=CONFIG.GENDER_WEIGHT,
            accent_weight=CONFIG.ACCENT_WEIGHT,
            region_weight=CONFIG.REGION_WEIGHT
        )
        logging.info("Evaluation complete.\n")
        logging.info(f"Average WER: {avg_wer}\n")
        logging.info(f"Speaker Verification Accuracy: {speaker_accuracy}\n")
        logging.info(f"Gender Accuracy: {gender_accuracy}\n")
        logging.info(f"Age Accuracy: {age_accuracy}\n")
        logging.info(f"Accent Accuracy: {accent_accuracy}\n")
        logging.info(f"Region Verification Accuracy: {region_accuracy}\n")

        logging.info(f"Combined Loss: {combined_loss}\n\n\n")

        return predictions, predicted_speakers, avg_wer, speaker_accuracy, combined_loss

    def optimize_params(trial):
        """Define and apply hyperparameter optimization using normal distribution for parameters."""
        params = {
            "distortion_drive_db": trial.suggest_float(
                "distortion_drive_db",
                low=max(
                    0,
                    CONFIG.DISTORTION_DRIVE_DB_MEAN
                    - 2 * CONFIG.DISTORTION_DRIVE_DB_STD,
                ),
                high=CONFIG.DISTORTION_DRIVE_DB_MEAN
                + 2 * CONFIG.DISTORTION_DRIVE_DB_STD,
            ),
            "pitch_shift_semitones": trial.suggest_float(
                "pitch_shift_semitones",
                low=CONFIG.PITCH_SHIFT_SEMITONES_MEAN
                - 2 * CONFIG.PITCH_SHIFT_SEMITONES_STD,
                high=min(
                    0,
                    CONFIG.PITCH_SHIFT_SEMITONES_MEAN
                    + 2 * CONFIG.PITCH_SHIFT_SEMITONES_STD,
                ),
            ),
            "highpass_cutoff": trial.suggest_float(
                "highpass_cutoff",
                low=max(
                    0, CONFIG.HIGHPASS_CUTOFF_MEAN - 2 * CONFIG.HIGHPASS_CUTOFF_STD
                ),
                high=CONFIG.HIGHPASS_CUTOFF_MEAN + 2 * CONFIG.HIGHPASS_CUTOFF_STD,
            ),
            "lowpass_cutoff": trial.suggest_float(
                "lowpass_cutoff",
                low=CONFIG.LOWPASS_CUTOFF_MEAN - 2 * CONFIG.LOWPASS_CUTOFF_STD,
                high=CONFIG.LOWPASS_CUTOFF_MEAN + 2 * CONFIG.LOWPASS_CUTOFF_STD,
            ),
            "time_stretch_factor": trial.suggest_float(
                "time_stretch_factor",
                low=CONFIG.TIME_STRETCH_FACTOR_MEAN
                - 2 * CONFIG.TIME_STRETCH_FACTOR_STD,
                high=CONFIG.TIME_STRETCH_FACTOR_MEAN
                + 2 * CONFIG.TIME_STRETCH_FACTOR_STD,
            ),
            "bitcrush_bit_depth": trial.suggest_int(
                "bitcrush_bit_depth",
                low=CONFIG.BITCRUSH_BIT_DEPTH_MEAN
                - 2 * CONFIG.BITCRUSH_BIT_DEPTH_STD,
                high=CONFIG.BITCRUSH_BIT_DEPTH_MEAN
                + 2 * CONFIG.BITCRUSH_BIT_DEPTH_STD,
            ),
            "chorus_rate_hz": trial.suggest_float(
                "chorus_rate_hz",
                low=CONFIG.CHORUS_RATE_HZ_MEAN - 2 * CONFIG.CHORUS_RATE_HZ_STD,
                high=CONFIG.CHORUS_RATE_HZ_MEAN + 2 * CONFIG.CHORUS_RATE_HZ_STD,
            ),
            "phaser_rate_hz": trial.suggest_float(
                "phaser_rate_hz",
                low=CONFIG.PHASER_RATE_HZ_MEAN - 2 * CONFIG.PHASER_RATE_HZ_STD,
                high=CONFIG.PHASER_RATE_HZ_MEAN + 2 * CONFIG.PHASER_RATE_HZ_STD,
            ),
            "gain_db": trial.suggest_float(
                "gain_db",
                low=CONFIG.GAIN_DB_MEAN - 2 * CONFIG.GAIN_DB_STD,
                high=CONFIG.GAIN_DB_MEAN + 2 * CONFIG.GAIN_DB_STD,
            ),
        }
        audio_data = []
        for file_path in file_paths:
            audio, sr = load_audio(file_path)
            processed_audio = apply_audio_effects(audio, float(sr), params)
            audio_data.append((processed_audio, sr))

        _, _, avg_wer, accuracy, combined_loss = evaluate_asr_and_asv(
            audio_data, transcriptions, speakers, ages, genders, accents, regions
        )

        trial.set_user_attr("avg_wer", avg_wer)
        trial.set_user_attr("speaker_accuracy", accuracy)
        trial.set_user_attr("combined_loss", combined_loss)

        return combined_loss

    logging.info("\n\nStarting audio effects optimization...\n\n")

    logging.info("Loading data...\n")
    file_paths, transcriptions, speakers, ages, genders, accents, regions = get_audio_data_wavs(
        CONFIG,
    )
    num_speakers = len(set(speakers))

    logging.info("Loading ASR model...\n")
    processor, asr_model = load_pretrained_model(CONFIG)

    logging.info("Loading Speaker Identification model...\n")
    speaker_identification = SpeakerIdentificationModel(
        num_speakers=num_speakers, CONFIG=CONFIG
    )

    logging.info("Finetuning Speaker Identification model...\n")
    speaker_identification.finetune_model(
        speakers, ages, genders, accents, regions, file_paths, n_epochs=CONFIG.SPEAKER_IDENTIFICATION_EPOCHS
    )
    logging.info("Speaker Identification model trained.\n\n")

    logging.info("Evaluating the initial models...\n")
    initial_audio_data = []
    for file_path in file_paths:
        audio, sr = load_audio(file_path)
        initial_audio_data.append((audio, sr))
    _, _, avg_wer, accuracy, combined_loss = evaluate_asr_and_asv(
        initial_audio_data, transcriptions, speakers, ages, genders, accents, regions
    )

    logging.info(f"Starting audio parameter optimization...\n")
    study = optuna.create_study(
        direction="minimize",
        study_name=CONFIG.STUDY_NAME,
        storage=CONFIG.STORAGE_NAME,
        load_if_exists=CONFIG.LOAD_IF_EXISTS,
    )
    study.optimize(
        optimize_params,
        n_trials=CONFIG.NUM_TRIALS,
        show_progress_bar=CONFIG.SHOW_PROGRESS_BAR,
        n_jobs=CONFIG.CONFIG_N_JOBS,
    )
    logging.info(
        f"Optimization complete. Best Parameters: {study.best_params}, Best Loss: {study.best_value}\n"
    )

    logging.info("Saving optimization plots...\n")
    images_dir = (
        f"images/{study.study_name}_{str(num_speakers)}_speakers_{study.best_value:.2f}"
    )
    os.makedirs(images_dir, exist_ok=True)

    save_optimization_plots(study, images_dir)


    logging.info("Anonymizing audio files using the best parameters...\n")
    best_params = study.best_params
    anon_folder = f"{CONFIG.ANONYMIZED_FOLDER}/{CONFIG.STUDY_NAME}"
    os.makedirs(anon_folder, exist_ok=True)

    for fp in file_paths:
        audio, sample_rate = load_audio(fp)
        processed_audio = apply_audio_effects(audio, sample_rate, best_params)
        path = os.path.join(
            anon_folder, os.path.splitext(os.path.basename(fp))[0] + ".wav"
        )
        save_audio_file(processed_audio, path, sample_rate)

    print(f"All anonymized audio files stored in: {anon_folder}")
