import os

import numpy as np
import optuna
import soundfile as sf
from asr import ASRModel, asr_loss
from asv import SpeakerVerificationModel, speaker_verification_loss
from tqdm import tqdm
from mods import (
    chorus,
    clipping,
    modspec_smoothing,
    resampling,
    vp_baseline2,
    vtln,
)

from data import get_audio_data_wavs

# 1. Load data
file_paths, ground_truth_transcriptions = get_audio_data_wavs()

# 2. After /p before _ is the speaker ID, subtract 255 and cast to int
speakers = [int(file.split("/")[-1].split("_")[0][1:]) - 225 for file in file_paths]

# 3. Initialize models
asr = ASRModel()
speaker_verification = SpeakerVerificationModel(num_speakers=len(set(speakers)))

# 4. Fine-tune the Speaker Verification model
speaker_verification.finetune_model(speakers, file_paths)

# 5. Fine-tune the ASR model
asr.finetune_model(ground_truth_transcriptions, file_paths)

# 6. Define starting hyperparameters
hyperparams = {
    "vtln": 0.0,
    "resampling": 1.0,
    "vp_baseline2": 1.0,
    "modspec": 0.1,
    "clipping": 0.5,
    "chorus": 0.1,
}


# 7. Define a combined loss function
def combined_loss(transcriptions, predictions, speakers, pred_speakers, weight_asv=1.0):
    print("Calculating combined loss...")
    print("Transcriptions:", transcriptions)
    print("Predictions:", predictions)
    print("Speakers:", speakers)
    print("Predicted Speakers:", pred_speakers)
    asr_error = asr_loss(transcriptions, predictions)
    speaker_accuracy = speaker_verification_loss(speakers, pred_speakers) * weight_asv
    return asr_error + (1 - speaker_accuracy)


# 8. Learn hyperparameters
def optimize_params(trial):
    # Suggesting parameters
    params = {
        "vtln": trial.suggest_float("vtln", -0.2, 0.2),
        "resampling": trial.suggest_float("resampling", 0.7, 1.3),
        "vp_baseline2": trial.suggest_float("vp_baseline2", 0.7, 1.3),
        "modspec": trial.suggest_float("modspec", 0.05, 0.3),
        "clipping": trial.suggest_float("clipping", 0.3, 1.0),
        "chorus": trial.suggest_float("chorus", 0.0, 0.2),
    }

    # Apply voice modifications
    anonymized_audios = [
        chorus(
            clipping(
                modspec_smoothing(
                    vp_baseline2(
                        resampling(
                            vtln(sf.read(fp)[0], params["vtln"]), params["resampling"]
                        ),
                        params["vp_baseline2"],
                    ),
                    params["modspec"],
                ),
                params["clipping"],
            ),
            params["chorus"],
        )
        for fp in file_paths
    ]

    # Save anonymized audio files
    anon_folder = "anonymized_files"
    os.makedirs(anon_folder, exist_ok=True)
    anon_paths = [
        os.path.join(anon_folder, f"{os.path.basename(fp)}") for fp in file_paths
    ]
    for apath, audio in zip(anon_paths, anonymized_audios):
        sf.write(apath, audio, 16000)

    # Transcribe anonymized audio
    anon_transcriptions = [asr.transcribe_wav_file(ap) for ap in anon_paths]

    # Re-predict speakers from anonymized audio
    pred_speakers = speaker_verification.get_speaker(anon_paths)

    # Compute combined loss
    loss = combined_loss(ground_truth_transcriptions, anon_transcriptions, speakers, pred_speakers)
    return loss


study = optuna.create_study(direction="minimize")
study.optimize(optimize_params, n_trials=50)

# 9. Save optimized parameters
best_params = study.best_params
print("Best Parameters:", best_params)

# 10. Final output
print("Optimization complete. Best loss:", study.best_value)
