#!/usr/bin/env python3
import copy
import json
import shutil
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import optuna as op
import soundfile as sf
import speech_recognition as sr

from jiwer import wer

# multiprocess
from joblib import Parallel, delayed
from voice_modification import (
    chorus,
    clipping,
    modspec_smoothing,
    resampling,
    vp_baseline2,
    vtln,
)


def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        # Log the error or handle it as appropriate
        return ""


def loss_asr(lst_fn, ground_truth_texts):
    total_wer = 0
    for audio_file, ground_truth in zip(lst_fn, ground_truth_texts):
        transcribed_text = transcribe_audio(audio_file)
        total_wer += wer(ground_truth, transcribed_text)
    average_wer = total_wer / len(lst_fn)
    return average_wer


def simulate_asv_system(audio_file, speaker_models):
    """
    Simulate the processing of an audio file through an ASV system.
    This function is a placeholder for actual ASV system processing.

    Args:
        audio_file (str): Path to the audio file.
        speaker_models (dict): A dictionary of speaker models.

    Returns:
        dict: A dictionary containing 'genuine' and 'impostor' scores for the audio file.
    """
    # This is a dummy implementation and should be replaced with actual ASV system calls.
    # Assume each output score is a random value between 0 and 1,
    # where a higher score means higher similarity.
    import random
    genuine_score = random.random()
    impostor_scores = [random.random() for _ in speaker_models]
    
    return {'genuine': genuine_score, 'impostor': max(impostor_scores)}

def compute_eer(scores):
    """
    Compute the Equal Error Rate (EER) from ASV system scores.

    Args:
        scores (list): A list of dictionaries with 'genuine' and 'impostor' scores.

    Returns:
        float: The EER value.
    """
    from sklearn.metrics import roc_curve
    y_true = [1] * len(scores) + [0] * len(scores)  # 1 for genuine, 0 for impostor
    y_scores = [score['genuine'] for score in scores] + [score['impostor'] for score in scores]

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    
    return eer

def loss_asv(lst_fn, speaker_models):
    """
    Compute the negative EER for a list of anonymized audio files.

    Args:
        lst_fn (list): List of filenames of anonymized audio.
        speaker_models (dict): A dictionary of speaker models.

    Returns:
        float: The negative EER (i.e., 1 - EER).
    """
    asv_system_results = [simulate_asv_system(fn, speaker_models) for fn in lst_fn]
    eer = compute_eer(asv_system_results)
    
    return 1.0 - eer


# cascaded voice modification modules
def anonymize(x, fs=16000, **kwargs):
    """anonymize speech using model parameters

    Args:
      x (list): waveform data
      fs (int): sampling frequency [Hz]
      kargs (dic): model parameters
    Returns:
      y (list): anonymized waveform data
    """

    module = {
        "vtln": vtln,
        "resamp": resampling,
        "mcadams": vp_baseline2,
        "modspec": modspec_smoothing,
        "clip": clipping,
        "chorus": chorus,
    }

    y = copy.deepcopy(x)
    for k, v in kwargs.items():  # cascaded modules
        y = module[k](y, v) if k != "resamp" else module[k](y, v, fs=fs)

    return y


def objective(trial, params, wav, weight_asv=1.0, fs=16000, n_jobs=-1, tempdir="temp"):
    """objective function for anonymization

    Args:
      trial (optuna.trial): trial variable of Optuna
      wav (dic): waveform data (value) and its basename (key)
      weight_asv (float): weight value of EER objective
      fs (int): sampling frequency
      n_jobs (int): the number of CPU threads during compution
      tempdir (str): dirname to save anonymized waveform
    Returns:
      score (float): objective value
    """

    # make dir
    Path(tempdir).mkdir(exist_ok=True, parents=True)

    # setup parameters
    params = {k: trial.suggest_uniform(k, v[0], v[1]) for k, v in params.items()}

    # anonymize
    wav_anon = Parallel(n_jobs=n_jobs)(
        [delayed(anonymize)(w, fs, **params) for w in wav.values()]
    )

    # save anonymize speech
    fn_lst = [str(Path(tempdir).joinpath(f + ".wav")) for f in wav.keys()]
    fn_lst_ground_truth = ['please call stella']
    Parallel(n_jobs=n_jobs)(
        [delayed(sf.write)(f, w, fs, "PCM_16") for f, w in zip(fn_lst, wav_anon)]
    )

    # compute total score
    score = loss_asr(fn_lst, fn_lst_ground_truth) + weight_asv * loss_asv(fn_lst)

    # remove dir
    shutil.rmtree(tempdir)

    return score


if __name__ == "__main__":
    # hyperparameters
    hparams = {
        "n_trial": 50,  # the number of trials for optimization
        "weight_asv": 1.0,  # weight of ASV objective
        "fs": 16000,  # sampling frequency
        "tempdir": "work",  # dirname to save anonymized speech for computing WER and EER
        "anon_params": {  # anonymization parameter and its search range.
            "vtln": [-0.2, 0.2],  # vocal tract length normalization
            # "resamp": [0.7, 1.3],   # resampling
            # "mcadams": [0.7, 1.3],  # McAdams transformation
            # "modspec": [0.05, 0.3],# modulation spectrum smoothing
            # "clip": [0.3, 1.0],    # clipping
            # "chorus": [0.0, 0.2]   # chorus effect
        },
    }

    # load training data (its basename [fn.stem] have speaker label.)
    wav = {
        fn.stem: librosa.load(fn, sr=hparams["fs"])[0]
        for fn in Path("data/vctk").glob("*.wav")
    }

    # optimize
    study = op.create_study()
    study.optimize(
        partial(
            objective,
            wav=wav,
            params=hparams["anon_params"],
            weight_asv=hparams["weight_asv"],
            fs=hparams["fs"],
            tempdir=hparams["tempdir"],
        ),
        n_trials=hparams["n_trial"],
    )
    best_param = copy.copy(study.best_params)

    # save optimized model parameters
    fn_model = "params/sample.json"
    print("save optimized model parameters to {}".format(fn_model))
    json.dump(best_param, open(fn_model, "w"), indent=2)
