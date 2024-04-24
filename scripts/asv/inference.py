import numpy as np


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

    import random

    genuine_score = random.random()
    impostor_scores = [random.random() for _ in speaker_models]

    return {"genuine": genuine_score, "impostor": max(impostor_scores)}


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
    y_scores = [score["genuine"] for score in scores] + [
        score["impostor"] for score in scores
    ]

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
