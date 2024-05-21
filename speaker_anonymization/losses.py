import jiwer
from sklearn.metrics import accuracy_score


def speaker_verification_loss(real_speakers, predicted_speakers):
    accuracy = accuracy_score(real_speakers, predicted_speakers)
    return accuracy


def normalized_wer(actual, predicted):
    transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
        ]
    )

    if not actual.strip() or not predicted.strip():
        return 1.0

    measures = jiwer.compute_measures(
        actual,
        predicted,
        truth_transform=transformation,
        hypothesis_transform=transformation,
    )

    wer = measures["wer"]
    return wer


def average_wer(actual_transcriptions, predicted_transcriptions):
    """
    Calculate the average Word Error Rate (WER) precisely, handling edge cases.
    """
    if (
        len(actual_transcriptions) != len(predicted_transcriptions)
        or not actual_transcriptions
    ):
        raise ValueError("Transcription lists are empty or of unequal length.")

    wers = [
        normalized_wer(actual, predicted) if actual and predicted else 1.0
        for actual, predicted in zip(actual_transcriptions, predicted_transcriptions)
    ]

    return sum(wers) / len(wers)


def calculate_combined_loss(
    transcriptions, predictions, speakers, pred_speakers, ages, pred_ages, genders, pred_genders, accents, pred_accents, regions, pred_regions, wer_weight=0.5, spi_weight=0.2, gender_weight=0.1, age_weight=0.1, accent_weight=0.05, region_weight=0.05
):
    asr_error = average_wer(transcriptions, predictions)

    asr_error = min(1.0, max(0.0, asr_error))

    speaker_accuracy = speaker_verification_loss(speakers, pred_speakers)
    
    age_accuracy = speaker_verification_loss(ages, pred_ages)
    
    gender_accuracy = speaker_verification_loss(genders, pred_genders)
    
    accent_accuracy = speaker_verification_loss(accents, pred_accents)
    
    region_accuracy = speaker_verification_loss(regions, pred_regions)

    combined_loss = (wer_weight * asr_error) + (spi_weight * speaker_accuracy) + (gender_weight * gender_accuracy) + (age_weight * age_accuracy) + (accent_weight * accent_accuracy) + (region_weight * region_accuracy)

    return combined_loss, asr_error, speaker_accuracy, age_accuracy, gender_accuracy, accent_accuracy, region_accuracy
