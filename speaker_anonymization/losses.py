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
    transcriptions, predictions, speakers, pred_speakers, wer_weight=0.7, spi_weight=0.3
):
    asr_error = average_wer(transcriptions, predictions)

    asr_error = min(1.0, max(0.0, asr_error))

    speaker_accuracy = speaker_verification_loss(speakers, pred_speakers)

    combined_loss = (wer_weight * asr_error) + (spi_weight * speaker_accuracy)

    return combined_loss, asr_error, speaker_accuracy
