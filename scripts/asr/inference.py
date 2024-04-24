
import speech_recognition as sr

from jiwer import wer



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
