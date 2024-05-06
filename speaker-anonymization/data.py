import os

from datasets import load_dataset
from pydub import AudioSegment


def normalize_sequence(seq):

    unique_values = sorted(set(seq))
    mapping = {value: idx for idx, value in enumerate(unique_values)}

    normalized_seq = [mapping[value] for value in seq]

    return normalized_seq


# categories in the VCTK dataset
  # 'age': '23',
  # 'gender': 'F' or 'M',
  # 'accent': 'English',
  # 'region': 'Southern England',
def get_audio_data_wavs(subset_size=500, gender = None, max_age = None, min_age = None, accent = None, region = None):


    dataset_path = "data/vctk"
    os.makedirs(dataset_path, exist_ok=True)

    dataset = load_dataset("vctk", split="train")

    if gender:
        dataset = dataset[dataset['gender'] == gender]
    if max_age:
        dataset = dataset[dataset['age'] <= max_age]
    if min_age:
        dataset = dataset[dataset['age'] >= min_age]
    if accent:
        dataset = dataset[dataset['accent'] == accent]
    if region:
        dataset = dataset[dataset['region'] == region]

    dataset = dataset.shuffle(seed=42).select(range(subset_size))

    file_paths = []
    transcriptions = []
    speakers = []

    for i, data in enumerate(dataset):
        file_url = data["audio"]["path"]
        file_name = os.path.basename(file_url)
        destination_file_path = os.path.join(dataset_path, file_name)

        if not os.path.exists(destination_file_path):
            audio = AudioSegment.from_file(file_url)
            audio.export(destination_file_path, format="wav")

        file_paths.append(destination_file_path)
        transcriptions.append(data["text"])
        speakers.append(data["speaker_id"])

    speakers = normalize_sequence(speakers)

    return file_paths, transcriptions, speakers


if __name__ == "__main__":
    file_paths, transcriptions, speakers = get_audio_data_wavs()
    print(f"Total audio files retrieved: {len(file_paths)}")
    print(
        "Sample audio file path:",
        file_paths[0] if file_paths else "No files downloaded.",
    )
    print(
        "Sample transcription:",
        transcriptions[0] if transcriptions else "No transcriptions available.",
    )
    print(
        "Sample speaker ID:", speakers[0] if speakers else "No speaker IDs available."
    )
