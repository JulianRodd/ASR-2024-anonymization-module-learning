import logging
import os
import pickle
import random

from datasets import load_dataset
from pydub import AudioSegment

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def normalize_sequence(seq):
    unique_values = sorted(set(seq))
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    normalized_seq = [mapping[value] for value in seq]
    logging.debug("Sequence normalized.")
    return normalized_seq

def map_gender_to_int(gender_seq):
    gender_mapping = {'M': 0, 'F': 1}
    return [gender_mapping[gender] for gender in gender_seq]

def normalize_strings(seq):
    unique_values = sorted(set(seq))
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    normalized_seq = [mapping[value] for value in seq]
    logging.debug("String sequence normalized.")
    return normalized_seq, mapping

def get_audio_data_wavs(CONFIG):
    n_speakers = CONFIG.N_SPEAKERS
    n_samples_per_speaker = CONFIG.N_SAMPLES_PER_SPEAKER
    gender = CONFIG.GENDER
    max_age = CONFIG.MAX_AGE
    min_age = CONFIG.MIN_AGE
    accent = CONFIG.ACCENT
    region = CONFIG.REGION
    random.seed(CONFIG.SEED)
    os.makedirs(CONFIG.CACHE_FOLDER, exist_ok=True)
    cache_file = os.path.join(
        CONFIG.CACHE_FOLDER,
        f"cache_{n_speakers}_{n_samples_per_speaker}_{gender}_{max_age}_{min_age}_{accent}_{region}.pkl".replace(
            " ", "_"
        ),
    )

    if os.path.exists(cache_file):
        logging.info(f"Loading data from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    dataset_path = os.path.join("data", "vctk")
    os.makedirs(dataset_path, exist_ok=True)

    logging.info("Downloading dataset...")
    dataset = load_dataset(
        "vctk", split="train", cache_dir=CONFIG.CACHE_FOLDER
    )

    filters = {
        "gender": gender,
        "max_age": max_age,
        "min_age": min_age,
        "accent": accent,
        "region": region,
    }
    logging.info(f"Total amount of samples: {len(dataset)}\n")
    logging.info("Applying filters...")
    # Applying filters based on the provided parameters
    for key, value in filters.items():
        if value:
            dataset = dataset.filter(lambda example: example[key] == value)
            logging.debug(f"Applied filter on {key} for {value}")

    logging.info(f"Filtered samples: {len(dataset)}\n")
    # make a distinct set of speaker_ids
    logging.info("Picking speakers...")
    speaker_ids = set(data["speaker_id"] for data in dataset)
    logging.info(f"Total speakers: {len(speaker_ids)}")
    speaker_ids = random.sample(sorted(speaker_ids), n_speakers)
    logging.info(f"Selected {len(speaker_ids)} speakers for anonymization.\n")
    # select target_speaker_samples from each speaker
    logging.debug(f"Selecting samples...")
    selected_data = []
    speaker_dataset = dataset.filter(
            lambda example: example["speaker_id"] in speaker_ids
        )
    for speaker_id in speaker_ids:
        speaker_data = speaker_dataset.filter(
            lambda example: example["speaker_id"] == speaker_id
        )
        selected_data.extend(random.sample(sorted(speaker_data, key=lambda x: x['speaker_id']), n_samples_per_speaker))

    logging.info(f"Total samples selected: {len(selected_data)}\n")
    file_paths = []
    transcriptions = []
    speakers = []
    gender_arr = []
    age_arr = []
    accent_arr = []
    region_arr = []

    print(f"Selected {len(selected_data)} samples for anonymization.\n")
    for data in selected_data:
        file_url = data["audio"]["path"]
        file_name = os.path.basename(file_url)
        destination_file_path = os.path.join(dataset_path, file_name)

        if not os.path.exists(destination_file_path):
            audio = AudioSegment.from_file(file_url)
            audio.export(destination_file_path, format="wav")
            logging.debug(f"Exported audio to {destination_file_path}")

        file_paths.append(destination_file_path)
        transcriptions.append(data["text"])
        speakers.append(data["speaker_id"])
        gender_arr.append(data["gender"])
        age_arr.append(data["age"])
        accent_arr.append(data["accent"])
        region_arr.append(data["region"])

    speakers = normalize_sequence(speakers)
    age_arr = normalize_sequence(age_arr)
    gender_arr = map_gender_to_int(gender_arr)
    accent_arr, accent_mapping = normalize_strings(accent_arr)
    region_arr, region_mapping = normalize_strings(region_arr)

    # Saving data to cache
    with open(cache_file, "wb") as f:
        pickle.dump((file_paths, transcriptions, speakers, gender_arr, age_arr, accent_arr, region_arr), f)
        logging.info(f"Data cached to {cache_file}")

    return file_paths, transcriptions, speakers, age_arr, gender_arr, accent_arr, region_arr
