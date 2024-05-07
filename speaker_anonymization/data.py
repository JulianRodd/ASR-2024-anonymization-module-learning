import os
import pickle
import random
from collections import defaultdict
from datasets import load_dataset
from pydub import AudioSegment
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_sequence(seq):
    unique_values = sorted(set(seq))
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    normalized_seq = [mapping[value] for value in seq]
    logging.debug("Sequence normalized.")
    return normalized_seq

def get_audio_data_wavs(
    CONFIG,
    subset_size=500,
    gender=None,
    max_age=None,
    min_age=None,
    accent=None,
    region=None,
):
    min_samples_per_speaker = subset_size // 10
    os.makedirs(CONFIG.CACHE_FOLDER, exist_ok=True)
    cache_file = os.path.join(CONFIG.CACHE_FOLDER, f"cache_{subset_size}_{gender}_{max_age}_{min_age}_{accent}_{region}.pkl".replace(" ", "_"))

    if os.path.exists(cache_file):
        logging.info(f"Loading data from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    dataset_path = os.path.join("data", "vctk")
    os.makedirs(dataset_path, exist_ok=True)

    logging.info("Downloading dataset...")
    dataset = load_dataset("vctk", split="train")

    filters = {
        'gender': gender,
        'max_age': max_age,
        'min_age': min_age,
        'accent': accent,
        'region': region
    }

    # Applying filters based on the provided parameters
    for key, value in filters.items():
        if value:
            dataset = dataset.filter(lambda example: example[key] == value)
            logging.debug(f"Applied filter on {key} for {value}")

    # Grouping data by speaker
    dataset = dataset.shuffle(seed=42)
    speaker_data = defaultdict(list)
    for item in dataset:
        speaker_data[item["speaker_id"]].append(item)

    # Selecting data to ensure at least 'min_samples_per_speaker' samples per speaker
    selected_data = []
    for speaker, items in speaker_data.items():
        if len(items) >= min_samples_per_speaker:
            selected_data.extend(random.sample(items, min_samples_per_speaker))
            logging.debug(f"Selected {min_samples_per_speaker} samples for speaker {speaker}")

    # Ensure the total subset size is met
    all_items = [item for sublist in speaker_data.values() for item in sublist]
    while len(selected_data) < subset_size and all_items:
        item = random.choice(all_items)
        if item not in selected_data:
            selected_data.append(item)
            logging.debug(f"Added extra item to selected data to meet subset size.")

    file_paths = []
    transcriptions = []
    speakers = []

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

    speakers = normalize_sequence(speakers)

    # Saving data to cache
    with open(cache_file, "wb") as f:
        pickle.dump((file_paths, transcriptions, speakers), f)
        logging.info(f"Data cached to {cache_file}")

    return file_paths, transcriptions, speakers
