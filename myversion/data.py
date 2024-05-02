import os
from datasets import load_dataset
from pydub import AudioSegment

def get_audio_data_wavs():
    dataset_path = 'data/vctk'
    os.makedirs(dataset_path, exist_ok=True)  # Ensure the directory exists

    # Load the dataset from Hugging Face
    dataset = load_dataset('vctk', split='train')

    # We'll take a subset for example purposes, e.g., first 100 samples
    dataset = dataset.select(range(min(100, len(dataset))))

    file_paths = []
    transcriptions = []
    speakers = []

    for i, data in enumerate(dataset):
        file_url = data['audio']['path']
        file_name = os.path.basename(file_url)
        destination_file_path = os.path.join(dataset_path, file_name)

        # Convert audio if necessary and save locally
        if not os.path.exists(destination_file_path):
            audio = AudioSegment.from_file(file_url)
            audio.export(destination_file_path, format="wav")

        file_paths.append(destination_file_path)
        transcriptions.append(data['text'])
        speakers.append(data['speaker_id'])

    return file_paths, transcriptions, speakers

# Example of usage
if __name__ == "__main__":
    file_paths, transcriptions, speakers = get_audio_data_wavs()
    print(f"Total audio files retrieved: {len(file_paths)}")
    print("Sample audio file path:", file_paths[0] if file_paths else "No files downloaded.")
    print("Sample transcription:", transcriptions[0] if transcriptions else "No transcriptions available.")
    print("Sample speaker ID:", speakers[0] if speakers else "No speaker IDs available.")
