def get_audio_data_wavs():
    # dataset_path = 'data/huggingface/wavs'
    # os.makedirs(dataset_path, exist_ok=True)  # Ensure the directory exists

    # # Load a dataset from Hugging Face - using a subset of LibriSpeech for example purposes
    # dataset = load_dataset('librispeech_asr', 'clean', split='train.100')

    # # Select a subset to get about 500 samples, for example purposes we take the first 500
    # dataset = dataset.select(range(min(500, len(dataset))))

    # file_paths = []
    # for i, data in enumerate(dataset):
    #     file_url = data['file']
    #     file_name = os.path.basename(file_url)
    #     destination_file_path = os.path.join(dataset_path, file_name)

    #     if not os.path.exists(destination_file_path):  # Only download if it doesn't exist
    #         torchaudio.save(destination_file_path, data['audio']['array'], data['audio']['sampling_rate'])

    #     file_paths.append(destination_file_path)

    # return file_paths
    wavs = [
        "data/vctk/p225_001.wav",
        "data/vctk/p225_002.wav",
        "data/vctk/p226_001.wav",
        "data/vctk/p226_002.wav",
        "data/vctk/p227_001.wav",
    ]
    transcriptions = [
        "please call Stella",
        "ask her to bring these things with her from the store",
        "please call Stella",
        "ask her to bring these things with her from the store",
        "please call Stella",
    ]
    return wavs, transcriptions


# Example of usage
if __name__ == "__main__":
    audio_files = get_audio_data_wavs()
    print(f"Total audio files retrieved: {len(audio_files)}")
    print(
        "Sample audio file path:",
        audio_files[0] if audio_files else "No files downloaded.",
    )
