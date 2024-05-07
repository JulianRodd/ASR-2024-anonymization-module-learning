import soundfile as sf
from pedalboard import (
    Chorus,
    Compressor,
    Distortion,
    Gain,
    Pedalboard,
    PitchShift,
    Reverb,
)


# Example functions using the pedalboard library
def apply_reverb(x, sample_rate, room_size=0.5):
    reverb = Reverb(room_size=room_size)
    return reverb(x, sample_rate)


def apply_compressor(x, sample_rate):
    compressor = Compressor(threshold_db=-20, ratio=4)
    return compressor(x, sample_rate)


def apply_chorus(x, sample_rate, rate=1.5):
    chorus = Chorus(rate_hz=rate)
    return chorus(x, sample_rate)


def apply_pitch_shift(x, sample_rate, semitones=1):
    pitch_shift = PitchShift(semitones=semitones)
    return pitch_shift(x, sample_rate)


def apply_gain(x, sample_rate, gain_db=-3.0):
    gain = Gain(gain_db=gain_db)
    Distortion()
    return gain(x, sample_rate)


def process_audio_effects(
    file_path,
    effects=[
        # PitchShift(semitones=-10),
        # Compressor(threshold_db=-20, ratio=4),
        Reverb(room_size=0.5),
        # Chorus(rate_hz=0),
        # Distortion(drive_db=40),
        #   Phaser(rate_hz=60 ),
        # HighpassFilter(cutoff_frequency_hz=20),
        # LowpassFilter(cutoff_frequency_hz=3000),
        # Phaser(rate_hz=1.5),
    ],
):
    audio, sr = sf.read(file_path)
    board = Pedalboard(effects)

    # Ensure the sample_rate is correctly passed
    processed_audio = board(audio, sample_rate=sr)
    return processed_audio, sr


# Usage example
if __name__ == "__main__":
    input_file = "data/vctk/p225_001.wav"
    output_file = "processed_audio.wav"
    processed_audio, sr = process_audio_effects(input_file)
    sf.write(output_file, processed_audio, sr)
