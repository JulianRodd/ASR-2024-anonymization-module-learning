import glob
import os
import random

"""
example of flac file: r"'C:\\Users\\Harsh Yadav\\PycharmProjects\\dataset\\LibriSpeech\\train-clean-100\\19\\198\\19-198-0000.flac"
"""
import glob
import os


def get_librispeech_speaker_to_utterance(data_dir):
    speaker_to_utterance = {}
    # Search recursively for all .flac files within the data directory.
    flac_files = glob.glob(os.path.join(data_dir, "**", "*.flac"), recursive=True)

    for file in flac_files:
        # Use os.path to split the path and extract speaker and utterance IDs.
        path_parts = file.split(os.sep)
        speaker_id = path_parts[-3]
        utterance_id = os.path.splitext(path_parts[-1])[0]  # Remove the file extension.

        if speaker_id not in speaker_to_utterance:
            speaker_to_utterance[speaker_id] = []
        speaker_to_utterance[speaker_id].append(file)

    return speaker_to_utterance


def get_triplet(spk_to_utts):
    """Get a triplet of anchor/pos/neg samples."""
    pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    while len(spk_to_utts[pos_spk]) < 2:
        pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    anchor_utt, pos_utt = random.sample(spk_to_utts[pos_spk], 2)
    neg_utt = random.sample(spk_to_utts[neg_spk], 1)[0]
    return (anchor_utt, pos_utt, neg_utt)
