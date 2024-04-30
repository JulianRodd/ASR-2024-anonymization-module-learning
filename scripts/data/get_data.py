from datasets import load_dataset

dataset = load_dataset("librispeech_asr",
                       "clean",
                       split="test")

print(dataset)