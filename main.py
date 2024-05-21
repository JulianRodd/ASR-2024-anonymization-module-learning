from multiprocessing import Pool, cpu_count

from speaker_anonymization.config import Config
from speaker_anonymization.optimize import optimize_audio_effects
from speaker_anonymization.data import get_audio_data_wavs


def run_optimization(config):
    optimize_audio_effects(config)


def run_optimizations(configs):
    # Repeat each config to fill up 50% of the CPU cores
    configs = configs * (cpu_count() // 2 // len(configs))

    with Pool() as pool:
        pool.map(optimize_audio_effects, configs)


if __name__ == "__main__":
    print("Ik vertouw mezelf niet om op een knop te drukken en wil bevestiging dat dit ding gestart is dankuwel")

    BASE_CONFIG = Config(n_speakers=2, n_samples_per_speaker=2)
    #FEMALE_ONLY_CONFIG = Config(gender="F")
    #MALE_ONLY_CONFIG = Config(num_trials=1, gender="M")
    
    # _, _, speakers, age, gender, accent, region = get_audio_data_wavs(BASE_CONFIG)
    # print(speakers)
    # print(gender)
    # print(age)
    # print(accent)
    # print(region)

    configs = [BASE_CONFIG]
    run_optimization(BASE_CONFIG)
