from multiprocessing import Pool, cpu_count

from speaker_anonymization.config import Config
from speaker_anonymization.optimize import optimize_audio_effects


def run_optimization(config):
    optimize_audio_effects(config)


def run_optimizations(configs):
    # Repeat each config to fill up 50% of the CPU cores
    configs = configs * (cpu_count() // 2 // len(configs))

    with Pool() as pool:
        pool.map(optimize_audio_effects, configs)


if __name__ == "__main__":

    BASE_CONFIG = Config(n_speakers=10, n_samples_per_speaker=10)
    FEMALE_ONLY_CONFIG = Config(gender="F")
    MALE_ONLY_CONFIG = Config(num_trials=1, gender="M")

    configs = [MALE_ONLY_CONFIG]
    run_optimization(MALE_ONLY_CONFIG)
