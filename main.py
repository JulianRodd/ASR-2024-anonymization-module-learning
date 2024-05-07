from multiprocessing import Pool, cpu_count

from speaker_anonymization.config import Config
from speaker_anonymization.optimize import optimize_audio_effects

if __name__ == "__main__":


    BASE_CONFIG = Config()
    FEMALE_ONLY_CONFIG = Config(gender="F")
    MALE_ONLY_CONFIG = Config(gender="M")

    configs = [BASE_CONFIG, FEMALE_ONLY_CONFIG, MALE_ONLY_CONFIG]

    # # Repeat each config to fill up 50% of the CPU cores
    # configs = configs * (cpu_count() // 2 // len(configs))

    with Pool() as pool:
        pool.map(optimize_audio_effects, configs)

