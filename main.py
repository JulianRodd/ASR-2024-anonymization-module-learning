from speaker_anonymization.optimize import optimize_audio_effects
from speaker_anonymization.config import Config
from concurrent.futures import ThreadPoolExecutor
if __name__ == "__main__":

    BASE_CONFIG = Config()
    FEMALE_ONLY_CONFIG = Config(gender="F")
    MALE_ONLY_CONFIG = Config(gender="M")

    configs = [BASE_CONFIG, FEMALE_ONLY_CONFIG, MALE_ONLY_CONFIG]

    with ThreadPoolExecutor() as executor:
        executor.map(optimize_audio_effects, configs)
