from multiprocessing import Pool, cpu_count

from speaker_anonymization.config import Config
from speaker_anonymization.optimize import optimize_audio_effects


def run_optimization(config):
    optimize_audio_effects(config)


def run_optimizations(configs):
    # Repeat each config to fill up 50% of the CPU cores
    amount_of_runners = (cpu_count() // 2 // len(configs))

    configs = configs * amount_of_runners

    # to make sure all configs types still have their num_trials add up to the one they were supposed to
    # So if we have config1 is now 2 times in the list, its num_trials should be halved
    unique_configs = list(set(configs))
    for config in unique_configs:
        amount_of_times_config_is_in_list = len([c for c in configs if c == config])
        config.NUM_TRIALS = config.NUM_TRIALS // amount_of_times_config_is_in_list

    with Pool() as pool:
        pool.map(optimize_audio_effects, configs)


if __name__ == "__main__":
    BASE_CONFIG = Config(n_speakers=2, n_samples_per_speaker=2)
    # FEMALE_ONLY_CONFIG = Config(gender="F")
    # MALE_ONLY_CONFIG = Config(num_trials=1, gender="M")

    #configs = [BASE_CONFIG]
    # run_optimization(BASE_CONFIG)
    run_optimizations([BASE_CONFIG])
