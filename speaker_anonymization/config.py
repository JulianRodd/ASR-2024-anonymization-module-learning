class Config:

    def __init__(
        self,
        num_trials=10,
        n_speakers=10,
        n_samples_per_speaker=10,
        gender=None,
        min_age=None,
        max_age=None,
        accent=None,
        region=None,
    ):
        self.NUM_TRIALS = num_trials
        self.N_SPEAKERS = n_speakers
        self.N_SAMPLES_PER_SPEAKER = n_samples_per_speaker
        self.GENDER = gender
        self.MIN_AGE = min_age
        self.MAX_AGE = max_age
        self.ACCENT = accent
        self.REGION = region

        self.STUDY_NAME = f"new_loss_n_speakers-{self.N_SPEAKERS}_n_samples_per_speaker-{self.N_SAMPLES_PER_SPEAKER}"
        if gender is not None:
            self.STUDY_NAME += f"_gender-{self.GENDER}"
        if min_age is not None:
            self.STUDY_NAME += f"_min_age-{self.MIN_AGE}"
        if max_age is not None:
            self.STUDY_NAME += f"_max_age-{self.MAX_AGE}"
        if accent is not None:
            self.STUDY_NAME += f"_accent-{self.ACCENT}"
        if region is not None:
            self.STUDY_NAME += f"_region-{self.REGION}"

    # General
    STORAGE_NAME = f"sqlite:///optimize_audio_effects_for_anonymization.db"
    LOAD_IF_EXISTS = True
    IMAGES_DIR = "images"
    CACHE_FOLDER = "cache"
    ANONYMIZED_FOLDER = "anonymized_audio"
    SHOW_PROGRESS_BAR = False
    CONFIG_N_JOBS = 1  # Number of jobs to run in parallel, -1 means use all
    SEED = 42

    # ASR CONFIG
    ASR_BACKBONE = "Somebody433/fine-tuned-vctkdataset"

    # Speaker Identification Config
    SPI_BACKBONE = "facebook/wav2vec2-base"
    SPEAKER_IDENTIFICATION_EPOCHS = 30

    # Combined Loss Config
    WER_WEIGHT = 0.2
    SPI_WEIGHT = 0.2
    AGE_WEIGHT = 0.2
    GENDER_WEIGHT = 0.2
    ACCENT_WEIGHT = 0.1
    REGION_WEIGHT = 0.1


    # Sound Effects Config
    DISTORTION_DRIVE_DB_MEAN = 25
    DISTORTION_DRIVE_DB_STD = 12.5

    PITCH_SHIFT_SEMITONES_MEAN = -5
    PITCH_SHIFT_SEMITONES_STD = 2.5

    HIGHPASS_CUTOFF_MEAN = 100
    HIGHPASS_CUTOFF_STD = 50

    LOWPASS_CUTOFF_MEAN = 3500
    LOWPASS_CUTOFF_STD = 750

    TIME_STRETCH_FACTOR_MEAN = 1.0
    TIME_STRETCH_FACTOR_STD = 0.1

    BITCRUSH_BIT_DEPTH_MEAN = 16
    BITCRUSH_BIT_DEPTH_STD = 6

    CHORUS_RATE_HZ_MEAN = 25
    CHORUS_RATE_HZ_STD = 12.5

    PHASER_RATE_HZ_MEAN = 25
    PHASER_RATE_HZ_STD = 12.5

    GAIN_DB_MEAN = 0
    GAIN_DB_STD = 6
