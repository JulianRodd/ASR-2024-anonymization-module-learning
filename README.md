
# Anonymization Pipeline Using Optuna and Pedalboard
To install all required packages, run:
```
pip install -r requirements.txt
```
To see optuna dashboard, run:
```
optuna-dashboard sqlite:///optimize_audio_effects_for_anonymization.db
```

# Results for Male, Female, and Combined Optuna Studies

## Experiment Results and Final Parameters

## Results

# Results for Male, Female, and Combined Optuna Studies

## Experiment Results and Final Parameters

| Experiment          | Metric       | Score    | Parameter       | Value          | Input Audio File | Pseudonymized Audio File |
|---------------------|--------------|----------|-----------------|----------------|------------------|--------------------------|
| **(a) - Male Only** | Best Trial   | 294      | Distortion      | 3.29 dB        | https://github.com/JulianRodd/ASR-2024-anonymization-module-learning/assets/45969914/f886ee27-16ef-479f-a170-8fe07bf2ec83 | [Pseudonymized Male](https://github.com/JulianRodd/ASR-2024-anonymization-module-learning/assets/45969914/ad68359f-9ffb-455c-ad78-90c377bfa3c4) |
|                     | avgWER       | 0.160    | PitchShift      | -0.90 semitones|                  |                          |
|                     | spvAcc       | 0.180    | HighpassFilter  | 187.50 Hz      |                  |                          |
|                     | psnyLoss     | 0.170    | LowpassFilter   | 4999.71 Hz     |                  |                          |
|                     |              |          | Bitcrush        | 12 bits        |                  |                          |
|                     |              |          | Chorus          | 1.45 Hz        |                  |                          |
|                     |              |          | Phaser          | 0.03 Hz        |                  |                          |
|                     |              |          | Gain            | 11.18 dB       |                  |                          |
|                     |              |          | Time Stretch    | 1.15           |                  |                          |
| **(b) - Female Only** | Best Trial | 223      | Distortion      | 1.82 dB        | [Input Female](https://github.com/JulianRodd/ASR-2024-anonymization-module-learning/assets/45969914/c77ffa7d-257f-45e1-a85f-31ef9f93bed0) | [Pseudonymized Female](https://github.com/JulianRodd/ASR-2024-anonymization-module-learning/assets/45969914/3fa74f62-85d0-42ea-b74e-a7bd33a97118) |
|                     | avgWER       | 0.205    | PitchShift      | -2.40 semitones|                  |                          |
|                     | spvAcc       | 0.100    | HighpassFilter  | 69.91 Hz       |                  |                          |
|                     | psnyLoss     | 0.152    | LowpassFilter   | 2174.70 Hz     |                  |                          |
|                     |              |          | Bitcrush        | 11 bits        |                  |                          |
|                     |              |          | Chorus          | 2.04 Hz        |                  |                          |
|                     |              |          | Phaser          | 21.71 Hz       |                  |                          |
|                     |              |          | Gain            | 10.09 dB       |                  |                          |
|                     |              |          | Time Stretch    | 1.03           |                  |                          |
| **(c) - Combined**   | Best Trial  | 272      | Distortion      | 12.95 dB       | [Input Combined (Female)](https://github.com/JulianRodd/ASR-2024-anonymization-module-learning/assets/45969914/3c6c30a6-c14b-41ca-8b68-ed0525e0238b) | [Pseudonymized Combined (Female)](https://github.com/JulianRodd/ASR-2024-anonymization-module-learning/assets/45969914/f9fd2cd0-ded0-45b5-a743-e26c23a79ff3) |
|                     | avgWER       | 0.200    | PitchShift      | -0.90 semitones| [Input Combined (Male)](https://github.com/JulianRodd/ASR-2024-anonymization-module-learning/assets/45969914/6fefc5e9-5f73-47d0-95a4-27ccde00ac8d)|  [Pseudonymized Combined (Male)](https://github.com/JulianRodd/ASR-2024-anonymization-module-learning/assets/45969914/28e18df3-4dc0-42ee-b918-ef3929c74441) |
|                     | spvAcc       | 0.160    | HighpassFilter  | 45.11 Hz       |                  |                          |
|                     | psnyLoss     | 0.180    | LowpassFilter   | 4402.55 Hz     |                  |                          |
|                     |              |          | Bitcrush        | 7 bits         |                  |                          |
|                     |              |          | Chorus          | 1.69 Hz        |                  |                          |
|                     |              |          | Phaser          | 30.77 Hz       |                  |                          |
|                     |              |          | Gain            | 1.34 dB        |                  |                          |
|                     |              |          | Time Stretch    | 1.00           |                  |                          |
