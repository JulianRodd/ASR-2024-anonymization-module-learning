
# Anonymization Pipeline Using Optuna and Pedalboard
To install all required packages, run:
```
pip install -r requirements.txt
```
To see optuna dashboard, run:
```
optuna-dashboard sqlite:///optimize_audio_effects_for_anonymization.db
```

## Results
# Results for Male, Female, and Combined Optuna Studies

## Experiment Results and Final Parameters

<table>
  <thead>
    <tr>
      <th>Experiment</th>
      <th>Metric</th>
      <th>Score</th>
      <th>Parameter</th>
      <th>Value</th>
      <th>Input Audio File</th>
      <th>Pseudonymized Audio File</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="9"><b>(a) - Male Only</b></td>
      <td>Best Trial</td>
      <td>294</td>
      <td>Distortion</td>
      <td>3.29 dB</td>
      <td rowspan="9">
        https://github.com/JulianRodd/ASR-2024-anonymization-module-learning/assets/45969914/f886ee27-16ef-479f-a170-8fe07bf2ec83
      </td>
      <td rowspan="9">
        https://github.com/JulianRodd/ASR-2024-anonymization-module-learning/assets/45969914/ad68359f-9ffb-455c-ad78-90c377bfa3c4
      </td>
    </tr>
    <tr>
      <td>avgWER</td>
      <td>0.160</td>
      <td>PitchShift</td>
      <td>-0.90 semitones</td>
    </tr>
    <tr>
      <td>spvAcc</td>
      <td>0.180</td>
      <td>HighpassFilter</td>
      <td>187.50 Hz</td>
    </tr>
    <tr>
      <td>psnyLoss</td>
      <td>0.170</td>
      <td>LowpassFilter</td>
      <td>4999.71 Hz</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Bitcrush</td>
      <td>12 bits</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Chorus</td>
      <td>1.45 Hz</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Phaser</td>
      <td>0.03 Hz</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Gain</td>
      <td>11.18 dB</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Time Stretch</td>
      <td>1.15</td>
    </tr>
    <tr>
      <td rowspan="9"><b>(b) - Female Only</b></td>
      <td>Best Trial</td>
      <td>223</td>
      <td>Distortion</td>
      <td>1.82 dB</td>
      <td rowspan="9"><audio controls><source src="input_female.mp3" type="audio/mpeg"></audio></td>
      <td rowspan="9"><audio controls><source src="pseudo_female.mp3" type="audio/mpeg"></audio></td>
    </tr>
    <tr>
      <td>avgWER</td>
      <td>0.205</td>
      <td>PitchShift</td>
      <td>-2.40 semitones</td>
    </tr>
    <tr>
      <td>spvAcc</td>
      <td>0.100</td>
      <td>HighpassFilter</td>
      <td>69.91 Hz</td>
    </tr>
    <tr>
      <td>psnyLoss</td>
      <td>0.152</td>
      <td>LowpassFilter</td>
      <td>2174.70 Hz</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Bitcrush</td>
      <td>11 bits</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Chorus</td>
      <td>2.04 Hz</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Phaser</td>
      <td>21.71 Hz</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Gain</td>
      <td>10.09 dB</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Time Stretch</td>
      <td>1.03</td>
    </tr>
    <tr>
      <td rowspan="9"><b>(c) - Combined</b></td>
      <td>Best Trial</td>
      <td>272</td>
      <td>Distortion</td>
      <td>12.95 dB</td>
      <td rowspan="9"><audio controls><source src="input_combined.mp3" type="audio/mpeg"></audio></td>
      <td rowspan="9"><audio controls><source src="pseudo_combined.mp3" type="audio/mpeg"></audio></td>
    </tr>
    <tr>
      <td>avgWER</td>
      <td>0.200</td>
      <td>PitchShift</td>
      <td>-0.90 semitones</td>
    </tr>
    <tr>
      <td>spvAcc</td>
      <td>0.160</td>
      <td>HighpassFilter</td>
      <td>45.11 Hz</td>
    </tr>
    <tr>
      <td>psnyLoss</td>
      <td>0.180</td>
      <td>LowpassFilter</td>
      <td>4402.55 Hz</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Bitcrush</td>
      <td>7 bits</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Chorus</td>
      <td>1.69 Hz</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Phaser</td>
      <td>30.77 Hz</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Gain</td>
      <td>1.34 dB</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td>Time Stretch</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
