import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
import torchaudio


def load_audio(file_path):
    waveform, sample_rate = sf.read(file_path)
    waveform = torch.tensor(waveform).float()  # Ensure the waveform is a float tensor
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)  # Convert stereo to mono by averaging channels.
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        waveform = resampler(waveform)
        sample_rate = 16000
    return waveform, sample_rate


def save_optimization_plots(study, images_dir):
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(images_dir, "optimization_history.png"))

    # Plot parameter importances
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(images_dir, "param_importances.png"))

    # Plot slice of each parameter
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(os.path.join(images_dir, "slice_plot.png"))

    # Optionally, save contour plot of interactions of each pair of parameters if you have many parameters
    if len(study.best_params) > 1:
        fig = optuna.visualization.plot_contour(study)
        fig.update_layout(
            autosize=False,
            width=1200,  # Width in pixels
            height=800,  # Height in pixels
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
            font=dict(size=10),
        )
        fig.write_image(os.path.join(images_dir, "contour_plot.png"))

    results = []
    for trial in study.trials:
        results.append(
            {
                "Trial": trial.number,
                "Average WER": trial.user_attrs["avg_wer"],
                "1 - Speaker Accuracy": 1 - trial.user_attrs["speaker_accuracy"],
                "Combined Loss": trial.user_attrs["combined_loss"],
            }
        )

    results_df = pd.DataFrame(results)

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results_df, x="Trial", y="Average WER", marker="o", label="Average WER"
    )
    sns.lineplot(
        data=results_df,
        x="Trial",
        y="1 - Speaker Accuracy",
        marker="o",
        label="1 - Speaker Accuracy",
    )
    sns.lineplot(
        data=results_df, x="Trial", y="Combined Loss", marker="o", label="Combined Loss"
    )
    plt.xlabel("Trial")
    plt.ylabel("Metric Value")
    plt.title("Effects of Audio Tweaks on Metrics")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(images_dir, "metrics_plot.png")
    plt.savefig(plot_path)

    print("Saved all optimization plots to:", images_dir)


def save_audio_file(audio, path, sample_rate):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Ensure the audio is in the correct format
    audio = np.ravel(audio)
    # Write the file
    sf.write(path, audio, samplerate=sample_rate, format="WAV", subtype="PCM_16")
