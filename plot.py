import matplotlib.pyplot as plt
import numpy as np

from scipy.io import wavfile
import sys
from scipy import signal
import argparse
import struct


def error_to_signal(y, y_pred, use_filter=1):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    
    Implementation based on ESRLoss from training.py
    """
    if use_filter == 1:
        y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    
    # Calculate squared error
    error = np.power(y - y_pred, 2)
    # Take mean of squared error
    mean_error = np.mean(error)
    # Calculate signal energy with small epsilon to prevent division by zero
    energy = np.mean(np.power(y, 2)) + 1e-5
    # Return error to signal ratio
    return mean_error / energy


def pre_emphasis_filter(x, coeff=0.95):
    return np.concatenate([x, np.subtract(x, np.multiply(x, coeff))])


def read_wave(wav_file):
    # Extract Audio and framerate from Wav File
    fs, signal = wavfile.read(wav_file)
    return signal, fs


def analyze_pred_vs_actual(args):
    """Generate plots to analyze the predicted signal vs the actual
    signal.
    Inputs:
        output_wav : The actual signal, by default will use y_test.wav from the test.py output
        pred_wav : The predicted signal, by default will use y_pred.wav from the test.py output
        input_wav : The pre effect signal, by default will use x_test.wav from the test.py output
        model_name : Used to add the model name to the plot .png filename
        path   :   The save path for generated .png figures
        show_plots : Default is 1 to show plots, 0 to only generate .png files and suppress plots
    1. Plots the two signals
    2. Calculates Error to signal ratio the same way Pedalnet evauluates the model for training
    3. Plots the absolute value of pred_signal - actual_signal  (to visualize abs error over time)
    4. Plots the spectrogram of (pred_signal - actual signal)
         The idea here is to show problem frequencies from the model training
    """
    model_name = args.model_name
    path = "Results/" + model_name
    output_wav = "Data/test/" + args.config_name + args.output_wav
    pred_wav = path + "/" + args.pred_wav
    input_wav = "Data/test/" + args.config_name + args.input_wav
    show_plots = args.show_plots

    # Read the input wav file
    signal3, fs3 = read_wave(input_wav)
    #signal3 = signal3 / 32768.0  ################### quick conversion
    # Read the output wav file
    signal1, fs = read_wave(output_wav)
    #signal1 = signal1 / 32768.0   ################### quick conversion
    
    Time = np.linspace(0, len(signal1) / fs, num=len(signal1))
    fig, (ax3, ax1, ax2) = plt.subplots(3, sharex=True, figsize=(13, 8))
    fig.suptitle("Predicted vs Actual Signal")
    ax1.plot(Time, signal1, label=output_wav, color="red")

    # Read the predicted wav file
    signal2, fs2 = read_wave(pred_wav)

    Time2 = np.linspace(0, len(signal2) / fs2, num=len(signal2))
    ax1.plot(Time2, signal2, label=pred_wav, color="green")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Wav File Comparison")
    ax1.grid("on")

    error_list = []
    for s1, s2 in zip(signal1, signal2):
        error_list.append(abs(s2 - s1))

    # Calculate error to signal ratio with pre-emphasis filter as
    #    used to train the model
    e2s = error_to_signal(signal1, signal2)
    e2s_no_filter = error_to_signal(signal1, signal2, use_filter=0)
    print("Error to signal (with pre-emphasis filter): ", e2s)
    print("Error to signal (no pre-emphasis filter): ", e2s_no_filter)
    fig.suptitle("Predicted vs Actual Signal (error to signal: " + str(round(e2s, 4)) + ")")
    # Plot signal difference
    signal_diff = signal2 - signal1
    ax2.plot(Time2, error_list, label="signal diff", color="blue")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("abs(pred_signal-actual_signal)")
    ax2.grid("on")

    # Plot the original signal
    Time3 = np.linspace(0, len(signal3) / fs3, num=len(signal3))
    ax3.plot(Time3, signal3, label=input_wav, color="purple")
    ax3.legend(loc="upper right")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Original Input")
    ax3.grid("on")

    # Save the plot
    comparison_plot_path = path+'/'+model_name + "_signal_comparison_e2s_" + str(round(e2s, 4)) + ".png"
    plt.savefig(comparison_plot_path, bbox_inches="tight")

    # Create a zoomed in plot of 0.01 seconds centered at the max input signal value
    sig_temp = signal1.tolist()
    plt.axis(
        [
            Time3[sig_temp.index((max(sig_temp)))] - 0.005,
            Time3[sig_temp.index((max(sig_temp)))] + 0.005,
            min(signal2),
            max(signal2),
        ]
    )
    detail_plot_path = path+'/'+model_name + "_Detail_signal_comparison_e2s_" + str(round(e2s, 4)) + ".png"
    plt.savefig(detail_plot_path, bbox_inches="tight")

    # Reset the axis
    plt.axis([0, Time3[-1], min(signal2), max(signal2)])

    # Plot spectrogram difference
    # plt.figure(figsize=(12, 8))
    # print("Creating spectrogram data..")
    # frequencies, times, spectrogram = signal.spectrogram(signal_diff, 44100)
    # plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
    # plt.colorbar()
    # plt.title("Diff Spectrogram")
    # plt.ylabel("Frequency [Hz]")
    # plt.xlabel("Time [sec]")
    # plt.savefig(path+'/'+model_name + "_diff_spectrogram.png", bbox_inches="tight")

    if show_plots == 1:
        plt.show()
    
    # Return the paths to the generated plots
    return comparison_plot_path, detail_plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--path", default=".")
    parser.add_argument("config_name")
    parser.add_argument("--output_wav", default="-target.wav")
    parser.add_argument("--pred_wav", default="best_val_out.wav")
    parser.add_argument("--input_wav", default="-input.wav")
    parser.add_argument("--model_name", default="plot")
    parser.add_argument("--path", default="wavs")
    parser.add_argument("--show_plots", default=1)
    args = parser.parse_args()
    plot_paths = analyze_pred_vs_actual(args)
    
    # Print paths in a special format that can be easily parsed in the notebook
    if plot_paths:
        print("\nNOTEBOOK_DISPLAY_IMAGES:" + plot_paths[0] + "," + plot_paths[1])