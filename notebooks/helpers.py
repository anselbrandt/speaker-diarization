import matplotlib.pyplot as plt
import os
import torchaudio


def split_waveform_by_timestamps(
    mono_waveform, sample_rate, output_dir, timestamps, sortby="timesamp"
):

    for i, (start, end, speaker) in enumerate(timestamps):
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        segment = mono_waveform[0:, start_frame:end_frame]

        if sortby == "speaker":
            output_file = os.path.join(output_dir, f"{speaker}_{start}_{end}.wav")
        else:
            output_file = os.path.join(output_dir, f"{start}_{end}_{speaker}.wav")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torchaudio.save(output_file, segment, sample_rate)


def aggregate_timestamps(timestamps):
    aggregated = []
    previous = timestamps[0]
    for timestamp in timestamps:
        start, end, speaker = timestamp
        prevstart, prevend, prevspeaker = previous
        if speaker == prevspeaker:
            previous = (prevstart, end, speaker)
        else:
            aggregated.append(previous)
            previous = timestamp
    return aggregated


def plot_waveform(waveform):
    plt.figure()
    plt.plot(waveform.t().numpy())
    plt.show()
