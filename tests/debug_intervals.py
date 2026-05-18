import librosa
import pandas as pd
import numpy as np
from scipy.signal import hilbert, medfilt

file = 'wav/output_morse.wav'
y, sr = librosa.load(file, mono=True)
analytic = hilbert(y)
envelope = np.abs(analytic)
envelope = medfilt(envelope, kernel_size=101)
threshold = 0.3 * np.max(envelope)
binary = (envelope > threshold).astype(int)

diffs = np.diff(binary.astype(int), prepend=1 - binary[0])
change_idx = np.where(diffs != 0)[0]
time = np.arange(len(binary)) / sr
starts = time[change_idx]
ends = time[np.append(change_idx[1:] - 1, len(binary) - 1)]
values = binary[change_idx]

intervals = pd.DataFrame({'start': starts, 'end': ends, 'value': values}).dropna()
intervals['duration'] = intervals['end'] - intervals['start']

print("All intervals:")
print(intervals)
print("\nTone intervals (value=1):")
print(intervals[intervals['value'] == 1])
print("\nSilence intervals (value=0):")
print(intervals[intervals['value'] == 0])