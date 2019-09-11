import sys
import mne
import numpy as np
import datetime

raw = mne.io.read_raw_brainvision(sys.argv[1], preload=True)

raw.pick_types(meg=False, eeg=True, eog=True)

chs = raw.info['ch_names'][0]
for i in range(1, len(raw.info['ch_names'])):
    chs = chs + '\n\t ' + raw.info['ch_names'][i]

from datetime import datetime, timedelta

unix_ts = raw.info['meas_date'][0]
dt = (datetime.fromtimestamp(unix_ts) - timedelta(hours = 2)).strftime('%Y-%m-%d %H:%M:%S')

output = 'Accuquired date: ' + str(dt) + '\n\nNumber of channels: ' + str(raw.info['nchan']) + '\n\nChannel names: ' + chs + '\n\nNumber of data points: ' + str(raw.n_times) + '\n\nFrequency: ' + str(raw.info['sfreq']) + '\n\nRecorded time period: ' + str(int(raw.n_times) / int(raw.info['sfreq'])) + 'seconds'

print(output)

raw_avg_ref, _ = mne.set_eeg_reference(raw, ref_channels='average')
raw_avg_df = raw_avg_ref.to_data_frame()
raw_avg_arr = raw_avg_df.to_numpy()
