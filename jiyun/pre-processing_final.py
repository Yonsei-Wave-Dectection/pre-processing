!pip install obspy matplotlib

from google.colab import drive
drive.mount('/content/drive')

import obspy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pytz
import os
from glob import glob
from datetime import datetime

# Data Folder Path
# You need to revise
data_dir = "/content/drive/MyDrive/seismic_wave_data/kquake_dataset/"
output_dir = "/content/drive/MyDrive/seismic_wave_data/preprocessed_csv/"
os.makedirs(output_dir, exist_ok=True)

# KST Timezone
kst = pytz.timezone("Asia/Seoul")

# Iterate through all MSEED files
for file_path in sorted(glob(os.path.join(data_dir, "*.mseed"))):
    try:
        # === 1~3. Preprossesing ===
        st_raw = obspy.read(file_path)
        st_filtered = st_raw.copy()
        for tr in st_filtered:
            tr.detrend(type='demean') # 1. DC offset Removal
            tr.taper(max_percentage=0.2, type='cosine') # 2. Cosine Taper (20%)
            tr.filter("bandpass", freqmin=0.05, freqmax=15.0, corners=2, zerophase=True) # Bandpass Filter(0.05Hz ~ 15.0Hz)

        # === 4. Save CSV (Save as KST Time String) ===
        for tr in st_filtered:
            time_array_utc = tr.times("utcdatetime")
            time_array_kst_str = [
                t.datetime.replace(tzinfo=pytz.utc).astimezone(kst).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                for t in time_array_utc
            ]
            output_data = np.column_stack((time_array_kst_str, tr.data))
            filename = os.path.basename(file_path).replace(".mseed", f"_{tr.stats.channel}.csv")
            np.savetxt(os.path.join(output_dir, filename),
                       output_data,
                       delimiter=",",
                       fmt='%s',
                       header="KST_datetime,amplitude", comments='')
        print("Complete!")

    except Exception as e:
        print(f"Error!: {file_path}")
        print(e)
