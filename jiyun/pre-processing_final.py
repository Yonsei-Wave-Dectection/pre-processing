import obspy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pytz
import os
from glob import glob
from datetime import datetime

# 데이터 폴더 경로
data_dir = "/content/drive/MyDrive/seismic_wave_data/kquake_dataset/"
output_dir = "/content/drive/MyDrive/seismic_wave_data/preprocessed_csv/"
os.makedirs(output_dir, exist_ok=True)

# KST 타임존
kst = pytz.timezone("Asia/Seoul")

# 모든 MSEED 파일 반복 처리
for file_path in sorted(glob(os.path.join(data_dir, "*.mseed"))):
    try:
        # === 1~4. 전처리 ===
        st_raw = obspy.read(file_path)
        st_filtered = st_raw.copy()
        for tr in st_filtered:
            tr.detrend(type='demean')
            tr.taper(max_percentage=0.2, type='cosine')
            tr.filter("bandpass", freqmin=0.05, freqmax=20.0, corners=2, zerophase=True)

        # === 5. 시각화 (전처리된 파형만) ===
        fig, axes = plt.subplots(len(st_filtered), 1, figsize=(14, 8), sharex=True)
        for i, tr in enumerate(st_filtered):
            time_array_utc = tr.times("utcdatetime")
            time_array_kst = [t.datetime.replace(tzinfo=pytz.utc).astimezone(kst).replace(tzinfo=None) for t in time_array_utc]
            axes[i].plot_date(time_array_kst, tr.data, 'red', label='Preprocessed', linewidth=0.8)
            axes[i].set_ylabel(tr.stats.channel)
            axes[i].legend(loc='upper right')
            axes[i].grid(True)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        # === 6. CSV 저장 (KST 기준 시간 문자열로 저장) ===
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

    except Exception as e:
        print(f"⚠️ 오류 발생: {file_path}")
        print(e)
