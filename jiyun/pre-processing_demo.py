# ==================================================
# Pre-processing

import obspy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pytz
from obspy.signal.trigger import classic_sta_lta, trigger_onset

def process_and_detect_p_wave(original_path,
                               sta_sec=0.5, lta_sec=5.0,
                               threshold_on=2.5, threshold_off=0.5):
    # 1. 원본 스트림 불러오기
    st_raw = obspy.read(original_path)

    # 2. DC Offset 제거
    st_demeaned = st_raw.copy()
    for tr in st_demeaned:
        tr.detrend(type='demean')

    # 3. Taper 적용 (20%)
    st_tapered = st_demeaned.copy()
    for tr in st_tapered:
        tr.taper(max_percentage=0.1, type='cosine')

    # 4. Bandpass 필터 적용
    st_filtered = st_tapered.copy()
    for tr in st_filtered:
        tr.filter("bandpass", freqmin=0.05, freqmax=20.0, corners=4, zerophase=True)

    # 5. 시각화 (KST 기준)
    fig, axes = plt.subplots(len(st_raw), 1, figsize=(14, 8), sharex=True)
    kst = pytz.timezone("Asia/Seoul")

    for i, (tr_demean, tr_filt) in enumerate(zip(st_demeaned, st_filtered)):
        # UTC 시간 → KST 전환
        time_array_utc = tr_demean.times("utcdatetime")
        time_array_kst = [t.datetime.replace(tzinfo=pytz.utc).astimezone(kst).replace(tzinfo=None) for t in time_array_utc]

        axes[i].plot_date(time_array_kst, tr_demean.data, 'gray', label='① DC Offset only', linewidth=1)
        axes[i].plot_date(time_array_kst, tr_filt.data, 'red', label='② + Taper + Bandpass', linewidth=0.8)

        axes[i].set_ylabel(tr_demean.stats.channel)
        axes[i].legend(loc='upper right')
        axes[i].grid(True)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # 6. Z 성분에 대해 STA/LTA 기반 P파 후보 탐지
    z_channels = [tr for tr in st_filtered if "Z" in tr.stats.channel.upper()]
    if not z_channels:
        print("Z 채널이 존재하지 않습니다.")
        return []

    tr_z = z_channels[0]
    df = tr_z.stats.sampling_rate
    cft = classic_sta_lta(tr_z.data, int(sta_sec * df), int(lta_sec * df))
    onsets = trigger_onset(cft, threshold_on, threshold_off)

    # 7. 탐지 결과 시각화 (KST)
    time_array_utc = tr_z.times("utcdatetime")
    time_array_kst = [t.datetime.replace(tzinfo=pytz.utc).astimezone(kst).replace(tzinfo=None) for t in time_array_utc]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot_date(time_array_kst, tr_z.data, 'k-', label='Filtered Z Channel', linewidth=0.8)

    for i, onset in enumerate(onsets):
        onset_time_utc = tr_z.stats.starttime + onset[0] / df
        onset_time_kst = onset_time_utc.datetime.replace(tzinfo=pytz.utc).astimezone(kst).replace(tzinfo=None)

        # 세로선 + 주석 추가
        ax.axvline(onset_time_kst, color='red', linestyle='--', linewidth=2,
                   label='P-wave candidate' if i == 0 else None)
        ax.text(onset_time_kst, max(tr_z.data)*0.9,
                onset_time_kst.strftime('%H:%M:%S'),
                color='red', rotation=90, verticalalignment='top')

    ax.set_title("STA/LTA-based P-wave Detection (Z Channel)")
    ax.set_xlabel("Time (KST)")
    ax.set_ylabel("Amplitude")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return onsets


# ==================================================
# Execute
original_file = "/content/drive/MyDrive/seismic_wave_data/kquake_dataset/KMA20230026_KG.BOG..HG.raw.mseed"
onsets = process_and_detect_p_wave(original_file)
