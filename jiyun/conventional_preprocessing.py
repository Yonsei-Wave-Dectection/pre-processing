import numpy as np
import obspy
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

print("=== 크래시 안전 전통적인 지진파 전처리 6단계 ===")
print("Demultiplexing → Trace Editing → Gain Recovery → Filtering → Deconvolution → CMP Gather")

# 데이터 로딩
stream = obspy.read("ANMO_sample.mseed")
print(f"원시 데이터 로딩: {len(stream)}개 트레이스")

# 원본 데이터 백업
original_stream = stream.copy()


print("\n=== 3채널 데이터 시각화 준비 ===")

# 시간 축 생성
time_axis = {}
# BH1, BH2, BHZ 채널별 시간 및 데이터 범위
for i, trace in enumerate(stream):
    sampling_rate = trace.stats.sampling_rate
    num_samples = len(trace.data)
    duration = num_samples / sampling_rate
    time_axis[trace.stats.channel] = np.linspace(0, duration, num_samples)
    
    print(f"{trace.stats.channel} 채널:")
    print(f"  시간 범위: 0 ~ {duration:.1f}초")
    print(f"  데이터 범위: {trace.data.min():.1f} ~ {trace.data.max():.1f}")

print("\n시각화 코드 (matplotlib 사용 시):")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
for i, trace in enumerate(stream):
    channel = trace.stats.channel
    time = time_axis[channel]
    axes[i].plot(time, trace.data)
    axes[i].set_title(f'{channel} 채널')
    axes[i].set_ylabel('진폭')
    if i == 2:
        axes[i].set_xlabel('시간 (초)')
plt.tight_layout()
plt.show()


# =====================================================
# 1단계: Demultiplexing (역다중화)
# 이미 데이터가 3개의 트레이스에서 BH1, BH2, BHZ로 분리되어있기 때문에 확인하고 정리하는 단계로 여기면 된다
# =====================================================
print("\n🔸 1단계: Demultiplexing (역다중화)")
print("- 다채널 지진파 데이터를 개별 채널로 분리")

demux_channels = {}
for i, trace in enumerate(stream):
    channel_id = trace.stats.channel # 앞의 시각화 코드에서 channel 가져오기
    # demux_channel 딕셔너리 생성
    demux_channels[channel_id] = {
        'trace': trace,
        'network': trace.stats.network,
        'station': trace.stats.station,
        'channel': trace.stats.channel,
        'sampling_rate': trace.stats.sampling_rate,
        'npts': trace.stats.npts,
        'starttime': trace.stats.starttime
    }
    print(f"  📊 {channel_id}: {trace.stats.sampling_rate}Hz, {trace.stats.npts} samples")

print(f"✅ 1단계 완료: {len(demux_channels)}개 채널 분리")


# =====================================================
# 2단계: Trace Editing (트레이스 편집)
# =====================================================
print("\n🔸 2단계: Trace Editing (트레이스 편집)")
print("- 불량 데이터 제거 및 품질 관리")

edited_channels = {}
# demux_channels의 item을 가져와서 trace editing 진행
for channel_id, ch_data in demux_channels.items():
    trace = ch_data['trace']
    data = trace.data.copy()
    
    print(f"  🔍 {channel_id} 품질 검사...")
    
    # Dead trace 검사
    if np.all(data == 0) or np.var(data) < 1e-12:
        print(f"    ❌ Dead trace - 제거")
        continue
    
    # NaN/Inf 값 검사
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(f"    ⚠️ NaN/Inf 값 발견 - 보정")
        # Data Cleaning: NaN을 0으로, Inf를 클리핑
        data = np.nan_to_num(data, 
                             nan=0.0, # NaN을 0으로
                             posinf=np.max(data[np.isfinite(data)]), # +∞를 max로
                             neginf=np.min(data[np.isfinite(data)])) # -∞를 min으로
    
    # 스파이크 제거 (Z-score > 5)
    # z_scores = (값-평균) / 표준편차
    z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-10))
    # z_scores가 5 이상인 값을 spike로 count (표준편차의 5배 이상 벗어난 값을 의미)
    spike_count = np.sum(z_scores > 5)
    if spike_count > 0:
        print(f"    ⚠️ {spike_count}개 스파이크 제거")
        # 위치 찾기: z_scores > 5가 True인 값이 spike_mask이다
        spike_mask = z_scores > 5
        # 스파이크를 주변 값의 평균으로 대체
        for idx in np.where(spike_mask)[0]:
            if idx > 0 and idx < len(data) - 1:
                data[idx] = (data[idx-1] + data[idx+1]) / 2
    
    # 편집된 데이터 저장
    edited_trace = trace.copy()
    edited_trace.data = data
    edited_channels[channel_id] = edited_trace
    print(f"    ✅ 품질 검사 통과")

print(f"✅ 2단계 완료: {len(edited_channels)}개 채널 유지")
