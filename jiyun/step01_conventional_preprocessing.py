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
    data = ch_data['trace'].data.copy()
    
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
                
    # 🆕 포화 검사 추가
    max_val = np.max(np.abs(data)) # 최대값
    saturation_threshold = max_val * 0.95 # 포화 임계값
    saturated_count = np.sum(np.abs(data) >= saturation_threshold)
    saturation_ratio = saturated_count / len(data)
    
    if saturation_ratio > 0.05:  # 5% 이상 포화
        print(f"    ⚠️ 포화 감지: {saturation_ratio:.1%} ({saturated_count}개 점)")
        print(f"    💡 이 채널은 정보 손실이 있을 수 있음")
        # 포화 정보를 메타데이터에 저장
        ch_data['quality_flags'] = ['saturation_detected']

    # 편집된 데이터 저장
    edited_trace = trace.copy()
    edited_trace.data = data
    edited_channels[channel_id] = edited_trace
    print(f"    ✅ 품질 검사 통과")

print(f"✅ 2단계 완료: {len(edited_channels)}개 채널 유지")


# =====================================================
# 3단계: Gain Recovery (이득 복구)
# =====================================================
print("\n🔸 3단계: Gain Recovery (이득 복구)")
print("- 기록 시 적용된 이득을 보상하여 원래 진폭 복원")

gain_recovered = {}
# edited_channels에서 items를 가져와 gain recovery 진행
for channel_id, trace in edited_channels.items():
    data = trace.data.copy()
    
    print(f"  ⚡ {channel_id} 이득 복구...")
    
    # 계기 응답 제거 (간단한 고역통과)
    # 매우 낮은 주파수 성분 제거 (0.01Hz 이하)
    if trace.stats.sampling_rate > 0.02:  # 나이퀴스트 조건
        # 수동으로 고역통과 필터 구현 (scipy 사용)
        nyquist = trace.stats.sampling_rate / 2
        low_cutoff = min(0.01, nyquist * 0.01)  # 안전한 cutoff
        
        try:
            b, a = signal.butter(2, low_cutoff / nyquist, btype='high')
            final_data = signal.filtfilt(b, a, final_data)
        except:
            print(f"    ⚠️ 고역통과 필터 실패 - 건너뜀")
    
    # 결과 저장
    recovered_trace = trace.copy()
    recovered_trace.data = final_data
    gain_recovered[channel_id] = recovered_trace
    
    print(f"    ✅ 이득 복구 완료")

print(f"✅ 3단계 완료: {len(gain_recovered)}개 채널 이득 복구")


# =====================================================
# 4단계: Filtering (필터링)
# =====================================================
print("\n🔸 4단계: Filtering (필터링)")
print("- 주파수 영역에서 노이즈 제거")

filtered_channels = {}
for channel_id, trace in gain_recovered.items():
    data = trace.data.copy()
    
    print(f"  🎛️ {channel_id} 필터링...")
    
    # 1. 선형 트렌드 제거 (수동 구현)
    x = np.arange(len(data))
    if len(data) > 1:
        slope = np.sum((x - np.mean(x)) * (data - np.mean(data))) / np.sum((x - np.mean(x))**2)
        intercept = np.mean(data) - slope * np.mean(x)
        trend = slope * x + intercept
        detrended_data = data - trend
    else:
        detrended_data = data
    
    # 2. 밴드패스 필터 (1-20Hz) - scipy 직접 사용
    sampling_rate = trace.stats.sampling_rate
    nyquist = sampling_rate / 2
    
    # 안전한 주파수 범위 설정
    low_freq = min(1.0, nyquist * 0.1)
    high_freq = min(20.0, nyquist * 0.9)
    
    if low_freq < high_freq and nyquist > low_freq:
        try:
            # 버터워스 밴드패스 필터
            b, a = signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
            bandpass_data = signal.filtfilt(b, a, detrended_data)
            print(f"    ✅ 밴드패스 필터 적용: {low_freq:.1f}-{high_freq:.1f}Hz")
        except Exception as e:
            print(f"    ⚠️ 밴드패스 필터 실패: {e}")
            bandpass_data = detrended_data
    else:
        print(f"    ⚠️ 부적절한 주파수 범위 - 필터 건너뜀")
        bandpass_data = detrended_data
    
    # 3. 노치 필터 (60Hz 전력선 간섭 제거)
    if sampling_rate > 120:  # 나이퀴스트 조건
        try:
            notch_freq = 60.0
            Q = 30
            b, a = signal.iirnotch(notch_freq, Q, sampling_rate)
            notched_data = signal.filtfilt(b, a, bandpass_data)
            print(f"    ✅ 노치 필터 적용: {notch_freq}Hz")
        except Exception as e:
            print(f"    ⚠️ 노치 필터 실패: {e}")
            notched_data = bandpass_data
    else:
        notched_data = bandpass_data
    
    # 결과 저장
    filtered_trace = trace.copy()
    filtered_trace.data = notched_data
    filtered_channels[channel_id] = filtered_trace

print(f"✅ 4단계 완료: {len(filtered_channels)}개 채널 필터링")


