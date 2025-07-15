import numpy as np
import obspy
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

print("Conventional preprocessing starts!")
print("="*60)

print("=== 6-steps of Conventional Seismic Wave Data Preprocessing ===")
print("Demultiplexing → Trace Editing → Gain Recovery → Filtering → Deconvolution → CMP Gather")

# Data loading
print("\nRaw data loading...")
stream = obspy.read("ANMO_sample.mseed")
print(f"✅ Loading complete: {len(stream)} tracees")

# Backup original data
original_stream = stream.copy()


print("\n=== 3-channel data visualization ===")

time_axis = {}
# Time and data range by BH1, BH2, BHZ channels
for i, trace in enumerate(stream):
    sampling_rate = trace.stats.sampling_rate
    num_samples = len(trace.data)
    duration = num_samples / sampling_rate
    time_axis[trace.stats.channel] = np.linspace(0, duration, num_samples)
    
    print(f"{trace.stats.channel} Channel:")
    print(f"  Time range: 0 ~ {duration:.1f} seconds")
    print(f"  Data Range: {trace.data.min():.1f} ~ {trace.data.max():.1f}")

print("\nVisualization (using matplotlib):")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
for i, trace in enumerate(stream):
    channel = trace.stats.channel
    time = time_axis[channel]
    axes[i].plot(time, trace.data)
    axes[i].set_title(f'{channel} Channel')
    axes[i].set_ylabel('Amplitude')
    if i == 2:
        axes[i].set_xlabel('second (s)')
plt.tight_layout()
plt.show()


# =====================================================
# 1단계: Demultiplexing (역다중화)
# 이미 데이터가 3개의 트레이스에서 BH1, BH2, BHZ로 분리되어있기 때문에 확인하고 정리하는 단계로 여기면 된다
# =====================================================
print("\n🔸 Step1: Demultiplexing")
print("- Seperate multi-channel seismic data into individual channels")

demux_channels = {}
for i, trace in enumerate(stream):
    channel_id = trace.stats.channel # 앞의 시각화 코드에서 channel 가져오기
    # demux_channel dictionary
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

print(f"✅ Step1 complete: {len(demux_channels)} channels seperated")


# =====================================================
# 2단계: Trace Editing (트레이스 편집)
# =====================================================
print("\n   Step2: Trace Editing")
print("- Poor data removal and quality check")

edited_channels = {}
# demux_channels의 item을 가져와서 trace editing 진행
for channel_id, ch_data in demux_channels.items():
    trace = ch_data['trace'] # ch_data 딕셔너리에서 Obspy 객체인 'trace' item을 추출한다.
    data = trace.data.copy() # Obspy trace 객체의 trace.data의 숫자만 복사하여 추출한다
    
    print(f"  🔍 {channel_id} Quality check...")
    
    # Dead trace 검사
    if np.all(data == 0) or np.var(data) < 1e-12:
        print(f"    ❌ Dead trace - Remove")
        continue
    
    # NaN/Inf 값 검사
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(f"    ⚠️ NaN/Inf values detected - Correction")
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
        print(f"    ⚠️ {spike_count} Spikes Remove")
        # 위치 찾기: z_scores > 5가 True인 값이 spike_mask이다
        spike_mask = z_scores > 5
        # 스파이크를 주변 값의 평균으로 대체
        for idx in np.where(spike_mask)[0]:
            if idx > 0 and idx < len(data) - 1:
                data[idx] = (data[idx-1] + data[idx+1]) / 2
                
    # 포화 검사 추가
    max_val = np.max(np.abs(data)) # 최대값
    saturation_threshold = max_val * 0.95 # 포화 임계값
    saturated_count = np.sum(np.abs(data) >= saturation_threshold)
    saturation_ratio = saturated_count / len(data)
    
    if saturation_ratio > 0.05:  # Saturation above 5%
        print(f"    ⚠️ Saturation detected: {saturation_ratio:.1%} ({saturated_count} points)")
        print(f"    💡 이 채널은 정보 손실이 있을 수 있음")
        # 포화 정보를 메타데이터에 저장
        ch_data['quality_flags'] = ['saturation_detected']

    # 편집된 데이터 저장
    edited_trace = trace.copy()
    edited_trace.data = data
    edited_channels[channel_id] = edited_trace
    print(f"       Data Quality Check Complete")

print(f"   Step2 complete: {len(edited_channels)} channels")


# =====================================================
# 3단계: Gain Recovery (이득 복구)
#- 기록 시 적용된 이득을 보상하여 원래 진폭 복원
# =====================================================
print("\n   Step3: Gain Recovery")
print("Restore original amplitude by compensating for gain applied during recording")

gain_recovered = {}
# edited_channels에서 items를 가져와 gain recovery 진행
for channel_id, trace in edited_channels.items():
    data = trace.data.copy()
    
    print(f"  ⚡ {channel_id} Gain Recovery...")
    
    # 계기 응답 제거 (간단한 고역통과)
    # 매우 낮은 주파수 성분 제거 (0.01Hz 이하)
    if trace.stats.sampling_rate > 0.02:  # 나이퀴스트 조건
        # 수동으로 고역통과 필터 구현 (scipy 사용)
        nyquist = trace.stats.sampling_rate / 2
        low_cutoff = min(0.01, nyquist * 0.01)  # safe cutoff
        
        try:
            b, a = signal.butter(2, low_cutoff / nyquist, btype='high')
            final_data = signal.filtfilt(b, a, data)
            print(f"        HPF applied (cutoff: {low_cutoff:.4f}Hz)")
        except:
            final_data = data
            print(f"    ⚠️ HPF Failed - Skip")
    else:
        final_data = data
        print(f"    ⚠️ Sampling frequency too low - Skip")
    # 결과 저장
    recovered_trace = trace.copy()
    recovered_trace.data = final_data
    gain_recovered[channel_id] = recovered_trace
    
    print(f"       Gain Recovery complete")

print(f"   Step 3 complete: {len(gain_recovered)} channels gain recovered")


def simple_trend_check(data, sampling_rate):
    """간단한 트렌드 확인"""

    # 시간축 생성
    time = np.arange(len(data)) / sampling_rate

    # 시각화
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, data)
    plt.title('raw data')
    plt.ylabel('Amplitude')

    # 간단한 이동평균으로 트렌드 확인
    window_size = len(data) // 20  # 전체의 5%
    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='same')

    plt.subplot(2, 1, 2)
    plt.plot(time, moving_avg, 'r-', linewidth=2)
    plt.title('moving average (approximate trend)')
    plt.xlabel('second (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    # 간단한 기준
    trend_range = np.max(moving_avg) - np.min(moving_avg)
    data_range = np.max(data) - np.min(data)
    trend_ratio = trend_range / data_range

    print(f"trend ratio: {trend_ratio:.2%}")

    return trend_ratio
    '''
    if trend_ratio > 0.3:  # 30% 이상이면
        return "트렌드 의심 - 제거 고려"
    else:
        return "트렌드 미미 - 제거 불필요"
    '''


# =====================================================
# 4단계: Filtering (필터링)
# =====================================================
print("\n    Step4: Filtering")
print("- Noise Removal in the frequency domain")

filtered_channels = {}
# gain_recovered의 items를 가져와 filtering을 수행한다
for channel_id, trace in gain_recovered.items():
    data = trace.data.copy()
    
    print(f"  🎛️ {channel_id} Filtering...")
    
    # 1. Linear Trend Removal
    x = np.arange(len(data))
    if simple_trend_check(data, sampling_rate) > 0.3:
        slope = np.sum((x - np.mean(x)) * (data - np.mean(data))) / np.sum((x - np.mean(x))**2)
        intercept = np.mean(data) - slope * np.mean(x)
        trend = slope * x + intercept
        detrended_data = data - trend
        """"
        1) 최소제곱법(Least Squares)
           slope = Σ[(x-x̄)(y-ȳ)] / Σ[(x-x̄)²]
           x̄ = np.mean(x)     # x의 평균
           ȳ = np.mean(data)   # y의 평균
        2) intercept
           직선의 방정식: y = slope * x + intercept
           intercept = ȳ - slope * x̄
        3) 트렌드 라인 생성
        4) 데이터에서 트렌드 제거
        """
        print(f"trend removed!\n")
    else:
        detrended_data = data
        print(f"trend not removed!(stabilized)\n")
    
    # 2. 밴드패스 필터 (1-20Hz) - directly using scipy
    # why 1-20Hz?: Practically optimized range of most useful seismic wave
    sampling_rate = trace.stats.sampling_rate   # 40 Hz here
    nyquist = sampling_rate / 2   # 20Hz here

    # 안전한 주파수 범위 설정
    low_freq = min(1.0, nyquist * 0.1) # 1 or 10% of nyquist
    high_freq = min(20.0, nyquist * 0.9) # 20 or 90% of nyquist
    
    if low_freq < high_freq and nyquist > low_freq: # the latter: checking if filtering is available
        try:
            # 버터워스 밴드패스 필터
            b, a = signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
            # butter(order, normalized frequency, ...)
            # butterworth BPF uses normalized frequency
            bandpass_data = signal.filtfilt(b, a, detrended_data)
            # filtfilt: two-way filtering (to compensate phase delay)
            print(f"    -> BPF applied: {low_freq:.1f}-{high_freq:.1f}Hz")  # It means, BPF range is 1.0-18.0Hz
        except Exception as e:
            print(f"    -> BPF failed: {e}")
            bandpass_data = detrended_data  # fallback to the former detrended_data
    else:
        print(f"       inappropriate frequency range- skip filtering") # not applying filtering
        bandpass_data = detrended_data
    
    # 3. 노치 필터 (60Hz 전력선 간섭 제거)
    if sampling_rate > 120:  # Nyquist condition: need at least 1120Hz to filter 60Hz
        try:
            # Infinite Impulse Response (IIR) Notch filter design
            notch_freq = 60.0
            Q = 30
            b, a = signal.iirnotch(notch_freq, Q, sampling_rate)
            notched_data = signal.filtfilt(b, a, bandpass_data)
            print(f"    -> Notch Filter Applied: {notch_freq}Hz")
        except Exception as e:
            print(f"    -> Nothch Filter Failed: {e}")
            notched_data = bandpass_data
    else:
        notched_data = bandpass_data
    
    # 결과 저장
    filtered_trace = trace.copy()
    filtered_trace.data = notched_data
    filtered_channels[channel_id] = filtered_trace

print(f"  Step4 completed: Filtering {len(filtered_channels)} channels")


# =====================================================
# 5단계: Deconvolution (역컨볼루션)
#- 지진파 전파 과정에서 발생한 파형 왜곡 보정
# =====================================================
print("\n   Step5: Deconvolution")
print("Correction of waveform distortion caused by seismic wave propagation")
 
deconvolved_channels = {}
# 필터 적용한 filtered_channels의 items를 가져온다
for channel_id, trace in filtered_channels.items():
    data = trace.data.copy()
    
    print(f"  🔄 {channel_id} Deconvolution...")
    
    # 1. Predictive Deconvolution (simple form)
    # Purpose?: Remove reverberation / Pulse compression / Improve signal / Improve noise characteristic

    # Autocorrelation based Predictive filter
    autocorr_length = min(100, len(data) // 10)  # safe length
    
    if len(data) > autocorr_length * 2:
        # Calculate Autocorrelation
        autocorr = np.correlate(data, data, mode='full')
        # autocorrelation: correlate by oneself / mode: decide output size- full(max info))
        center = len(autocorr) // 2     # finding center
        autocorr = autocorr[center:center + autocorr_length]    # extract only positive lag
        
        # Prediction Filter design
        if len(autocorr) > 1 and np.std(autocorr) > 0:
            # Standardize pattern 
            # autocorr[0]: Total energy of the signal
            # effect: 신호의 크기 정보는 없어지지만, 신호의 패턴을 알 수 있다.
            autocorr = autocorr / autocorr[0]
            
            # 2-order prediction filter
            if len(autocorr) >= 3: # at least 3 to accesss autocorr[0], [1], [2]
                # when the predictive deconvolution is applied, the equation looks like below
                #   e[n]   =   x[n]   -   autocorr[1]*x[n-1]   +   (autocorr[2]/2) * x[n-2]
                pred_filter = np.array([1, -autocorr[1], autocorr[2]/2])
            else:
                # when the predictive deconvolution is applied, the equation looks like below
                #   e[n]   =   x[n]   -   0.5 * x[n-1]
                pred_filter = np.array([1, -0.5])
        else:
            # when the predictive deconvolution is applied, the equation looks like below
            #   e[n]   =   x[n]
            pred_filter = np.array([1])
        
        # Apply predictive deconvolution 
        try:
            deconv_data = signal.lfilter(pred_filter, [1], data)
        except:
            deconv_data = data
    else:
        deconv_data = data
    
    # 2. Spiking deconvolution effect (enhance High Frequency (HF) components)
    # : to enhance seismic wave resolution, we enhance HF components
    # Enhance HF by 1-order differentiation
    diff_data = np.diff(deconv_data, prepend=deconv_data[0])
    
    # Weighted sum of original and derivative (70% original + 30% HF enhancement)
    enhanced_data = 0.7 * deconv_data + 0.3 * diff_data
    
    # Save results
    deconv_trace = trace.copy()
    deconv_trace.data = enhanced_data
    deconvolved_channels[channel_id] = deconv_trace
    
    print(f"       Deconvolution Complete")

print(f"   Step5 Complete: Deconvolution of {len(deconvolved_channels)} channels")


# =====================================================
# 6단계: CMP Gather (공통 중점 집합)
#- 같은 지하 점을 반사한 신호들을 그룹화
# =====================================================
print("\n   Step6: CMP Gather")
print("- Group signals that reflect the same underground point")

# Grouping by channels (Z, N, E components)
gathered_channels = {}
channel_groups = {
    'vertical': [],    # Z component
    'horizontal_1': [], # N, 1 component 
    'horizontal_2': []  # E, 2 component
}

# Take items of deconvolved_channels
for channel_id, trace in deconvolved_channels.items():
    if 'Z' in channel_id: # BHZ
        channel_groups['vertical'].append((channel_id, trace))
    elif 'N' in channel_id or '1' in channel_id: #BH1
        channel_groups['horizontal_1'].append((channel_id, trace))
    elif 'E' in channel_id or '2' in channel_id: #BH2
        channel_groups['horizontal_2'].append((channel_id, trace))

# Representative channel selection for each group
for group_name, traces in channel_groups.items():
    if traces:
        if len(traces) == 1:
            # Single channel
            channel_id, trace = traces[0]
            gathered_channels[f"{group_name}_{channel_id}"] = trace
            print(f"  -> {group_name}: Select {channel_id}")
        else:
            # Multi channel: Select the first one
            channel_id, trace = traces[0]
            gathered_channels[f"{group_name}_{channel_id}"] = trace
            print(f"  -> {group_name}: Select {channel_id} among ({len(traces)} traces)")

print(f"   Step6 Completed: Form {len(gathered_channels)} groups")


# =====================================================
# Final results
# =====================================================
print("\nCompleted 6-steps conventional seismic preprocessing")
print("="*50)
print("Processing results: ")
for step, count in [
    ("1. Demultiplexing", len(demux_channels)),
    ("2. Trace Editing", len(edited_channels)), 
    ("3. Gain Recovery", len(gain_recovered)),
    ("4. Filtering", len(filtered_channels)),
    ("5. Deconvolution", len(deconvolved_channels)),
    ("6. CMP Gather", len(gathered_channels))
]:
    print(f"  {step}: {count} channels")

print("\nFinal output:")
final_processed_data = {}
for channel_name, trace in gathered_channels.items():
    final_processed_data[channel_name] = {
        'data': trace.data,
        'sampling_rate': trace.stats.sampling_rate,
        'channel': trace.stats.channel,
        'length': len(trace.data)
    }
    print(f"  🔸 {channel_name}: {len(trace.data)} samples @ {trace.stats.sampling_rate}Hz")

print("\n✅ Completed conventional preprocessing. Now proceed Deeplearning preprocessing step.")




import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("Deeplearning preprocessing starts!")
print("="*60)

# ============================================================================
# Step1. Data preparation and validation
# ============================================================================
print("\n   Step1: Data preparation and validation")

# Check conventionally preprocessed seismic wave data
print("Preprocessed seismic wave data:")
for channel_name, data_info in final_processed_data.items():
    print(f"  🔸 {channel_name}: {data_info['length']} samples @ {data_info['sampling_rate']}Hz")

# 위도/경도 파싱 함수
def parse_coordinate(coord_str):
    """위도/경도 문자열을 숫자로 변환"""
    try:
        if pd.isna(coord_str) or coord_str == '':
            return None
            
        coord_str = str(coord_str).strip()
        
        # 숫자와 방향 분리
        import re
        match = re.match(r'([0-9.]+)\s*([NSEW])', coord_str)
        
        if match:
            value = float(match.group(1))
            direction = match.group(2).upper()
            
            # 남쪽(S)과 서쪽(W)은 음수
            if direction in ['S', 'W']:
                value = -value
                
            return value
        else:
            # 순수 숫자인 경우
            try:
                return float(coord_str)
            except:
                return None
    except Exception as e:
        print(f"    ⚠️ 좌표 파싱 오류: {coord_str} -> {str(e)}")
        return None
    
# CSV 파일 구조 문제 해결
print("🔧 CSV 카탈로그 로딩 및 정리...")

try:
    # 원본 CSV 파일 로딩
    raw_catalog = pd.read_csv("outCountryEarthquakeList_2000-01-01_2025-07-04.csv")
    
    print(f"📋 원본 CSV 구조 확인:")
    print(f"  -> 총 행수: {len(raw_catalog)}")
    print(f"  -> 컬럼들: {list(raw_catalog.columns)}")
    
    # 첫 5행 내용 확인
    print("📊 첫 5행 내용:")
    for i in range(min(5, len(raw_catalog))):
        print(f"  행 {i}: {raw_catalog.iloc[i, 0]}")
    
    # 실제 데이터가 시작하는 행 찾기
    data_start_row = None
    for i in range(len(raw_catalog)):
        first_col = str(raw_catalog.iloc[i, 0])
        # 숫자로 시작하는 행 찾기 (실제 데이터)
        if first_col.isdigit():
            data_start_row = i
            break
    
    if data_start_row is not None:
        print(f"✅ 실제 데이터 시작 행: {data_start_row}")
        
        # 헤더를 data_start_row-1로, 데이터를 data_start_row부터
        if data_start_row > 0:
            catalog_df = pd.read_csv("outCountryEarthquakeList_2000-01-01_2025-07-04.csv", 
                                   header=data_start_row-1, 
                                   skiprows=range(0, data_start_row-1))
        else:
            catalog_df = raw_catalog
    else:
        # 헤더가 명확하지 않으면 수동으로 설정
        print("⚠️ 데이터 시작 행을 찾을 수 없음 - 수동 처리")
        catalog_df = raw_catalog.iloc[2:].copy()  # 3행부터 데이터
        
        # 컬럼명 수동 설정
        expected_columns = ['number', 'magnitude', 'depth', 
                           'latitude', 'longitude', 'location']
        catalog_df.columns = expected_columns[:len(catalog_df.columns)]
    
    print(f"📊 정리된 컬럼들: {list(catalog_df.columns)}")
    
    # 유효한 데이터만 필터링
    # 첫 번째 컬럼이 숫자인 행만 선택
    if len(catalog_df.columns) > 0:
        first_col_name = catalog_df.columns[0]
        # 숫자로 변환 가능한 행만 선택
        numeric_mask = pd.to_numeric(catalog_df[first_col_name], errors='coerce').notna()
        catalog_clean = catalog_df[numeric_mask].copy()
        
        # 컬럼명이 이상하면 표준 이름으로 변경
        if 'magnitude' not in catalog_clean.columns:
            column_mapping = {}
            cols = list(catalog_clean.columns)
            
            # 예상 순서에 따라 매핑
            standard_names = ['number', 'magnitude', 'depth', 
                             'latitude', 'longitude', 'location']
            
            for i, std_name in enumerate(standard_names):
                if i < len(cols):
                    column_mapping[cols[i]] = std_name
            
            catalog_clean = catalog_clean.rename(columns=column_mapping)
            print(f"🔄 컬럼명 변경 완료: {column_mapping}")
        
        # 위도/경도 특별 처리
        print("🌍 위도/경도 파싱 중...")
        
        if 'latitude' in catalog_clean.columns:
            print(f"  📊 위도 샘플: {catalog_clean['latitude'].head(3).tolist()}")
            catalog_clean['latitude'] = catalog_clean['latitude'].apply(parse_coordinate)
            valid_lat = catalog_clean['latitude'].dropna()
            print(f"  ✅ 위도 파싱 완료: {len(valid_lat)}개 성공")
            if len(valid_lat) > 0:
                print(f"      범위: {valid_lat.min():.2f}° ~ {valid_lat.max():.2f}°")
        
        if 'longitude' in catalog_clean.columns:
            print(f"  📊 경도 샘플: {catalog_clean['longitude'].head(3).tolist()}")
            catalog_clean['longitude'] = catalog_clean['longitude'].apply(parse_coordinate)
            valid_lon = catalog_clean['longitude'].dropna()
            print(f"  ✅ 경도 파싱 완료: {len(valid_lon)}개 성공")
            if len(valid_lon) > 0:
                print(f"      범위: {valid_lon.min():.2f}° ~ {valid_lon.max():.2f}°")

        # 숫자 컬럼들 타입 변환
        numeric_columns = ['magnitude', 'depth']
        for col in numeric_columns:
            if col in catalog_clean.columns:
                catalog_clean[col] = pd.to_numeric(catalog_clean[col], errors='coerce')
        
        # 유효한 magnitude가 있는 행만 선택
        if 'magnitude' in catalog_clean.columns:
            valid_magnitude_mask = catalog_clean['magnitude'].notna()
            catalog = catalog_clean[valid_magnitude_mask].reset_index(drop=True)
        else:
            catalog = catalog_clean.reset_index(drop=True)
        
        print(f"✅ 카탈로그 정리 완료: {len(catalog)}개 유효 이벤트")
        
        if len(catalog) > 0:
            print(f"📊 카탈로그 정보:")
            print(f"  -> 컬럼들: {list(catalog.columns)}")
            if 'magnitude' in catalog.columns:
                valid_mag = catalog['magnitude'].dropna()
                if len(valid_mag) > 0:
                    print(f"  -> 규모 범위: {valid_mag.min():.1f} - {valid_mag.max():.1f}")
            
            if 'latitude' in catalog.columns and 'longitude' in catalog.columns:
                valid_coords = catalog[['latitude', 'longitude']].dropna()
                if len(valid_coords) > 0:
                    print(f"  -> 위치 범위:")
                    print(f"      위도: {valid_coords['latitude'].min():.2f}° ~ {valid_coords['latitude'].max():.2f}°")
                    print(f"      경도: {valid_coords['longitude'].min():.2f}° ~ {valid_coords['longitude'].max():.2f}°")
            
            print(f"📋 첫 3개 이벤트:")
            display_cols = [col for col in ['number', 'magnitude', 'depth', 'latitude', 'longitude'] 
                          if col in catalog.columns]
            if display_cols:
                first_3 = catalog[display_cols].head(3)
                print(first_3)
                
                # 좌표 값 상세 확인
                print(f"\n🔍 좌표 상세 정보:")
                for i in range(min(3, len(catalog))):
                    lat = catalog.iloc[i]['latitude'] if 'latitude' in catalog.columns else None
                    lon = catalog.iloc[i]['longitude'] if 'longitude' in catalog.columns else None
                    print(f"  이벤트 {i+1}: 위도={lat}, 경도={lon}")
    
    else:
        raise ValueError("컬럼이 없습니다")

except Exception as e:
    print(f"❌ CSV 처리 실패: {str(e)}")

# ============================================================================
# 2단계: 3-channel data combination
# ============================================================================
print(f"\n   Step2: 3-channel data combination")

# Define channel order
channel_order = ['vertical_BHZ', 'horizontal_1_BH1', 'horizontal_2_BH2']
combined_channels = []

print("Channel combining progress:")
for channel_name in channel_order:
    if channel_name in final_processed_data:
        data = final_processed_data[channel_name]['data']
        combined_channels.append(data)
        print(f"  ✅ {channel_name}: Add {len(data)} samples")
    else:
        print(f"  ❌ {channel_name}: No such channel")

if len(combined_channels) == 3:
    # (시간, 채널) 형태로 결합
    combined_data = np.column_stack(combined_channels)
    print(f"✅ Completed 3-channel combining: {combined_data.shape} (Time x Channel)")
    
    # 기본 통계
    print(f"  -> Data range: {combined_data.min():.3f} ~ {combined_data.max():.3f}")
    print(f"  -> Mean: {np.mean(combined_data):.3f}")
    print(f"  -> Standarad Deviation: {np.std(combined_data):.3f}")
else:
    print("❌ 3-channel combining Failed")
    combined_data = None

# ============================================================================
# Step3: Time-based windowing
# ============================================================================
print(f"\n   Step3: Time-based windowing")

# 윈도우 파라미터 설정
sampling_rate = 40  # Hz
window_duration = 20  # 초
window_samples = int(window_duration * sampling_rate)  # 800 samples
overlap_ratio = 0.5  # 50% 겹침
overlap_samples = int(window_samples * overlap_ratio)

print(f"Window setting:")
print(f"  🕐 Window length: {window_duration} seconds ({window_samples} samples)")
print(f"  🔄 Overlap: {overlap_ratio*100}% ({overlap_samples} samples)")

# Catalog-based windowing function
def create_earthquake_windows(combined_data, catalog, sampling_rate, 
                             window_samples, before_seconds=10, after_seconds=10):
    """Create window based on earthquake events"""
    
    windows = []
    labels = []
    metadata = []
    
    print(f"\nCreate window for earthquake events:")
    
    # Iterate over a pandas DataFrame (순회)
    for idx in range(len(catalog)):
        try:
            # Row Acess using iloc
            event = catalog.iloc[idx]
            magnitude = float(event['magnitude'])
            
            print(f"  📍 Event {idx+1}: M{magnitude}")
            
            # 시간 정보 (실제로는 카탈로그의 시간과 지진파 데이터의 시간을 매칭해야 함)
            # 여기서는 예시로 데이터 중앙 부분을 지진 발생 시점으로 가정
            total_samples = len(combined_data)
            earthquake_sample = total_samples // 2  # 중앙점을 지진 발생으로 가정
            
            # 지진 전후 구간 계산
            before_samples = int(before_seconds * sampling_rate)
            after_samples = int(after_seconds * sampling_rate)
            
            start_sample = earthquake_sample - before_samples
            end_sample = earthquake_sample + after_samples
            
            # 유효 범위 확인
            if start_sample >= 0 and end_sample <= total_samples:
                event_window = combined_data[start_sample:end_sample]
                
                # 윈도우가 충분히 긴지 확인
                if len(event_window) >= window_samples:
                    # 이벤트 윈도우 내에서 슬라이딩 윈도우 생성
                    event_window_count = 0
                    overlap_samples = window_samples // 2  # 50% 겹침
                    
                    for i in range(0, len(event_window) - window_samples + 1, overlap_samples):
                        window = event_window[i:i + window_samples]
                        
                        if len(window) == window_samples:
                            windows.append(window)
                            
                            # 안전하게 라벨 생성
                            label_dict = {
                                'magnitude': magnitude,
                                'depth': float(event['depth']),
                                'latitude': float(event['latitude']),
                                'longitude': float(event['longitude']),
                                'event_id': idx,
                                'window_in_event': event_window_count
                            }
                            labels.append(label_dict)
                            
                            metadata_dict = {
                                'earthquake_sample': earthquake_sample,
                                'window_start': start_sample + i,
                                'window_end': start_sample + i + window_samples,
                                'relative_to_earthquake': i - before_samples
                            }
                            metadata.append(metadata_dict)
                            
                            event_window_count += 1
                    
                    print(f"    ✅ {event_window_count} Windows created")
                else:
                    print(f"    ⚠️ Event window too short: {len(event_window)} samples")
            else:
                print(f"    ⚠️ Event is out of data range")
                
        except Exception as e:
            print(f"    ❌ 이벤트 {idx+1} 처리 중 오류: {str(e)}")
            continue
    
    return np.array(windows), labels, metadata

# 지진 이벤트 기반 윈도우 생성
print("Generating window based on earthquake event...")

# 데이터 존재 여부 안전하게 확인
if combined_data is not None:
    print(f"✅ combined_data ready: {combined_data.shape}")
else:
    print("❌ No combined_data")

try:
    if len(catalog) > 0:
        print(f"✅ catalog ready: {len(catalog)} events")
        catalog_ready = True
    else:
        print("❌ Empty catalog")
        catalog_ready = False
except:
    print("❌ There is a catalog problem")
    catalog_ready = False

# Actual window generation
if combined_data is not None and catalog_ready:
    print("Call window generation function")
    event_windows, event_labels, event_metadata = create_earthquake_windows(
        combined_data, catalog, sampling_rate, window_samples
    )
else:
    print("⚠️ Window generation conditions unsatisfied")
    event_windows = np.array([])
    event_labels = []
    event_metadata = []
    
    print(f"\n✅ Completed event-based window generation:")
    print(f"  -> Total number of windows: {len(event_windows)}")
    if len(event_windows) > 0:
        print(f"  -> Window shape: {event_windows[0].shape}")
        print(f"  -> Total shape: {event_windows.shape}")

# ============================================================================
#  Step4: Create a background noise window (non-earthquake region)
# ============================================================================
print(f"\n   Step4: Create a background noise window (non-earthquake region)")

def create_background_windows(combined_data, window_samples, overlap_samples, 
                             exclude_ranges=None):
    """Create a background noise window (non-earthquake region)"""
    
    background_windows = []
    background_metadata = []
    
    # 전체 데이터에서 지진 구간을 제외한 부분에서 윈도우 생성
    total_samples = len(combined_data)
    
    # 지진 구간 제외 (단순화를 위해 중앙 1/3 구간을 지진 구간으로 가정)
    exclude_start = total_samples // 3
    exclude_end = total_samples * 2 // 3
    
    print(f"Background noise range:")
    print(f"  🔸 Range 1: 0 ~ {exclude_start} samples")
    print(f"  🔸 Range 2: {exclude_end} ~ {total_samples} samples")
    
    # 첫 번째 구간에서 윈도우 생성
    for i in range(0, exclude_start - window_samples, overlap_samples):
        window = combined_data[i:i + window_samples]
        if len(window) == window_samples:
            background_windows.append(window)
            background_metadata.append({
                'window_start': i,
                'window_end': i + window_samples,
                'type': 'background_1'
            })
    
    # 두 번째 구간에서 윈도우 생성
    for i in range(exclude_end, total_samples - window_samples, overlap_samples):
        window = combined_data[i:i + window_samples]
        if len(window) == window_samples:
            background_windows.append(window)
            background_metadata.append({
                'window_start': i,
                'window_end': i + window_samples,
                'type': 'background_2'
            })
    
    return np.array(background_windows), background_metadata

# 배경 노이즈 윈도우 생성
if combined_data is not None:
    background_windows, background_metadata = create_background_windows(
        combined_data, window_samples, overlap_samples
    )
    
    print(f"✅ Background noise window creation complete:")
    print(f"  📊 Number of background windows: {len(background_windows)}")
    if len(background_windows) > 0:
        print(f"  📊 Window shape: {background_windows[0].shape}")

# ============================================================================
# Step5: Create data pairs for noise removal
# ============================================================================
print(f"\n   Step5: Create data pairs for noise removal")

def add_realistic_noise(clean_windows, noise_level=0.1):
    """Add realistic noise to clean signal"""
    
    noisy_windows = []
    
    print(f"Add noise in progress:")
    print(f"  🔊 Nosie level: {noise_level}")
    
    for i, clean_window in enumerate(clean_windows):
        # 1. Gaussian noise
        gaussian_noise = np.random.normal(0, noise_level * np.std(clean_window), 
                                        clean_window.shape)
        
        # 2. Power line interference (전력선 간섭) (60Hz)
        time_axis = np.arange(len(clean_window)) / sampling_rate
        power_line_noise = 0.05 * noise_level * np.sin(2 * np.pi * 60 * time_axis)
        
        # 3. Low Frequency Drift
        drift_noise = 0.02 * noise_level * np.sin(2 * np.pi * 0.1 * time_axis)
        
        # Apply different noise characteristics to each channel 
        noisy_window = clean_window.copy()
        for ch in range(clean_window.shape[1]):
            channel_noise = (gaussian_noise[:, ch] + 
                           power_line_noise * (0.5 + 0.5 * ch) +  # Different intensity for each channel
                           drift_noise[:, ch] if len(drift_noise.shape) > 1 
                           else np.broadcast_to(drift_noise, (len(drift_noise),)))
            noisy_window[:, ch] += channel_noise
        
        noisy_windows.append(noisy_window)
        
        if (i + 1) % 10 == 0 or i == len(clean_windows) - 1:
            print(f"    Progress: {i+1}/{len(clean_windows)} ({(i+1)/len(clean_windows)*100:.1f}%)")
    
    return np.array(noisy_windows)

# Use Event Window with clean data
if 'event_windows' in locals() and len(event_windows) > 0:
    clean_data = event_windows
    noisy_data = add_realistic_noise(clean_data, noise_level=0.15)
    
    print(f"✅ Data pair creation for noise removal completed:")
    print(f"  📊 Clean Data: {clean_data.shape}")
    print(f"  📊 Noisy Data: {noisy_data.shape}")
    
    # Check the effect of adding noise
    print(f"  -> Effect of adding noise:")
    print(f"    Clean std: {np.std(clean_data):.4f}")
    print(f"    Noisy std: {np.std(noisy_data):.4f}")
    print(f"    Noise ratio: {(np.std(noisy_data) - np.std(clean_data))/np.std(clean_data)*100:.1f}%")

# ============================================================================
# Step6: Final normalization and data construction
# ============================================================================
print(f"\n   Step6: Final normalization and data construction")

def normalize_data(data, method='z_score'):
    """데이터 정규화"""
    if method == 'z_score':
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / (std + 1e-8)  # Prevent division by zero
    elif method == 'min_max':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
    
    return normalized, {'mean': mean if method == 'z_score' else min_val,
                       'scale': std if method == 'z_score' else (max_val - min_val)}

# Normalize data
if 'noisy_data' in locals() and 'clean_data' in locals():
    print("Proceed data normalization:")
    
    # Compute statistics for the entire dataset
    all_data = np.concatenate([noisy_data.flatten(), clean_data.flatten()])
    
    # Z-score Normalization
    normalized_noisy, norm_stats = normalize_data(noisy_data, method='z_score')
    normalized_clean, _ = normalize_data(clean_data, method='z_score')
    
    print(f"  ✅ Completed Z-score Normalization")
    print(f"    Mean: {norm_stats['mean']:.6f}")
    print(f"    Standard deviation: {norm_stats['scale']:.6f}")
    print(f"    Range after normalization: {normalized_noisy.min():.3f} ~ {normalized_noisy.max():.3f}")

# ============================================================================
# Step7: Split train/validation data
# ============================================================================
print(f"\n   Step7: Split train/validation data")

def split_dataset(X, y, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Split train/validation data"""
    
    total_samples = len(X)
    print(f"  📊 Total samples: {total_samples}")
    
    # 최소 샘플 수 보장
    if total_samples < 3:
        print(f"  ⚠️ Too few samples ({total_samples}) for proper split")
        print(f"  🔧 Applying emergency split strategy")
        
        if total_samples == 1:
            # 1개뿐이면 모두 train에
            return {
                'train': {'X': X, 'y': y, 'indices': np.array([0])},
                'val': {'X': X[:0], 'y': y[:0], 'indices': np.array([])},
                'test': {'X': X[:0], 'y': y[:0], 'indices': np.array([])}
            }
        elif total_samples == 2:
            # 2개면 train 1개, val 1개, test 0개
            indices = np.random.permutation(total_samples)
            return {
                'train': {'X': X[indices[:1]], 'y': y[indices[:1]], 'indices': indices[:1]},
                'val': {'X': X[indices[1:2]], 'y': y[indices[1:2]], 'indices': indices[1:2]},
                'test': {'X': X[:0], 'y': y[:0], 'indices': np.array([])}
            }
    
    # 정상적인 경우 (3개 이상)
    indices = np.random.permutation(total_samples)
    
    # 최소 1개씩 보장하면서 분할
    min_val_samples = max(1, int(total_samples * val_ratio))
    min_test_samples = max(1, int(total_samples * test_ratio))
    min_train_samples = total_samples - min_val_samples - min_test_samples
    
    # 음수가 되는 경우 조정
    if min_train_samples < 1:
        min_train_samples = 1
        min_val_samples = (total_samples - 1) // 2
        min_test_samples = total_samples - min_train_samples - min_val_samples
    
    # 분할 지점 계산
    train_end = min_train_samples
    val_end = train_end + min_val_samples
    
    # Split
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return {
        'train': {'X': X[train_idx], 'y': y[train_idx], 'indices': train_idx},
        'val': {'X': X[val_idx], 'y': y[val_idx], 'indices': val_idx},
        'test': {'X': X[test_idx], 'y': y[test_idx], 'indices': test_idx}
    }

# Split data
if 'normalized_noisy' in locals() and 'normalized_clean' in locals():
    # 시드 설정으로 재현 가능한 분할
    np.random.seed(42)
    
    dataset = split_dataset(normalized_noisy, normalized_clean, 
                          train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    
    print(f"✅ Data Split Completed:")
    for split_name, split_data in dataset.items():
        print(f"  📊 {split_name.upper()}: {len(split_data['X'])} samples")
        print(f"    X shape: {split_data['X'].shape}")
        print(f"    y shape: {split_data['y'].shape}")

# ============================================================================
# Summary of final results
# ============================================================================
print(f"\n Summary of final results")
print("="*60)

if 'dataset' in locals():
    print(f"📊 Final Dataset:")
    print(f"  🎯 Progress: Seismic wave Denoising")
    print(f"  📐 Input order: {dataset['train']['X'].shape[1:]} (시간 x 채널)")
    print(f"  📈 Data Normalization: Z-score")
    print(f"  🔀 Data Split:")
    for split_name, split_data in dataset.items():
        ratio = len(split_data['X']) / (len(dataset['train']['X']) + 
                                      len(dataset['val']['X']) + 
                                      len(dataset['test']['X'])) * 100
        print(f"    - {split_name.upper()}: {len(split_data['X'])}개 ({ratio:.1f}%)")
    # print(f"\n🚀 다음 단계: 딥러닝 모델 학습 준비 완료!")
    # print(f"  💡 추천 모델: U-Net, Autoencoder, or Transformer-based denoiser")
    # print(f"  📝 사용법:")
    # print(f"    X_train = dataset['train']['X']")
    # print(f"    y_train = dataset['train']['y']")

else:
    print(f"❌ Some steps Failed - Debugging required")

print(f"\n✅ Deeplearning preprocessing completed!")




import pandas as pd
import numpy as np

print("Result data file saving starts!")
print("="*60)

def save_earthquake_data_to_csv(dataset):
    """딥러닝 전처리된 지진파 데이터를 CSV로 저장"""
    
    def reshape_and_save(data_X, data_y, filename_prefix):
        """3D 지진파 데이터를 2D CSV로 변환하여 저장"""
        
        print(f"💾 {filename_prefix} 세트 저장 중...")
        
        n_samples, n_time, n_channels = data_X.shape
        print(f"  📊 형태: {n_samples}개 샘플 × {n_time}시점 × {n_channels}채널")
        
        # === X 데이터 (노이즈 있는 데이터) 저장 ===
        # (1664, 800, 3) → (1664, 2400) 형태로 변환
        X_reshaped = data_X.reshape(n_samples, -1)
        
        # 컬럼명 생성: t0_ch0, t0_ch1, t0_ch2, t1_ch0, t1_ch1, t1_ch2, ...
        X_columns = []
        for t in range(n_time):
            for ch in range(n_channels):
                channel_name = ['BHZ', 'BH1', 'BH2'][ch]  # 실제 채널명 사용
                X_columns.append(f't{t}_{channel_name}')
        
        # DataFrame 생성 및 저장
        X_df = pd.DataFrame(X_reshaped, columns=X_columns)
        X_df.to_csv(f'{filename_prefix}_X_noisy.csv', index=False)
        print(f"  ✅ {filename_prefix}_X_noisy.csv 저장완료 ({X_df.shape})")
        
        # === y 데이터 (깨끗한 데이터) 저장 ===
        y_reshaped = data_y.reshape(n_samples, -1)
        
        # 같은 컬럼명 사용
        y_columns = X_columns  # 동일한 구조
        
        y_df = pd.DataFrame(y_reshaped, columns=y_columns)
        y_df.to_csv(f'{filename_prefix}_y_clean.csv', index=False)
        print(f"  ✅ {filename_prefix}_y_clean.csv 저장완료 ({y_df.shape})")
        
        return X_df.shape, y_df.shape
    
    # 각 데이터셋 저장
    total_saved = 0
    
    for split_name in ['train', 'val', 'test']:
        if split_name in dataset and len(dataset[split_name]['X']) > 0:
            X_shape, y_shape = reshape_and_save(
                dataset[split_name]['X'], 
                dataset[split_name]['y'], 
                split_name
            )
            total_saved += X_shape[0]
        else:
            print(f"⚠️ {split_name} 데이터가 없습니다.")
    
    return total_saved

# 메타데이터 저장
def save_metadata_csv(dataset):
    """데이터셋 메타데이터를 CSV로 저장"""
    
    print(f"\n📊 메타데이터 저장 중...")
    
    # 기본 정보
    metadata = []
    
    for split_name in ['train', 'val', 'test']:
        if split_name in dataset:
            split_data = dataset[split_name]
            metadata.append({
                'split': split_name,
                'samples': len(split_data['X']),
                'time_steps': split_data['X'].shape[1] if len(split_data['X']) > 0 else 0,
                'channels': split_data['X'].shape[2] if len(split_data['X']) > 0 else 0,
                'total_features': split_data['X'].shape[1] * split_data['X'].shape[2] if len(split_data['X']) > 0 else 0
            })
    
    # 전체 정보 추가
    total_samples = sum(len(dataset[split]['X']) for split in ['train', 'val', 'test'] if split in dataset)
    
    metadata.append({
        'split': 'TOTAL',
        'samples': total_samples,
        'time_steps': 800,
        'channels': 3,
        'total_features': 2400
    })
    
    # 설정 정보 추가
    settings = pd.DataFrame([
        {'parameter': 'sampling_rate', 'value': '40 Hz'},
        {'parameter': 'window_duration', 'value': '20 seconds'},
        {'parameter': 'normalization', 'value': 'Z-score'},
        {'parameter': 'noise_level', 'value': '15%'},
        {'parameter': 'original_events', 'value': '1664'},
        {'parameter': 'channels', 'value': 'BHZ, BH1, BH2'},
        {'parameter': 'data_split', 'value': '70% train, 20% val, 10% test'}
    ])
    
    # 저장
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv('dataset_metadata.csv', index=False)
    settings.to_csv('dataset_settings.csv', index=False)
    
    print(f"  ✅ dataset_metadata.csv 저장완료")
    print(f"  ✅ dataset_settings.csv 저장완료")
    
    return metadata_df

# 실행
if 'dataset' in locals():
    print(f"🎯 현재 데이터셋 상태:")
    for split_name in ['train', 'val', 'test']:
        if split_name in dataset:
            print(f"  📚 {split_name}: {len(dataset[split_name]['X'])}개 샘플")
    
    # CSV 저장 실행
    total_saved = save_earthquake_data_to_csv(dataset)
    metadata_df = save_metadata_csv(dataset)
    """""
    print(f"\n🎉 CSV 저장 완료!")
    print(f"  📊 총 저장된 샘플: {total_saved}개")
    print(f"  📄 생성된 파일들:")
    print(f"    - train_X_noisy.csv (노이즈 있는 훈련 데이터)")
    print(f"    - train_y_clean.csv (깨끗한 훈련 데이터)")
    print(f"    - val_X_noisy.csv (노이즈 있는 검증 데이터)")
    print(f"    - val_y_clean.csv (깨끗한 검증 데이터)")
    print(f"    - test_X_noisy.csv (노이즈 있는 테스트 데이터)")
    print(f"    - test_y_clean.csv (깨끗한 테스트 데이터)")
    print(f"    - dataset_metadata.csv (데이터셋 정보)")
    print(f"    - dataset_settings.csv (설정 정보)")
    """""
    
    # 파일 크기 확인
    import os
    print(f"\n📏 파일 크기:")
    csv_files = [
        'train_X_noisy.csv', 'train_y_clean.csv',
        'val_X_noisy.csv', 'val_y_clean.csv', 
        'test_X_noisy.csv', 'test_y_clean.csv'
    ]
    
    for file in csv_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / 1024 / 1024
            print(f"  📄 {file}: {size_mb:.1f} MB")
    
    print(f"\n💡 사용법:")
    print(f"  # 데이터 불러오기")
    print(f"  train_X = pd.read_csv('train_X_noisy.csv')")
    print(f"  train_y = pd.read_csv('train_y_clean.csv')")
    print(f"  # 딥러닝 학습에 사용!")
    
else:
    print("❌ 'dataset' 변수를 찾을 수 없습니다.")
    print("💡 먼저 딥러닝 전처리를 완료해주세요.")


## 딥러닝용으로 변환
import pandas as pd
import numpy as np

print("🚀 CSV → 딥러닝 데이터 변환 시작!")

def csv_to_deeplearning_ready(csv_prefix_list=['train', 'val', 'test']):
    """CSV에서 딥러닝 준비 완료 데이터로 한 번에 변환"""
    
    dataset = {}
    
    for prefix in csv_prefix_list:
        print(f"🔄 {prefix} 데이터 로딩 중...")
        
        try:
            # CSV 파일 읽기
            X_file = f'{prefix}_X_noisy.csv'
            y_file = f'{prefix}_y_clean.csv'
            
            X_df = pd.read_csv(X_file)
            y_df = pd.read_csv(y_file)
            
            # 3D 변환: (샘플, 2400) → (샘플, 800, 3)
            n_samples = len(X_df)
            X_3d = X_df.values.reshape(n_samples, 800, 3)
            y_3d = y_df.values.reshape(n_samples, 800, 3)
            
            # 딕셔너리에 저장
            dataset[prefix] = {
                'X': X_3d,  # 노이즈 있는 데이터
                'y': y_3d   # 깨끗한 데이터
            }
            
            print(f"  ✅ {prefix}: {X_3d.shape} → {y_3d.shape}")
            
        except FileNotFoundError:
            print(f"  ❌ {prefix} 파일들을 찾을 수 없습니다.")
        except Exception as e:
            print(f"  ❌ {prefix} 처리 중 오류: {e}")
    
    return dataset

# 실행!
dl_dataset = csv_to_deeplearning_ready()

# 결과 확인
if dl_dataset:
    print(f"\n🎉 변환 완료!")
    for split_name, data in dl_dataset.items():
        print(f"  📊 {split_name}: {data['X'].shape} (노이즈) → {data['y'].shape} (깨끗함)")
    
    print(f"\n💡 사용법:")
    print(f"  train_X = dl_dataset['train']['X']")
    print(f"  train_y = dl_dataset['train']['y']")
    print(f"  # 이제 딥러닝 모델에 바로 사용 가능! 🚀")
else:
    print("❌ 변환 실패")
