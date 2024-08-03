import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import os

# Hàm trích xuất biên dạng âm thanh
def amplitude_envelope(signal, frame_size, hop_length):
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])

# Hàm tính toán Root Mean Square Energy (RMS)
def rmse(signal, frame_size, hop_length):
    rmse_values = []
    for i in range(0, len(signal), hop_length):
        rmse_current_frame = np.sqrt(sum(signal[i:i+frame_size]**2) / frame_size)
        rmse_values.append(rmse_current_frame)
    return np.array(rmse_values)

# Hàm tính toán Zero Crossing Rate (ZCR)
def zero_crossing_rate(signal, frame_size, hop_length):
    zcr_values = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_length)[0]
    return zcr_values

# Hàm tính toán Spectral Centroid (SC)
def spectral_centroid(signal, sample_rate, frame_size, hop_length):
    sc_values = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)[0]
    return sc_values

# Hàm tính toán Spectral Bandwidth (SB)
def spectral_bandwidth(signal, sample_rate, frame_size, hop_length):
    sb_values = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)[0]
    return sb_values

# Hàm tính toán Band Energy Ratio (BER)
def band_energy_ratio(spectrogram, split_frequency, sample_rate):
    split_frequency_bin = librosa.fft_frequencies(sr=sample_rate)[:len(spectrogram[0])].searchsorted(split_frequency)
    band_energy_ratio_values = []
    power_spectrogram = np.abs(spectrogram) ** 2
    power_spectrogram = power_spectrogram.T
    for frame in power_spectrogram:
        sum_power_low_frequencies = frame[:split_frequency_bin].sum()
        sum_power_high_frequencies = frame[split_frequency_bin:].sum()
        if sum_power_high_frequencies:
            band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        else:
            band_energy_ratio_current_frame = 0
        band_energy_ratio_values.append(band_energy_ratio_current_frame)
    return np.array(band_energy_ratio_values)

# Hàm trích xuất đặc trưng MFCC
def extract_mfccs(signal, sample_rate):
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate)
    mfccs_median = np.median(mfccs, axis=0)
    return mfccs_median

# Hàm trích xuất đặc trưng và tính toán trọng số
def extract_features(audio_file, sample_rate, frame_size, hop_length):
    ae = amplitude_envelope(audio_file, frame_size, hop_length)
    rms = rmse(audio_file, frame_size, hop_length)
    zcr = zero_crossing_rate(audio_file, frame_size, hop_length)
    sc = spectral_centroid(audio_file, sample_rate, frame_size, hop_length)
    sb = spectral_bandwidth(audio_file, sample_rate, frame_size, hop_length)
    ber = band_energy_ratio(librosa.stft(y=audio_file, n_fft=frame_size, hop_length=hop_length), 2000, sample_rate)
    mfccs = extract_mfccs(audio_file, sample_rate)
    
    # Áp dụng trọng số cho các đặc trưng
    ae_weight = 1.0
    rms_weight = 0.8
    zcr_weight = 0.5
    sc_weight = 0.7
    sb_weight = 0.6
    ber_weight = 0.9
    mfccs_weight = 1.2
    
    # Tính toán tổng trọng số
    total_weight = (ae_weight + rms_weight + zcr_weight + sc_weight + sb_weight + ber_weight + mfccs_weight)
    
    # Tính toán giá trị đặc trưng trọng số
    weighted_ae = ae * (ae_weight / total_weight)
    weighted_rms = rms * (rms_weight / total_weight)
    weighted_zcr = zcr * (zcr_weight / total_weight)
    weighted_sc = sc * (sc_weight / total_weight)
    weighted_sb = sb * (sb_weight / total_weight)
    weighted_ber = ber * (ber_weight / total_weight)
    weighted_mfccs = mfccs * (mfccs_weight / total_weight)
    
    # Kết hợp các giá trị đặc trưng trọng số
    features = np.vstack((weighted_ae, weighted_rms, weighted_zcr, weighted_sc, weighted_sb, weighted_ber, weighted_mfccs)).T
    
    return features

# Hàm tính khoảng cách giữa query và một phần của data
def calculate_distance(query, part_of_data):
    deviation = np.abs(query - part_of_data)
    total_deviation = np.sum(deviation)
    return total_deviation

# Hàm tính toán khoảng cách nhỏ nhất giữa query và data
def calculate_min_distance(query, data):
    min_distance = float('inf')
    query_length = len(query)
    data_length = len(data)
    
    for i in range(data_length - query_length + 1):
        distance = calculate_distance(query, data[i:i+query_length])
        min_distance = min(min_distance, distance)
    
    return min_distance

# Hàm tìm kiếm và sắp xếp các bản nhạc dựa trên đặc trưng của bản nhạc truy vấn
def search(query_features, data_features, file_names):
    similarities = []
    for i in range(len(data_features)):
        similarity = calculate_min_distance(query_features, data_features[i])
        similarities.append((file_names[i], similarity))
    
    sorted_similarities = sorted(similarities, key=lambda x: x[1])
    return sorted_similarities
import os
import librosa
import numpy as np

import os
import librosa
import numpy as np

def load_and_extract_features(data_dir, frame_size, hop_length):
    data_features = []
    file_names = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            data_signal, data_sr = librosa.load(os.path.join(data_dir, file_name))
            data_feature = extract_features(data_signal, data_sr, frame_size, hop_length)
            if len(data_feature) < len(data_features):
                data_feature = np.pad(data_feature, ((0, len(data_features) - len(data_feature)), (0, 0)), mode='constant', constant_values=0)
            elif len(data_feature) > len(data_features):
                data_feature = data_feature[:len(data_features), :]
            data_features.append(data_feature)
            file_names.append(file_name)

    # Chuyển đổi danh sách các đặc trưng của dữ liệu thành mảng numpy
    data_features = np.array(data_features)

    return data_features, file_names


def main():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính của tkinter

    # Chọn tệp âm thanh truy vấn từ cửa sổ
    query_audio_file = filedialog.askopenfilename(title="Chọn file âm thanh truy vấn", filetypes=[("Audio Files", "*.wav")])

    if not query_audio_file:
        print("Không có tệp âm thanh truy vấn được chọn.")
        return

    # Load bản nhạc cần tìm bản gốc
    query_signal, query_sr = librosa.load(query_audio_file)

    # Thiết lập kích thước cửa sổ và bước nhảy cho việc trích xuất đặc trưng
    frame_size = 2048
    hop_length = 512

    # Trích xuất đặc trưng từ bản nhạc cần tìm bản gốc
    query_features = extract_features(query_signal, query_sr, frame_size, hop_length)

    # Chọn thư mục chứa dữ liệu âm thanh
    data_dir = filedialog.askdirectory(title="Chọn thư mục chứa dữ liệu âm thanh")

    if not data_dir:
        print("Không có thư mục dữ liệu âm thanh được chọn.")
        return

    # Load và trích xuất đặc trưng từ tất cả các bản nhạc trong thư mục dữ liệu
    data_features, file_names = load_and_extract_features(data_dir, frame_size, hop_length)

    # Tìm kiếm và xếp hạng các bản nhạc dựa trên đặc trưng của bản nhạc cần tìm bản gốc
    sorted_similarities = search(query_features, data_features, file_names)

    # In ra kết quả
    for file_name, similarity in sorted_similarities[:3]:
        print(f"{file_name}: Similarity Score: {similarity}")

if __name__ == "__main__":
    main()
