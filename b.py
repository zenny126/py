import os
import numpy as np
import librosa
from sklearn.metrics.pairwise import euclidean_distances

# def extract_features(audio_file):
#     # Load audio file
#     y, sr = librosa.load(audio_file)
    
#     # Extract features
#     AE = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
#     RMS = np.mean(librosa.feature.rms(y=y))
#     ZCR = np.mean(librosa.feature.zero_crossing_rate(y))
#     SC = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
#     bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
#     BER = np.mean(librosa.feature.spectral_flatness(y=y))
#     chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
#     mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
    
#     # Normalize features
#     AE = (AE - np.mean(AE)) / np.std(AE)
#     RMS = (RMS - np.mean(RMS)) / np.std(RMS)
#     ZCR = (ZCR - np.mean(ZCR)) / np.std(ZCR)
#     SC = (SC - np.mean(SC)) / np.std(SC)
#     bandwidth = (bandwidth - np.mean(bandwidth)) / np.std(bandwidth)
#     BER = (BER - np.mean(BER)) / np.std(BER)
#     chroma = (chroma - np.mean(chroma)) / np.std(chroma)
#     mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    
#     # Concatenate features
#     features = np.concatenate((np.array([AE, RMS, ZCR, SC, bandwidth, BER]), chroma, mfccs))
#     return features

# Function to preprocess audio features
# def preprocess_features(features):
#     # Chuẩn hóa đặc trưng để có giá trị trung bình bằng 0 và độ lệch chuẩn bằng 1
#     normalized_features = (features - np.mean(features)) / np.std(features)
#     return normalized_features

# # Giai đoạn 1: Trích xuất đặc trưng từ file âm thanh 
# def extract_features(audio_file):
#     y, sr = librosa.load(audio_file)
#     # Tiền xử lý đặc trưng trước khi trả về
#     preprocessed_features = preprocess_features(np.array([
#         np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
#         np.mean(librosa.feature.rms(y=y)),
#         np.mean(librosa.feature.zero_crossing_rate(y)),
#         np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
#         np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
#         np.mean(librosa.feature.spectral_flatness(y=y)),
#         *np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1),
#         *np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
#     ]))
#     return preprocessed_features

# Giai đoạn 1: Trích xuất đặc trưng từ file âm thanh 
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    AE = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    RMS = np.mean(librosa.feature.rms(y=y))
    ZCR = np.mean(librosa.feature.zero_crossing_rate(y))
    SC = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    BER = np.mean(librosa.feature.spectral_flatness(y=y))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
    return np.concatenate((np.array([AE, RMS, ZCR, SC, bandwidth, BER]), chroma, mfccs))
    # mfccs = librosa.feature.mfcc(y=y, sr=sr)
    # return np.mean(mfccs.T, axis=0)

# Giai đoạn 2: So sánh độ tương đồng giữa các đặc trưng
# def compare_features(query_features, data_features):
#     distance = euclidean_distances([query_features], [data_features])[0][0]
#     # Chuyển đổi khoảng cách thành độ tương đồng (càng gần 0 càng tương đồng)
#     similarity = 1 / (1 + distance)
#     return similarity
# Giai đoạn 2: So sánh độ tương đồng giữa các đặc trưng




# Giai đoạn 2: So sánh độ tương đồng giữa các đặc trưng
def compare_features(query_features, data_features):
    min_distance = float('inf')  # Khởi tạo khoảng cách nhỏ nhất với giá trị vô cực
    len_query = len(query_features)
    len_datafeatures= len(data_features)
    if(len_datafeatures<len_query) :
        return 0
    for i in range(len(data_features) - len_query + 1):
        # Lấy cụm cửa sổ đặc trưng tương ứng từ file trong dữ liệu
        data_window = data_features[i:i+len_query]
        # Tính khoảng cách Euclid giữa cụm cửa sổ đặc trưng của file trong dữ liệu và của file truy vấn
        distance = np.linalg.norm(np.array(data_window) - np.array(query_features))
        # Cập nhật khoảng cách nhỏ nhất
        if distance < min_distance:
            min_distance = distance
    # Trả về khoảng cách nhỏ nhất
    similarity = 1 / (1 + min_distance) *len_query/len_datafeatures
    return similarity
    # return min_distance


# Giai đoạn 3: Tìm kiếm và xếp hạng các file âm thanh
def search_audio(query_features, data_dir, top_k=3):
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    similarities = []
    for data_file in data_files:
        data_features = extract_features(os.path.join(data_dir, data_file))
        similarity = compare_features(query_features, data_features)
        similarities.append((data_file, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:top_k]
    return top_matches

# Example usage
def main():
    query_audio_file = "blues.000071.wav"
    data_dir = "data"
    query_features = extract_features(query_audio_file)
    top_matches = search_audio(query_features, data_dir)
    print("Top matching audio files:")
    for idx, (audio_file, similarity) in enumerate(top_matches, 1):
        print(f"{idx}. {audio_file} - Similarity: {similarity:.4f}")

if __name__ == "__main__":
    main()
