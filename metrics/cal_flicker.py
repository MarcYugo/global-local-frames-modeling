import cv2
import numpy as np
import argparse
from scipy.fft import fft
import matplotlib.pyplot as plt

# 读取视频并提取帧
def read_video(file_path):
    video = cv2.VideoCapture(file_path)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames

# 计算每帧的亮度（使用灰度图）
def compute_brightness(frames):
    brightness = []
    for frame in frames:
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算图像亮度（平均灰度值）
        brightness.append(np.mean(gray))
    return np.array(brightness)

# 计算时间域亮度变化（相邻帧亮度差异）
def compute_temporal_variation(brightness):
    brightness_diff = np.abs(np.diff(brightness))
    return brightness_diff

# 计算频域变化（低频闪烁成分）
def compute_frequency_variation(brightness, frame_rate):
    N = len(brightness)
    fft_result = fft(brightness)
    freqs = np.fft.fftfreq(N, d=1/frame_rate)
    
    # 只取正频率部分
    positive_freqs = freqs[:N//2]
    positive_fft = np.abs(fft_result[:N//2])
    
    # 过滤出低频部分（假设闪烁主要发生在低频段）
    low_freqs = (positive_freqs > 0.5) & (positive_freqs < 10)  # 0.5Hz到10Hz的频率范围
    low_freq_power = np.sum(positive_fft[low_freqs])
    
    return low_freq_power

# 计算 Flicker Value
def compute_flicker_value(brightness, frame_rate):
    # 时间域变化（相邻帧亮度差）
    temporal_variation = compute_temporal_variation(brightness)
    
    # 频域变化（低频闪烁成分）
    frequency_variation = compute_frequency_variation(brightness, frame_rate)
    
    # 计算 Flicker Value，结合时间域变化和频域变化
    flicker_value = np.mean(temporal_variation) + frequency_variation
    return flicker_value

# 主程序：读取视频并计算 Flicker Value
def main(video_path, frame_rate=30):
    # 读取视频帧
    frames = read_video(video_path)
    
    # 计算每帧的亮度
    brightness = compute_brightness(frames)
    
    # 计算 Flicker Value
    flicker_value = compute_flicker_value(brightness, frame_rate)
    
    print(f'Flicker Value of the {video_path}: {flicker_value}')

    # 可视化亮度变化（可选）
    plot_brightness(brightness)

# 可视化亮度变化
def plot_brightness(brightness):
    plt.plot(brightness)
    plt.title('Brightness Variation Across Frames')
    plt.xlabel('Frame')
    plt.ylabel('Brightness')
    plt.show()

if __name__ == '__main__':
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='Detect Flicker Value in a Video')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--frame_rate', type=int, default=10, help='Frame rate of the video (default: 30)')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主程序
    main(args.video_path, args.frame_rate)
