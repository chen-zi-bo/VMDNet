import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader  # 假设您会使用这个来创建DataLoader
# matplotlib.use('Qt5Agg')

# 假设 TimeSeriesDataset 是一个基类，您需要根据实际情况定义或导入
# 例如:
from torch_timeseries.datasets.dataset import TimeSeriesDataset


class CustomVaryFreq(TimeSeriesDataset):
    name: str = 'CustomVaryFreq'
    freq: str = 't'  # 时间频率，例如 't' for time steps
    num_features: int = 1  # 简化为1个特征，便于观察分解效果
    length: int = 1000  # 适当的长度，便于观察时频变化
    seed: int = 42


    signal_components_params: list = [
        {'amplitude': 1.0, 'start_freq': 0.02, 'end_freq': 0.05, 'phase_offset': 0},  # 频率范围较低
        {'amplitude': 0.7, 'start_freq': 0.10, 'end_freq': 0.15, 'phase_offset': np.pi / 4},  # 频率范围中等，与第一个有明显间隔
        # {'amplitude': 0.5, 'start_freq': 0.20, 'end_freq': 0.25, 'phase_offset': np.pi/2}, # 可选的第三个成分，频率范围较高
    ]

    def _load(self):
        np.random.seed(self.seed)

        dates = pd.date_range(start='2022-01-01', periods=self.length, freq=self.freq)
        data = np.zeros((self.length, self.num_features))
        t = np.arange(self.length) / self.length  # 归一化时间轴 [0, 1)

        for f_idx in range(self.num_features):
            composite_feature_signal = np.zeros(self.length)

            for params in self.signal_components_params:
                amplitude = params['amplitude']
                start_freq = params['start_freq']
                end_freq = params['end_freq']
                phase_offset = params['phase_offset']  # 固定相位偏移

                # 线性变化的瞬时频率
                instantaneous_freq = start_freq + (end_freq - start_freq) * t

                # 累积相位：∫(2 * pi * f(t)) dt
                total_phase = (start_freq * t + 0.5 * (end_freq - start_freq) * t ** 2) * (2 * np.pi * self.length)

                # 添加随机初始相位，确保每次生成的数据略有不同
                component_signal = amplitude * np.sin(total_phase + phase_offset + np.random.rand() * 2 * np.pi)
                composite_feature_signal += component_signal

            # 添加高斯白噪声
            noise = np.random.normal(0, 0.1, self.length)  # 噪声标准差 0.1
            data[:, f_idx] = composite_feature_signal + noise

        self.df = pd.DataFrame(data, columns=[f"data{i}" for i in range(self.num_features)])
        self.df['date'] = dates
        self.dates = pd.DataFrame({'date': dates})
        self.data = self.df.drop('date', axis=1).values  # 确保 self.data 是纯数据 NumPy 数组

        print(f"Dataset '{self.name}' generated with shape: {self.data.shape}")
        return self.data


# # --- 使用示例 ---
# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#
#     print("\n--- 2 Components Signal Example with Increased Spacing ---")
#     # 示例1：默认的2个线性变频成分，频率间距加大
#     # Component 1: Freq 0.02 -> 0.05
#     # Component 2: Freq 0.10 -> 0.15
#     # 两个成分之间有 0.05 的起始频率间隔
#     dataset_2_comp_spaced = CustomVaryFreq()
#     loader_2_comp_spaced = DataLoader(dataset_2_comp_spaced, batch_size=1, shuffle=False)
#
#     # 绘制原始信号的第一个特征
#     plt.figure(figsize=(12, 4))
#     plt.plot(dataset_2_comp_spaced.data[:, 0])
#     plt.title('Synthetic Signal (2 Varying Freq Components, Widely Spaced)')
#     plt.xlabel('Time Step')
#     plt.ylabel('Amplitude')
#     plt.grid(True)
#     plt.show()
#
#     print("\n--- 3 Components Signal Example with Increased Spacing ---")
#     # 示例2：3个线性变频成分，频率间距加大
#     # Component 1: Freq 0.01 -> 0.04
#     # Component 2: Freq 0.06 -> 0.10
#     # Component 3: Freq 0.12 -> 0.16
#     params_3_comp_spaced = [
#         {'amplitude': 1.0, 'start_freq': 0.01, 'end_freq': 0.04, 'phase_offset': 0},
#         {'amplitude': 0.8, 'start_freq': 0.06, 'end_freq': 0.10, 'phase_offset': np.pi / 3},
#         {'amplitude': 0.6, 'start_freq': 0.12, 'end_freq': 0.16, 'phase_offset': np.pi / 2},
#     ]
#     dataset_3_comp_spaced = CustomVaryFreq()
#     loader_3_comp_spaced = DataLoader(dataset_3_comp_spaced, batch_size=1, shuffle=False)
#
#     plt.figure(figsize=(12, 4))
#     plt.plot(dataset_3_comp_spaced.data[:, 0])
#     plt.title('Synthetic Signal (3 Varying Freq Components, Widely Spaced)')
#     plt.xlabel('Time Step')
#     plt.ylabel('Amplitude')
#     plt.grid(True)
#     plt.show()
#
#     # 简要检查 DataLoader 的输出
#     first_batch_x, _, _, _, _, _ = next(iter(loader_2_comp_spaced))
#     print(f"First batch from DataLoader shape: {first_batch_x.shape}")


