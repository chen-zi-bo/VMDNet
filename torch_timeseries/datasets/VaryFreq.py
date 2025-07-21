import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# matplotlib.use('Qt5Agg')
from .dataset import TimeSeriesDataset


class VaryFreq(TimeSeriesDataset):
    name: str = 'VaryFreq'
    num_features: int = 1
    sample_rate: int = 1
    length : int = 2000
    freq: str = 't'

    def download(self):
        pass

    def _load(self):
        dates = pd.date_range(start='2022-01-01', periods=self.length, freq=self.freq)
        data = np.zeros((self.length, self.num_features))
        t = np.arange(self.length)  # 时间步序列

        # --- 定义两个分量的参数变化趋势 ---
        # 我们定义关键点处的幅值和周期，然后通过线性插值实现平滑过渡
        # key_points_t: 关键时间点在总长度中的比例
        key_points_t = [0, 0.3, 0.7, 1.0]  # 0%, 30%, 70%, 100% 处定义关键值
        key_t_indices = [int(p * self.length) for p in key_points_t]
        key_t_indices[-1] = self.length  # 确保最后一个点是精确的末尾

        # Component 1 参数在关键点的数值
        # (时间点0, 时间点30%, 时间点70%, 时间点100%)
        amp1_key_values = [0.8, 2.5, 0.5, 1.2]  # 幅值从低到高再到低再到中
        period1_key_values = [10, 40, 15, 30]  # 周期从高频->低频->中高频->中频

        # Component 2 参数在关键点的数值 (增加复杂性)
        amp2_key_values = [0.2, 0.6, 1.0, 0.3]  # 幅值从低到中到高再到低
        period2_key_values = [70, 30, 90, 50]  # 周期从低频->中高频->超低频->中频

        # --- 对幅值和周期进行平滑插值 ---
        # numpy.interp 用于一维线性插值
        amp1_t = np.interp(t, key_t_indices, amp1_key_values)
        period1_t = np.interp(t, key_t_indices, period1_key_values)

        amp2_t = np.interp(t, key_t_indices, amp2_key_values)
        period2_t = np.interp(t, key_t_indices, period2_key_values)

        # --- 生成信号 ---
        # 避免周期为0，并计算角频率
        w1_t = 2 * np.pi / (period1_t + 1e-6)
        w2_t = 2 * np.pi / (period2_t + 1e-6)

        # 通过累积和来计算相位，确保频率变化时的波形连续性
        signal_c1 = amp1_t * np.sin(np.cumsum(w1_t))
        signal_c2 = amp2_t * np.sin(np.cumsum(w2_t))

        # 叠加两个分量，并加入少量随机高斯噪声
        feature_signal = signal_c1 + signal_c2 + np.random.normal(0, 0.05, self.length)

        data[:, 0] = feature_signal  # 存入第一个也是唯一的特征

        self.df = pd.DataFrame(data, columns=[f"data0"])
        self.df['date'] = dates
        self.dates = pd.DataFrame({'date': dates})
        self.data = self.df.drop('date', axis=1).values

        print(f"Dataset '{self.name}' generated with shape: {self.data.shape} (Single Feature, Continuous Frequency)")
        return self.data
# # --- Visualization Code ---
# # 实例化数据集
# # 可以调整 length 或 num_features 来测试不同的情况
# dataset_sim = VaryFreq() # 使用较短的 length 方便可视化
#
# # 获取数据
# sim_data = dataset_sim.data # 这是 NumPy 数组，shape: [length, num_features]
#
# # 可视化一个或多个特征，以展示非平稳性
# plt.figure(figsize=(15, 8))
#
# for i in range(dataset_sim.num_features):
#     plt.subplot(dataset_sim.num_features, 1, i + 1) # 创建子图
#     plt.plot(sim_data[:, i], label=f'Feature {i+1}', linewidth=0.8) # 调整线宽让图更清晰
#     plt.title(f'Simulated Time Series - Feature {i+1} (Non-Stationary)', fontsize=12)
#     plt.xlabel('时间步 (Time Step)', fontsize=10)
#     plt.ylabel('幅值 (Amplitude)', fontsize=10)
#     plt.grid(True, linestyle='--', alpha=0.6)
#
#     # 标记出阶段边界，帮助可视化理解非平稳性
#     phase_ratios = [0.4, 0.3, 0.3]
#     phase_lengths_vis = [int(dataset_sim.length * r) for r in phase_ratios]
#     current_offset = 0
#     for p_len in phase_lengths_vis[:-1]: # 只绘制前两个边界
#         current_offset += p_len
#         # 只在第一个子图上显示图例，避免重复
#         plt.axvline(x=current_offset, color='red', linestyle='--', alpha=0.7,
#                     label='阶段变化点 (Phase Change)' if i == 0 else "")
#     plt.legend(fontsize=8)
#
# plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
# plt.show()