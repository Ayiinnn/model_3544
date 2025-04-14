import pandas as pd
import numpy as np
import torch

'''
提前处理好并存储是不是好一点
'''
class MultiModalDataLoader:
    def __init__(self, window_size=24, horizon=1):
        self.window_size = window_size  # 输入窗口长度（小时）
        self.horizon = horizon          # 预测未来时间步（小时）

    def load_data(self):
        # 读取三个数据集并按时间对齐
        finance = pd.read_csv('datasets/finance.csv', parse_dates=['timestamp'])
        media = pd.read_csv('datasets/media.csv', parse_dates=['timestamp'])
        fear_greed = pd.read_csv('datasets/fear_greed.csv', parse_dates=['Timestamp'])
        
        # 统一时间戳列名并合并
        finance = finance.rename(columns={'timestamp': 'time'})
        media = media.rename(columns={'timestamp': 'time'})
        fear_greed = fear_greed.rename(columns={'Timestamp': 'time'})
        
        # 按小时对齐数据（左连接以金融数据为基准）
        merged = finance.merge(media, on='time', how='left')
        merged = merged.merge(fear_greed[['time', 'Value']], on='time', how='left')
        
        # 填充缺失值（前向填充）
        merged.fillna(method='ffill', inplace=True)
        return merged

    def sliding_window(self, data):
        features = data.drop(columns=['time']).values
        num_samples = len(features) - self.window_size - self.horizon
        
        # 预分配numpy数组
        sequences = np.zeros((num_samples, self.window_size, features.shape[1]))
        targets = np.zeros(num_samples)
        
        for i in range(num_samples):
            sequences[i] = features[i:i+self.window_size]
            targets[i] = features[i+self.window_size+self.horizon-1, 0]
        
        # 一次性转换为张量
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)

# 使用示例
loader = MultiModalDataLoader(window_size=24, horizon=1)
data = loader.load_data()
X, y = loader.sliding_window(data)
print(f'输入形状: {X.shape}, 输出形状: {y.shape}')  # [样本数, 24小时, 24特征], [样本数]
