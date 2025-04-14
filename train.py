# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# Modified for Bitcoin Price Forecasting

import argparse
import os
import torch
import numpy as np
from modeling import TemporalFusionTransformer
from data_loader import MultiModalDataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    # 数据加载
    loader = MultiModalDataLoader(window_size=24, horizon=1)  # 可扩展其他参数 
    X, y = loader.load_and_process()  # 假设已实现
    
    # 数据集划分
    train_size = int(0.8 * len(X))
    train_X, train_y = X[:train_size], y[:train_size]
    test_X, test_y = X[train_size:], y[train_size:]
    
    # 模型初始化
    config = argparse.Namespace(d_model=64)  # 可扩展其他参数
    model = TemporalFusionTransformer(config)
    criterion = torch.nn.MSELoss()          # MSE损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    # 测试
    model.eval()
    with torch.no_grad():
        preds = model(test_X)
        test_loss = criterion(preds, test_y)
        print(f'Test Loss: {test_loss.item():.4f}')

if __name__ == '__main__':
    main()