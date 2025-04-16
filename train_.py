import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from modeling import TemporalFusionTransformer
from sklearn.model_selection import train_test_split


BATCH_SIZE = 8
EPOCHS = 100
DEVICE = torch.device("cuda")
CLIP_GRADIENT = True  
CLIP_THRESHOLD = 1.0  
TEST_SIZE = 0.2  
SEED = 42  
LEARNING_RATE = 1e-3


class TimeSeriesDataset(Dataset):  #可能要修改成延迟加载
    def __init__(self, samples, targets):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]
      
class Config:
    hidden_size = 16
    d_model = 16 
    dropout = 0.1
    fin_varible_num = 13  
  
samples = np.load('sample.npy')  
targets = np.load('target.npy')  
train_samples, test_samples, train_targets, test_targets = train_test_split(
    samples, targets, 
    test_size=TEST_SIZE,
    random_state=SEED
)


train_dataset = TimeSeriesDataset(train_samples, train_targets)
test_dataset = TimeSeriesDataset(test_samples, test_targets)

train_loader = DataLoader(train_dataset, 
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         pin_memory=True,
                         drop_last=False)  

test_loader = DataLoader(test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        pin_memory=True)

config = Config()
model = TemporalFusionTransformer(config).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_test_loss = float('inf')  # 最佳测试损失


for epoch in range(EPOCHS):
    # 训练
    model.train()
    train_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # 前向
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if CLIP_GRADIENT:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_THRESHOLD)
        
        optimizer.step()
        train_loss += loss.item()
    
    # 测试
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)
    
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), f"best_model.pth")
    
    # 保存
    torch.save({
        'epoch': epoch+1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, f"latest_model.pth")
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Test Loss: {avg_test_loss:.4f}")
