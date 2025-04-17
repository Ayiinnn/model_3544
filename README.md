# 共享代码和数据

### 时间窗口化的数据集链接
### ---样本集[3025,1000,20] (20=13+1+6)目标集 [3025,3] (T+1~T+3的price)--步长=4---484MB---
https://drive.google.com/file/d/1NhH97OZJ0LD-z9K5yna4Xhpe7Ydmgob4/view?usp=drive_link

----

## 更新

### ---新样本集[2890,1500,24] (24=16+1+6+1)目标集 [2890,6] ([6，24，48小时后的值]+[6，24，48小时后是否涨（涨为1不涨为0）])
（时间窗口处理后，6小时后指最后一个时间步的6小时后）
new_sample.npy和new_targets.npy
### ---新样本序列[13056,30] (30=24+6 , 即新样本集时间窗口化之前得到的序列)
final.csv
#### 以上在
https://drive.google.com/drive/folders/1UuBPlaieS1o-0OOANmJcy4BeMS3x4PbP?usp=sharing

