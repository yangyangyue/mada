import torch  
import torch.nn as nn  

WINDOW_SIZE = 1024
  
class S2pNet(nn.Module):  
    def __init__(self, input_window_length):  
        super().__init__()  
        # 注意：PyTorch的Conv2d需要[channels, height, width]的输入，所以我们用1个通道来表示原始数据  
        self.conv1 = nn.Conv1d(1, 30, kernel_size=11, stride=1, padding=5)  
        self.conv2 = nn.Conv1d(30, 30, kernel_size=9, stride=1, padding=4)  
        self.conv3 = nn.Conv1d(30, 40, kernel_size=7, stride=1, padding=3)  
        self.conv4 = nn.Conv1d(40, 50, kernel_size=5, stride=1, padding=2)  
        self.conv5 = nn.Conv1d(50, 50, kernel_size=5, stride=1, padding=2)  
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(50 * WINDOW_SIZE, 1024)  # 这里的+4是考虑padding后的输出尺寸，可能需要根据实际情况调整  
        self.fc2 = nn.Linear(1024, 1)  
  
    def forward(self, x):  
        x = self.conv1(x[:, None, :]).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = self.conv5(x).relu()
        x = self.flatten(x)  
        x = self.fc1(x).relu()
        x = self.fc2(x)  
        return x  
