import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        
        # 定义一个名为weight的参数
        self.weight = nn.Parameter(torch.randn(5), requires_grad=True)
        
        # 定义一个名为bias的参数
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        return x * self.weight + self.bias

# 创建模型实例
model = SimpleModel()

# 打印模型的参数
for name, param in model.named_parameters():
    print(f"Parameter name: {name}, size: {param.size()}")