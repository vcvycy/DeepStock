import sys
from util import enum_instance
if __name__ == "__main__":
    pass 
    path = "/Users/jianfeng/Documents/DeepLearningStock/training_data/data.daily.20240608_0124"
    for item in enum_instance(path):
        print(item)
        break