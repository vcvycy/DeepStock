# 模型训练
from common.utils import *
from common.stock_pb2 import *
import json
from google.protobuf.json_format import MessageToDict 
import sys
path = "/Users/jianfeng/Documents/DeepLearningStock/training_data/data.daily.20240608_0124" 
print(path)

# # 提取 fid 和 label
# features = []
# labels = []

# for instance in enum_instance(path, max_ins = 1000):
#     for feature_column in instance['feature']:
#         fids = feature_column.get('fids', [])
#         feature_dict = {fid: 1 for fid in fids}  # 使用 One-Hot 编码
#         features.append(feature_dict)
#     labels.append(instance['label'])

# # 将 features 和 labels 转换为适合模型训练的格式
# # 使用 DictVectorizer 来处理稀疏特征
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.model_selection import train_test_split

# vec = DictVectorizer()

# X = vec.fit_transform(features)
# y = [label['your_label_key'] for label in labels]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)