# 配置
## 在bashrc中调用我们的环境信息
```bash
echo "source $(pwd)/env.sh" >> ~/.bashrc
```
# DEBUG
查看sqlite3数据库:  SQLiteStudio

# 项目架构
* src/data: 从tushare读取数据，写到sqlite3数据库。
### 需求
1. 底部涨停的
###


# 模型训练

# 关于Loss
  对于稠密fid，假设一个batch这个fid出现100次, 那么使用MSE(reduction="mean")更好，否则梯度累计起来，直接爆炸
  对于稀疏fid，假设一个batch这个fid出现1次, 那么使用MSE(reduction="sum")更好，否则 grad = grac/batch_size，梯度直接没了