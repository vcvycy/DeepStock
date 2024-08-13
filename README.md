# 经验
# LRModel训练20个epoch后，avg_label能对得上

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

## 关于Loss
  对于稠密fid，假设一个batch这个fid出现100次, 那么使用MSE(reduction="mean")更好，否则梯度累计起来，直接爆炸
  对于稀疏fid，假设一个batch这个fid出现1次, 那么使用MSE(reduction="sum")更好，否则 grad = grac/batch_size，梯度直接
## 关于shuffle
    训练数据不shuffle，会导致出现问题：由于越靠后的梯度越重要，如果负样本集中在后面，就会导致被后面的负样本主导。


耗时： 
-----------------------------------------
| Func              TimeElapsed  ratio  |
|                                       |
| __next__          5.78秒       0.13%  |
| get_embedding     306.17秒     7.12%  |
| emit_summary      287.43秒     6.69%  |
| forward           596.79秒     13.89% |
| backward          727.58秒     16.93% |
| opt_step          0.28秒       0.01%  |
| update_embedding  368.90秒     8.58%  |
| validate          122.10秒     2.84%  |
| main              1882.43秒    43.80% |
| Total             4297.46秒    100%   |
-------------------------------------------