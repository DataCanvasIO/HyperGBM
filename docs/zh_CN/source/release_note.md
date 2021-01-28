## Release Note

本次发布新增以下texing：

特征工程	
  - 特征衍生
  - 特征降维

数据清洗
  - 特殊空值字符处理
  - 列类型校正
  - 常量列清理
  - 重复列清理
  - 删除标签列为空的样本
  - 非法值替换
  - id列清理
  - 共线性特征清理

数据集拆分
  - 对抗验证

建模算法
  - XGBoost
  - Catboost
  - LightGBM
  - HistGridientBoosting

模型训练
  - 自动任务类型推断
  - 命令行工具

评估方法
  - 交叉验证(Cross-Validation)
  - Train-Validation-Holdout验证

搜索算法
  - 蒙特卡洛树算法
  - 进化算法
  - 随机搜索算法

不平衡数据处理
  - 类平衡（Class Weight）
  - 降采样(Under -Samping)支持 
    - Near miss
    - Tomeks links 
    - Random
  - 过采样(Over-Samping)支持
    - SMOTE 
    - ADASYN 
    - Random

提前停止策略
  - n次搜索都不再提升，提前停止
  - 最大用时提前停止
  - 到达预期指标提前停止

其他高级特性
  - 二阶段搜索-伪标签
  - 二阶段搜索-特征选择
  - 概念漂移处理
  - 模型融合

其中hypergbm有3中运行模式，分别为：
- 单机模式
- 单机分布式
- 多机分布式

三种模式对与以上特性的执行情况如图：

todo 增加功能矩阵


