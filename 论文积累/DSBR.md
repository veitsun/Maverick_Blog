# 一个在结构化网格上的稀疏三角求解器向量化的高效存储格式





DBSR ： 

为了让 SPTRSV 充分利用 SIMD 指令



促进了连续的内存访问和向量化计算，同时也优化了内存使用率（optimizing memory usage）



我们通过使用多重网格算法和零填充不完全LU预条件来应用 DBSR 评估 DBSR
