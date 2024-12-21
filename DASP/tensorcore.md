# 什么是tensor core

tensor core 是 nvidia gpu 中的一种专用硬件， 用于加速混合精度矩阵乘法和累加操作。它能够处理小矩阵，如4*4 或者 8 * 8 的运算

特点：

- 高吞吐量： 每个 tensor core 每周期可以完成 64 次浮点运算

- 混合精度： 支持 FP16、 BF16、TF32、FP32 等数据类型

# 使用 tensorcore的前提条件

- 硬件支持

GPU 必须是 volta 、turing、Ampere 、hopper 或者是更高的架构 

- 数据类型

通常使用混合精度： FP16 或者 BF16 输入， FP32 输出

- 内存对齐

输入矩阵的维度应满足 8 的倍数，因为 tensorcore 操作对内存对齐有严格的要求
