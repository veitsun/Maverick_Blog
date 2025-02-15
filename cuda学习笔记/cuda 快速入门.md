# cuda 中的一些概念

Grid 是 CUDA 中线程块的集合，Block 是 CUDA 中线程的集合，线程 Thread 是 CUDA 中执行的最小单元

。Grid 由多个 Block 组成， Block 由多个 Thread 组成。



Thread 是 CUDA 中执行的最小上下文单元，类似于 CPU 中 SIMD 中的一个计算通道。并且 Thread 是实际存在的硬件，而 Block 和 Grid 是逻辑概念， 是 CUDA 为了方便管理线程而引入的抽象
