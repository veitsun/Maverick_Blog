# An Automatic Framework to Optimize Sparse Triangular Solve on GPUs

# AG-SpTRSV 在GPU上优化稀疏三角求解的自动框架

SpTRSV 稀疏三角求解 长期以来一直是科学计算领域的重要内核

由于SPTRSV的计算强度和内部数据依赖性低，因此很难在图形处理单元GPU 上实现和优化。

SpTRSV is hard to implement and optimize on graphics processing units

based on our experimental oberservations

GPU上现有的实现无法实现最佳性能 due to their suboptimal parallelism and code implementations plus lack of consideration of the irregular data distributuin 

moreover, their algorithm design lacks the adaptability to different input matrices

这可能涉及大量的手动算法重新涉及和参数调整以实现性能一致性。

提出 -> AG-SpTRSV 这是一个在GPU上优化 SpTRSV 的自动框架，它在各种矩阵上提供高性能，同时消除了手动设计的成本

AG-SpTRSV 将 SpTRSV 内核的优化过程抽象为一个方案，并在此基础上构建一个全面的优化空间。通过定义统一的代码模板并准备代码变体 AG-SpTRSV 可实现精细的动态并行和自适应代码优化，以处理各种任务。

通过计算图变换和多层次启发式调度，AG-SpTRSV 生成任务分区和映射方案，有效解决了数据分布不规则和内部数据依赖问题
