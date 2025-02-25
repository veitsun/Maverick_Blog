# 如何优化cuda矩阵乘内核以获得类似cublas的性能

## kernel 1 naive Implementation

在 CUDA 编程模型中，计算按三级层次结构排序。 每次调用 CUDA 内核都会创建一个新网格，该网格由多个块组成。 每个块由最多 1024 个单独的线程组成。

类似地, 网格中的线程块数可以使用 gridDim 变量进行配置. 当我们从主机6启动一个新内核时, 它会创建一个单独的网格, 包含指定的线程块和线程7. 重要的是要记住, 我们刚才讨论的线程层次结构主要关注程序的正确性. 对于程序性能, 正如我们稍后将看到的, 将同一线程块中的所有线程等同看待并不是一个好主意.

对于我们的第一个内核，我们将使用网格，线程块和线程的层次结构，为每个线程分配结果矩阵C中的唯一一个条目，然后，该线程将计算矩阵A对应行和矩阵B对应列的点积，并将结果写入到C。由于矩阵C的每个位置只由一个线程写入，因此我们不需要进行同步操作

CUDA 代码是从单线程的角度编写的. 在内核代码中, 我们访问 `blockIdx` 和 `threadIdx` 内置变量. 这些将包含基于访问它们的线程返回的不同值.

GPU宣传具有30TFLOPs/s的FP32计算吞吐量和768GB/s 的全局内存带宽。如果我们到达了这些数据指标，那么计算需要4.5ms，内存传输需要0.34ms。因此，在我们的草稿计算中，计算时间约为内存访问时间的10倍。这意味着只要我们需要传输的内存量小于绝对最小值278MB的10倍，我们的最终优化的内核将受到计算限制。（这是因为计算操作的时间远大于内存传输的时间，表明计算密集型compute-bound 的特性）（这段话强调了在进行大规模计算时候，合理利用缓存和优化内存传输的重要性，以及如何平衡计算和内存传输以实现最佳性能）

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2073ce19220266355821a0062dd682ed.png)

ABC是以行主序存储的，意味着y索引在内存中是连续的地址迭代的

## kernel 2 全局内存合并

为了便于执行，线程块的线程被分组为由32个线程组成的所谓的warp。然后一个warp被分配给warp调度器，该调度器是执行指令的物理核心。

每个多处理器有四个warp调度器，分组为warp是基于连续的threadId进行的。

当我们将blockDim设置成多维的，那么threadId的计算如下：

```
threadId = threadIdx.x + blockDim.x*(threadIdx.y + blockDim.y*threadIdx.z)
```

属于同一个warp的线程的顺序内存访问可以被组合并作为一个整体执行，这被称为内存合并，这是在优化内核的全局内存（GMEM）访问以达到峰值带宽时最重要的事情

## kernel 3 共享内存缓存分块(12)

在较大的全局内存旁边，GPU有一个小的多的内存区域，物理上位于芯片上，称为共享内存（SMEM），在物理上，每个SM有一个共享内存。从逻辑上讲，这个共享内存是在线程块之间划分的，这意味着线程可以通过共享内存块中的其他线程进行通信。

由于共享内存位于片上，因此它比全局内存具有更低的延迟和更高的带宽。

因此，对于下一个内核，我们将全局内存中加载A的一个分块和B的一个分块到共享内存中。然后，我们将在这两个分块上执行尽可能多的工作，每个线程仍被分配一个C条目。我们将沿着A的列和B的行移动分块，对C执行部分求和，直到计算出结果。

由于kernel的blockDim是32 * 32的，矩阵C的线程块分片大小均为32 * 32，所以线程块的每个线程恰好处理矩阵C的线程块分片对应的一个元素。从矩阵A和矩阵B来说，每次内层for循环时，每个线程同时对应A分片dotIdx行中的一个元素和B分片中dotIdx中的一个元素

工作负载按线程块粒度在SM上调度，每个SM将尽可能加载更多的线程块，只要它有足够的资源来容纳他们

- 共享内存: 8192B/Block + 1024B/Block 用于 CUDA 运行时使用 = 9216B/Block. (每 SM 102400B) / (每线程块 9216B per Block) = 11.11 ⇒ 上限 11 个线程块.

- 线程: 每线程块 1024 个线程, 每个 SM 最多 1536 个线程 ⇒ 上限 1 个线程块.

- 寄存器: 每线程 37 个寄存器 * 每 warp 32 线程 = 每 warp 1184 个寄存器. 寄存器分配粒度在 warp 级别上为 256个寄存器, 因此每个 warp 四舍五入到 1280 个寄存器. 我们有 (1024 线程 / 32) = 每个线程块 32 个 warp, 因此每个 warp 1280 个寄存器 * 每个线程块 32 warp = 每个线程块 40960 个寄存器. 每 SM 最多 65536 个寄存器 ⇒ 上限 1 个线程块.32

## kernel 4 用于计算每个线程的多个结果的一维块分片(36)

添加一个新的内部循环，用于计算每个线程的多个C条目。我们现在使用共享内存缓存1024个浮点数大小，每个线程块总共4KB。

<img src="https://i-blog.csdnimg.cn/blog_migrate/b97c0ef4b2f60f1a0c798a16d2255b26.png" title="" alt="在这里插入图片描述" width="504">

我们当前的内核仍然存在与kernel 3 相同的内存停滞（stalling-for-memory）问题，只是程度较低。因此，我们将再次应用相同的优化：让每个线程计算更多的结果。使我们的内核运行得更快的主要原因是增强了计算强度

总之，我们所有的内核执行相同数量的FLOP，但我们可以通过每个线程计算更多的结果来减少GMEM访问的数量。只要我们仍然是内存限制的（memory bound），我们就可以继续优化计算强度

## kernel 5 通过二维块分片增加计算强度(68)

每个线程计算C的8 * 8个元素的网格，内核的第一阶段是让所有线程一起工作来填充SMEM缓存。我们将每个线程加载多个元素。 

<img title="" src="https://i-blog.csdnimg.cn/blog_migrate/0b788bf50d0e928b0098f6d638e2ef9e.png" alt="在这里插入图片描述" data-align="center" width="422">

kernel 5 的二维分块和kernel4的一维分块的整体思路是一致的，核心在于将一维分片改成了二维分片，换句话说， 从“仅将As的线程块分片进一步划分为线程分片”到“As和Bs的线程分片均进一步划分为线程分片”

由于线程现在处理TM * TN个元素，因此As分片和Bs分片需要多次加载才能将其元素加载到共享内存。

由于 `threadCol = threadIdx.x % (BN / TN)`, `threadRow = threadIdx.x / (BN / TN)`.即相邻线程对应Bs分片的不同列线程分片，As分片的同一行线程分块，与先前kernel的思路是一致的。

`for dotIdx` 循环仍然是线程块分片沿着K维度逐一计算，即每次As处理一列，Bs处理一行。

at each timestep, load the 4 relevant As&Bs entries into regM and regN registers, and accumulate outer product into threadResults.

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6596b2b4ca39212000edfce664eae694.png)

benefit：we only issue 16 SMEM loads in total

性能慢慢地达到了可以接受的水平，但是，由于内存流水线（memory pipeline）拥塞而导致的warp停滞仍然过于频繁。对于kernel 6， 我们将采取两种措施来尝试来尝试改进这一点：转置As以实现共享内存加载的自动向量化，以及向编译器承诺在全局内存访问时的内存对齐。

## kernel 6 向量化SMEM和GMEM访问

第一个优化--> 转置As，从As加载时使用向量化共享内存

As加载到寄存器过去是用的32b的LDS加载负载，而现在是使用128b的LDS.128加载，就像之前对Bs的加载一样

使用向量数据类型（float4）对GMEM的所有加载和存储进行向量化

```
float4 tmp =
    reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
// transpose A during the GMEM to SMEM transfer
As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
    reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
__syncthreads();
```
