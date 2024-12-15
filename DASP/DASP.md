# DASP:特殊稠密矩阵乘累加单元加速一般稀疏矩阵向量乘法

## DASP

### Overview

### Data Structure

根据每行非零元的个数，把每行分组成三种情况

> long rows : Row_len > MAX_LEN
> 
> medium rows : 4 < Row_len <= MAX_LEN
> 
> short rows : Row_len <= 4

MAX_LEN 是一个可调节的参数，代表着 medium rows 的最小可能长度

在实际计算中，mma 指令支持的最小 layout 是 m8n8k4 ，所以在实际的数据结构中 block 是 8 * 4 的大小（在论文中使用的例子是 m2n2k4 所以非零元素的 block 大小是 2 * 4）

- long rows part

每行的非零元将被分成多个组每组非零元的数量 2 * MMA_M * MMA_K 。如果元素中非零元的个数不足以成为一个组，那么需要用零元素来填充。

因此 long rows 部分可以用 三个数组 来存储其相关数据

    (1) longVal : 存储 值，包括填充的零元素。数组大小是 nnz_long_new

    (2) longCid : 存储相应元素的列索引，数组大小是 nnz_long_new

    (3) groupPtr : 偏移量

- medium rows part

所有的行按照稳定的降序排列

将 MMA_M 宽度的 rows 看作一个行块（row-block），根据阈值， row-block 被分成好多个 block （大小为 MMA_M * MMA_K ） 。阈值是一个自定义的参数，在 DASP 中设置为 0.75 。 当 MMA_M * MMA_K 大小的space中非零元素个数超过 MMA_M * MMA_K * 阈值 时，我们将其识别为一个块，空的位置被零元素填充，我们称这些块为 medium rows 的规则部分；否则不被认为是块的 非零元值 是属于 medium rows 的不规则部分（irregular part）。我们将规则部分和不规则部分的元素分开存储，因此在图中 medium rows 部分使用了 六个数组 来存储其相关数据。

    (1) regVal : 存储 regular part 的元素值，包括原始矩阵中的非零元素 和 填充的零元素。 这个矩阵采用了块内 行主序布局，大小为 nnz_reg_new

    (2) regCid : 将对应元素的列索引存储到数组中，其大小是 nnz_reg_new

    (3) rowblockPtr : 在每一个 row-block 中存储 属于 regular part 的第一个元素的偏移量

    (4) irregVal : 存储 irregular part 中非零元的值，其大小是 nnz_irreg

    (5) irregCid : 将 irregular part 相应的非零元的列索引存储在数组中， 其大小是 nnz_irreg  

    (6) irregPtr : 存储 irregular part 的每行第一个非零的内存偏移量

- short rows part

采用 piecing to blocks 的策略来提高 MMA 单元的利用率

用两个数组来存储 short part 部分的数据

    (1) shortVal : 存储 short part 部分所有元素的值 ，数组大小是 nnz_short

    (2) shortCid : 将相应元素的列索引存储到数组中，该数组的大小是 nnz_short_new 。因为在short part 中的四种categories 都是固定的，所以不需要冗余的数组来存储元素的内存偏移量。

### ALgorithm Description

DASP 对不同类型的行，给出了不同的计算策略。在接下来的算法中我们采用了 bypass cache 方法提高x的缓存命中率。

#### long rows

每行将被分成若干组 2 * MMA_M * MMA_K 单元，Algorithm 2 给出了 long rows 部分 计算的伪代码

在每次调用 mma 指令之前， 数据块(MMA_M * MMA_K) 被暂时存放在每个线程的 fragA 和 fragX 的寄存器

然后 调用 mma 指令让一个 warp 中的32个线程一起计算 m8n8k4 大小的 GEMM 。 在完成两次 MMA 计算之后，产生8个分散在32个线程寄存器（fragY） 中的结果 

下一个步骤是，调用cuda shuffle 指令对这 8 个值进行求和， 并将结果存储在第一个线程的寄存器中

然后将结果写入预先分配好的全局数组warpVal中

在所有的warp 都计算完结果，并且结果已经写入到 warpVal 数组中，通过对一行产生的warpVal 中的所有值进行累加操作，得到最终的 y 值。

设置一个线程块中有 4 个warp ，每个 warp 计算 long rows part中两个数据块，所以一个线程块将在一行中计算 256 个元素

因此 MAX_LEN 的值可以说是正好是一个线程块的工作负载

#### medium rows

引进了新的变量（一个warp 处理多少个行块 数量）

当 row_medium 少于 59990时， loop_num 的值是 1 ， 当 row_medium 大于 59990 小于 400000 时， loop_num 的 值是2， 当 row_medium 大于等于400000 时， loop_num 的值是 4 

同样， 在计算前 将相应块的数据加载到本地寄存器中， 然后调用 mma 指令执行计算。

一般来说， 一个行块会由多个块组成，所以循环计算的次数由行块的长度控制，一个行块的多次计算的乘积累加到寄存器 fragY 中

在通过 CUDA_Shuffle 指令将寄存器 fragy 中的值提取到寄存器res对应的位置后，medium part 的 regular part 几倍呢上计算完成了， 对于 irregular part 部分的元素， 每个线程并行的负责一行， 并将结果累加到相应的寄存器res中， 最后一个一起写回到 数组valy 中。

图 7 展示的是当 loop_num = 1 的情况，图中的行块是由三个块组成，需要执行三个 MMA 计算。

#### short rows

short rows 部分被分成了 四种情况，它们也是分别对应四种算法。

算法4 通过 1&3 拼接给出了 short rows 计算过程的伪代码。 为了优化 计算资源 和 mma 单元的利用率，一个warp 将调用 四次mma指令来完成两个 blocks 的计算，这样， 一个 warp 可以精确的计算出 y 的 32 个连续值。

在调用 mma 指令进行计算之前，相关的block 数据必须存储在寄存器 fragA 和 fragX 中。 然而， 并不是两个完整的数据加载 和 mma计算， 每个 block 只需要加载一次矩阵 A 中value 和 两次 x 。

第一次是 加载关于 每个 block 第一列的 x 的值 （ fragY 中的空位置 设置为0 ）

第二次是 加载关于 每个 block 最后三列 的x的值。mma 计算的结果 分布在寄存器 fragY 的 32 个线程之间。 使用 cuda shuffle 指令， 将这些结果按照顺序存储在寄存器 res 中， 然后写回到矩阵 valY 中。

图 8 表示的是 1&3 拼接的 在一个warp 中的计算。

同样的， 2&2 拼接 采用了几乎相同的计算策略。当加载一个 block 中的值到寄存器 fragX 中的时候， 前两列会先被加载，紧接着是后两列。piecing together 的方法高效的减少了 数据传输的开销， 和 提高了mma 单元的利用率

对于 row _ len = 4 (包括由零填充构成的行)的行，一个  warp 也会调用4次mma指令，完成4个块的计算。每次计算前，先将一整块数据加载到寄存器f rag A和f rag X中，然后调用mma指令执行计算。每次MMA计算得到的8个有意义的结果通过CUDA Shuffle指令存储在连续8个线程的寄存器中，在所有4次计算完成之后立即将结果写回数组valY。

对于只有一个非零的行，我们使用基本计算单元进行计算。算法5给出了这种行的计算过程的伪代码。对于每一行，分配一个线程计算一行，并将结果写回该行对应的valY。





### Experimental results
