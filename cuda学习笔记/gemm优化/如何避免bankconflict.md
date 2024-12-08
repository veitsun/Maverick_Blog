# CUDA GPU编程应该如何避免bank conflict

## 1， 为什么要避免bank conflict

共享内存相比片外的全局内存有大的多的内存带宽

当对该内存数据读写操作频繁，则可以将全局内存中的数据加载到共享内存再进行读写操作，会大大提高程序的计算性能。然而共享内存和诸如if else的条件分支语句引起的warp divergence类似，当操作不当的时候，会导致程序性能的大大降低。

> warp divergence(束缚发散) 
> 
> 发生在同一个warp 中的线程由于不同的控制流路径而执行不同的指令，从而导致性能下降。（warp 是由32个并行执行的线程组成，在cuda中，线程以warp为单位由硬件调度，因此warp中的所有线程是一起执行同一条指令的，cuda的SIMT架构允许多个线程执行相同的指令）
> 
> warp divergence发生的条件
> 
> - 同一个线程因为遇到了条件分支语句，而执行不同的路径。GPU必须按照顺序执行这些不同的分支，而不是并行的执行整个warp的线程。这意味着同一个warp中的不同的线程不能同时执行，造成性能下降。
> 
> warp divergence的影响
> 
> - 性能下降：warp divergence会使GPU串行执行分支路径，而不是并行执行线程。GPU必须等待不同的分支执行完毕，导致某些线程在执行某个分支的时候被闲置，实际吞吐量大大降低。
> 
> - 延迟增加：由于warp divergence，执行不同分支的线程不能同时运行，必须一个接着一个执行，增加了整体的执行时间。

```csharp
if (threadIdx.x % 2 == 0) {
    // 偶数线程执行这段代码
    // 执行分支 A
} else {
    // 奇数线程执行这段代码
    // 执行分支 B
}/*在这个例子中，由于warp必须作为一个单元执行同一条指令，GPU会将先执行A分支，
然后执行B分支，尽管部分线程在这两个分支上可能是闲置的，这就是warp divergence*/
```

## 2， 什么是bank conflict 以及 bank conflict的产生

为了提高内存读写带宽，共享内存被分割成了32个等大小的内存块，即bank。因为一个warp有32个线程，相当于一个线程对应一个内存bank。这个分割方法通常是按照每4个字节一个bank，计算能力为3.x的GPU也可以8个字节一个bank

![](https://i-blog.csdnimg.cn/blog_migrate/2ab06cb7b1cc2d1aeb4a9a7dd5978fa4.jpeg)

理想情况下就是不同的线程访问不同的bank，可能是规则的访问，也可能是不规则的访问。这种同一时刻每个bank只被最多一个线程访问的情况下不会产生 。

特殊情况，如果有多个线程同时访问同一个bank的同一个地址的时候不会产生bank conflict，即broadcast。但当多个线程访问同一个bank的不同地址的时候，bank conflict就产生了。例如线程0访问地址0，而线程1访问地址32，由于它们在同一个bank，就会导致bank冲突。

bank 冲突产生之后，同一个内存读写将被串行化，而写入同一个地址时将只有其中一个线程能够成功写入（要使得每个线程都能够成功写入，需要使用原子操作atomic Instructions）。

## 3，如何避免产生bank conflict

bank conflict主要出现在global memory与shared memory的数据交换，以及设备函数对shared memory的操作中。

global memory与shared memory的数据交换中，最好是每次32个线程读写连续的32个word，这样既可以不产生bank conflict，又满足global memory的coalesced Access。如shared merory为64个字

> memory coalescing就是内存合并，通常用于高效地在全局内存，共享内存，寄存器之间传输数据
> 
> cuda kernel性能地一个重要因素是访问全局内存中数据的速度，有限的带宽很可能成为瓶颈，毕竟cuda kernel一下子要处理很多数据，数据从哪里来，得先有数据才能算了。所以如何高效传输数据当然就比较重要了。
> 
> memory coalescing 通常与tile技术结合使用，从而高效的利用带宽8
> 
> 内存合并和拼车安排非常相似。我们可以将数据视为通勤者，将DRAM访问请求视为车辆。当DRAM请求的速率超过DRAM系统提供的访问带宽时，交通拥堵情况加剧，算术单元变得空闲。如果多个线程从同一DRAM位置访问数据，它们可以形成一个“拼车团队”并将它们的访问合并成一个DRAM请求。然而，这些线程具有类似的执行时间表，以便它们的数据访问可以合并为一个DRAM请求。然而，这要求线程具有类似的执行时间表，以便它们的数据访问可以合并为一个。同一个warp中的线程是完美的候选者，因为它们都同时执行加载指令，这是由SIMD执行的特性。

设备函数对shared memory的操作中有很多注意点，需要仔细分析自己的程序，特别是不规则访问时。当每个线程只访问自己专属的数据时候，当每个线程只有1个word 的共享内存时则不会出现冲突。但当每个线程保有一个向量或者矩阵的时候，则需要仔细分析。

例如一个线程块有32个线程，每个线程有一个长度为6的数组（向量），则这个数组可以有三种申明方式：

方式1：一维数组方式：__shared__ int vector1[32*6];

方式2： 二维数组方式：__shared__ int Vector1[32][6];

方式3：二维数组方式：__shared__ int Vector1[6][32];

## 4， 共享内存的地址映射方式

要解决bank冲突，首先我们要了解一下共享内存的地址映射方式。

连续的32bits字被分配到连续的32个bank中。

<img src="https://segmentfault.com/img/bVFMd3" title="" alt="bank-layout" data-align="center">

图： 数字为bank编号

## 5， 典型的bank访问方式

<img src="https://segmentfault.com/img/bVFLSX" title="" alt="bank-access1" data-align="center">

这种访问方式是线性访问方式，访问步长为1，每个warp中的线程id与每个bank的id一一对应，因此不会产生bank冲突。

<img src="https://segmentfault.com/img/bVFLSZ" title="" alt="bank-access2" data-align="center">

这种访问方式是交叉访问方式，每个线程并没有与bank一一对应，但每个线程都会对应一个唯一的bank，所以也不会产生bank冲突。

<img src="https://segmentfault.com/img/bVFLS0" title="" alt="bank-access3" data-align="center">

这种访问方式虽然是线性访问bank，但这种访问方式与第一种的区别在于访问的补偿（stride）变为2，这就造成了线程0与线程28都访问到了bank0 ， 线程1与线程29都访问到了bank2，，，于是造成了2路的bank冲突。

<img src="https://segmentfault.com/img/bVFLS1" title="" alt="bank-access4" data-align="center">

这种情况造成了8路的bank冲突

下面有两种特殊情况：

<img src="https://segmentfault.com/img/bVFLS2" title="" alt="bank-access5" data-align="center">

这种情况是所有线程都访问同一个bank，产生了32路的bank冲突，但是由于广播（broadcast）机制，这种情况下并不会产生bank冲突。

> broadcast： 当一个warp中的所有线程访问一个bank中的同一个字word地址时，就会向所有的线程广播这个字word，这种情况并不会发生bank冲突。

<img src="https://segmentfault.com/img/bVFLS3" title="" alt="bank_access6" data-align="center">

这种情况是所谓的多播机制--当一个warp中的几个线程访问同一个bank中的相同字地址时，会将改字广播给这些线程。

这种多播机制只适用于计算能力2.0及以上的设备

## 6， 数据类型与bank冲突

当每一个线程访问一个32bits大小的数据类型的数据（如int， float）时，不会发生bank冲突。

```
extern __shared__ int shrd[];
foo = shrd[baseIndex + threadIdx.x]
```

但是如果每个线程访问一个字节(8-bits)的数据时，会不会发生bank冲突呢？很明显这种情况会发生bank冲突的，因为四个线程访问了同一个bank，造成了四路bank冲突。同理，如果是short类型（16-bits）也会发生bank冲突，会产生两路的bank冲突，下面是这种情况的两个例子：

```
extern __shared__ char shrd[];
foo = shrd[baseIndex + threadIdx.x];
```

<img src="https://segmentfault.com/img/bVFLTw" title="" alt="bank_access7" data-align="center">

```
extern __shared__ short shrd[];
foo = shrd[baseIndex + threadIdx.x];
```

<img src="https://segmentfault.com/img/bVFLS4" title="" alt="bank_access8" data-align="center">

## 7， 访问步长与bank冲突

我们通常这样来访问数组：每个线程根据线程编号tid与s的乘积来访问数组的32-bits字(word)：

```
extern __shared__ float shared[];
float data = shared[baseIndex + s * tid];
```

如果按照上面的方式，那么当s*n是bank的数量(即32)的整数倍时或者说n是32/d的整数倍(d是32和s的最大公约数)时，线程tid和线程tid+n会访问相同的bank。我们不难知道如果tid与tid+n位于同一个warp时，就会发生bank冲突，相反则不会。

仔细思考你会发现，只有warp的大小(即32)小于等于32/d时，才不会有bank冲突，而只有当d等于1时才能满足这个条件。要想让32和s的最大公约数d为1，s必须为**奇数**。于是，这里有一个显而易见的结论：当访问步长s为奇数时，就不会发生bank冲突。

## 8， bank冲突的例子

> 规约算法，是一种通过逐步合并数据来简化计算的算法。它通常用于处理大量数据，比如在并行计算和分布式系统中。规约操作的目标是将多个输入数据合并成一个输出，常见的规约操作包括求和，求最大值，求最小值等。
> 
> 在并行计算中，规约算法通常分为两个阶段：局部规约和全局规约。局部规约在每个处理单元中进行，生成部分结果；然后全局规约将这些结果合并，得到最终结果。这种算法在性能优化方面非常重要，因为它可以有效利用计算资源，减少数据传输的开销。
> 
> 比如现在有一个大数组，想要计算其所有元素的总和。我们可以将数组分成多个部分，让每个处理但与那（如线程或者是进程）计算自己部分的和。然后，再将这些部分和进行合并，得到最终的总和。   

**为什么不同的warp线程之间不会产生bank冲突**

在cuda中，warp是一个包含32个线程的基本调度单位。不同的warp线程之间不会产生bank冲突。

- 不同warp之间的独立性：  
  关键点在于，不同的warp是独立调度和执行的。它们不一定同时访问内存，即使同时访问，也是在不同的时钟周期进行的。

- 硬件调度：  
  GPU的硬件调度器会管理不同warp的执行。当一个warp因为内存访问而停止时，调度器可以切换到另一个准备好的warp。

- 时间分割：  
  即使多个warp同时需要访问相同的bank，硬件也会在时间上分割这些访问，使它们串行化，从而避免冲突。

- 并行性的保持：  
  由于有大量的warp可供调度，即使某些warp因为内存访问而暂停，整体的并行性和性能仍然可以保持。

## 避免bank冲突主要有以下的几种方法：

- 填充Padding

通过在数组中添加额外的元素来改变数据布局，使得连续的线程访问不同的bank。

```
// 假设有32个bank
__shared__ float sharedData[32][33];  // 而不是 [32][32]
```

- 交错访问（Interleaved Accessing）：

重新组织数据访问模式，使得相邻线程访问不同的bank

```
int tid = threadIdx.x;
int stride = blockDim.x;
for (int i = tid; i < N; i += stride) {
    // 使用 i 而不是 tid 来访问数据
}
```

- 线性寻址（Linear Addressing）：

使用一维数组而不是多维数组，可以简化访问模式并减少冲突。

```
__shared__ float sharedData[1024];
int index = threadIdx.y * blockDim.x + threadIdx.x;
sharedData[index] = someValue;
```

- 冲突感知的数据布局（Conflict-Aware Data Layout）:

设计数据结构时考虑到bank的数量和访问模式。

- 使用warp shuffles

在可能的情况下，使用warp shuffle操作代替共享内存访问。

```
int value = __shfl_sync(0xffffffff, sourceValue, sourceThread);
```

- 使用texture内存:  
  对于只读数据，使用texture内存可以避免bank冲突。

- 使用常量内存:  
  对于小型的只读数据，使用常量内存也可以避免bank冲突。

- 局部内存优化:  
  尽可能减少局部内存的使用，因为过多的局部内存使用可能导致寄存器溢出到全局内存。

- 动态共享内存:  
  使用动态共享内存可以在运行时调整共享内存的大小，有助于优化数据布局。
  
  cuda
  
  复制
  
  ```
  extern __shared__ float dynamicSharedMem[];
  ```

- 批处理（Batching）:  
  将多个小操作合并成一个大操作，减少总体的内存访问次数。

- 异步内存访问:  
  使用异步内存操作（如 `__ldg()`）可以帮助隐藏内存延迟。

- 合并访问（Coalesced Access）:  
  确保全局内存访问是合并的，这虽然不直接关联bank冲突，但可以显著提高内存访问效率。

> bank 是内存的物理结构，warp和block是cuda编程模型的概念
> 
> bank是内存系统中可以独立访问的单元，通常在硬件层面实现
> 
> warp是GPU执行的基本单位，通常包含32个线程
> 
> block是线程的逻辑分组，包含多个warp

> 注意：
> 
> 在cuda编程模型中，计算按照三级层次结构排序。每次调用cuda内核都会创建一个新的网格，该网格由许多个块组成。每个块最多由1024个单独的线程组成。
> 
> 位于同一块中的线程可以访问同一共享内存区域
> 
> 对于程序的性能，将同一线程块中的所有线程同等看待并不是一个好的主意。
