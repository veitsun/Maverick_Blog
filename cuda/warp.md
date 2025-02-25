# warp level

### _shfl_sync

是 CUDA 中一个非常重要的函数，用于在同一个 Warp 内部线程之间进行数据交换。

这个函数允许线程在同一个warp内相互共享数据，直接交换寄存器中的数据，而不需要通过全局内存或者共享内存。

```cpp
int __shfl_sync(unsigned mask, int var, int srcLane, cudaShuffleMode mode = cudaShuffleModeDefault);
```

mask: 用于指定参与通信的线程掩码，表示哪些线程会参与到数据交换中。 mask 是 32 为的整数， 每一位代表一个线程的参与状态。 0xFFFFFFFF 表示所有的 线程都参与， 0x00000001 表示只有第 0 个线程参与

所有参与数据交换的线程都会接受到这个数据

var：要传输的数据。通常是一个寄存器变量， var 在调用 __shuf_sync 时 会被传输给其他的线程

srclane : 是数据来源的线程 id 。 表示希望从哪个线程获取数据。 可以是 0 到 31 之间的值。

mode: 这个是一个 cudashufflemode 类型的美剧，用于指定数据传输的方式。它决定了是否传输寄存器的值，以及哪些线程之间可以交换数据。默认为 cudaShuffleModeDefault。 

线程之间的数据传输时基于 shuffle 的概念， 即通过低延迟的硬件实现数据传递， 而不是通过共享内存 或者 是全局内存

### `__shfl_down_sync` 与 `__shfl_sync` 的区别

- **`__shfl_sync`**：在同一个 Warp 内的线程之间进行全互通的数据交换，任何线程可以从任何其他线程获取数据，灵活性更强。
- **`__shfl_down_sync`**：只允许当前线程的数据向下（即 ID 更小的线程）传递，适用于逐步传递数据的场景（例如归约、分治算法等）。
