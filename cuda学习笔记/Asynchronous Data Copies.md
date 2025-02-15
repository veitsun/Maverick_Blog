## Asynchronous Data Copies

### 7.27.1 API

异步：操作不需要等待前一个操作完成即可开始

memcpy_async

cooperative_groups::memcpy_async

> 协作组（cooperative groups 是 cuda9 引入的一种编程模型，用于在 GPU 上实现更高效的线程协作和同步），使用 `cooperative_groups::thread_block::sync()` 可以在协作组内同步所有线程，确保所有线程都执行到这一点后再继续执行后续的操作。

### 7.27.2 复制和计算模式

without memcpy_async : 需要使用中间寄存器

with memcpy_async : 不需要使用中间寄存器

### 7.27.3. Without `memcpy_async`

> shared[local_idx] = global[global_idx]
> 
> 全局至共享内存副本将从全局存储器扩展到寄存器，然后将其写入寄存器中的共享内存。
