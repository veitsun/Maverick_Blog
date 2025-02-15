## Asynchronous Barrier

without the arrive/wait barrier：

 __syncthreads() : 在一个block 中同步所有的线程

group.sync() : 使用协作组的时候，在一个 block 中同步所有线程

### 7.26.1 简单的同步模式

__syncthreads() : 在一个block 中同步所有的线程

group.sync() : 使用协作组的时候，在一个 block 中同步所有线程

### 7.26.2. Temporal Splitting and Five Stages of Synchronization

cuda::barrier： cuda中的一种同步原语，用于协调线程块（block）内的线程。它允许线程在某个同步点等待其他线程完成特定操作，然后再继续执行。

传统的同步点： __syncthreads() 是一个阻塞操作，所有线程必须到达该点并等待其他线程完成。

cuda::barrier :  同步点被拆分为两个部分

- **`arrive()`**：线程通过调用 `bar.arrive()` 来通知屏障（barrier）自己已经到达某个点。这个操作不会阻塞线程，线程可以继续执行其他不依赖于同步点之后的操作。

- **`wait()`**：线程通过调用 `bar.wait(std::move(token))` 来等待其他线程完成 `arrive()`。只有当所有参与线程都调用了 `arrive()` 指定的次数（由 `init()` 中的 `expected arrival count` 参数决定），线程才会从 `wait()` 中解除阻塞。

### 7.26.3. Bootstrap Initialization, Expected Arrival Count, and Participation
