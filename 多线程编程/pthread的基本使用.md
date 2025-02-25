原生的多线程库

- <thread> (std::thread 类) ==> C11

- <pthread.h> (p_thread 类)  ==> linux(Windows环境下没有pthread)

两种多线程库的区别

    std::thread 更加好用

    pthread功能更加强大

**线程的创建和管理**

每个线程都有一个在进程中唯一的线程标识符

用一个数据类型 pthread_t 表示，该数据类型在 Linux 中就是一个无符号长整型数据。

```
int pthread_create (pthread_t *thread,pthread_attr_t *attr,void *(*start_routine)(void *),void *arg);
```

若创建成功，返回0；若出错，则返回错误编号。

- thread 是线程标识符，但这个参数不是由用户指定的，而是由 pthread_create 函数在创建时将新的线程的标识符放到这个变量中。
- attr 指定线程的属性，可以用 NULL 表示默认属性。
- start_routine 指定线程开始运行的函数。
- arg 是 start_routine 所需要的参数，是一个无类型指针。

默认地，线程在被创建时要被赋予一定的属性，这个属性存放在数据类型 pthread_attr_t 中，它包含了线程的调度策略，堆栈的相关信息，join or detach 的状态等。

pthread_attr_init 和 pthread_attr_destroy 函数分别用来创建和销毁 pthread_attr_t

**结束线程**

当发生以下情形之一时，线程就会结束：

- 线程运行的函数return了，也就是线程的任务已经完成；
- 线程调用了 pthread_exit 函数；
- 其他线程调用 pthread_cancel 结束这个线程；
- 进程调用 exec() 或 exit()，结束了；
- main() 函数先结束了，而且 main() 自己没有调用 pthread_exit 来等所有线程完成任务。

当然，一个线程结束，并不意味着它的所有信息都已经消失，后面会看到僵尸线程的问题。

当然，一个线程结束，并不意味着它的所有信息都已经消失，后面会看到僵尸线程的问题。

下面介绍这两个函数。

```c
void pthread_exit (void *retval);
```

retval 是由用户指定的参数， pthread_exit 完成之后可以通过这个参数获得线程的退出状态。

```c
int pthread_cancel (pthread_t thread);
```

一个线程可以通过调用 pthread_cancel 函数来请求取消同一进程中的线程，这个线程由thread 参数指定。

如果操作成功则返回0，失败则返回对应的错误编号。

**对线程的阻塞**

阻塞是线程之间同步的一种方法。

```c
int pthread_join(pthread_t threadid, void **value_ptr);
```

pthread_join 函数会让调用它的线程等待threadid线程运行结束之后再运行

value_ptr 存放了其他线程的返回值。

一个可以被join的线程，仅仅可以被别的一个线程 join，如果同时有多个线程尝试 join 同一个线程时，最终结果是未知的。另外，线程不能 join 自己。

**互斥锁**

Mutex常常被用来保护那些可以被多个线程访问的共享资源，比如可以防止多个线程同时更新同一个数据时出现混乱

使用互斥锁的一般步骤是：

- 创建一个互斥锁，即声明一个pthread_mutex_t类型的数据，然后初始化，只有初始化之后才能使用；
- 多个线程尝试锁定这个互斥锁；
- 只有一个成功锁定互斥锁，成为互斥锁的拥有者，然后进行一些指令；
- 拥有者解锁互斥锁；
- 其他线程尝试锁定这个互斥锁，重复上面的过程；
- 最后互斥锁被显式地调用 pthread_mutex_destroy 来进行销毁。

> 有两种方式初始化一个互斥锁
> 
> - 利用已经定义的常量初始化
> 
> - 调用`pthread_mutex_init(mutex, attr)`进行初始化

当多个线程同时去锁定同一个互斥锁时，失败的那些线程，如果是用 pthread_mutex_lock 函数，那么会被阻塞，直到这个互斥锁被解锁，它们再继续竞争；如果是用 pthread_mutex_trylock 函数，那么失败者只会返回一个错误。

保护共享数据是程序员的责任。程序员要负责所有可以访问该数据的线程都使用mutex这种机制，否则，不使用mutex的线程还是有可能对数据造成破坏

**条件变量 Condition Variable**

互斥锁只有两种状态，这限制了它的用途。条件变量允许线程再阻塞的时候等待另一个线程发送的信号，当收到信号之后，阻塞的线程就会被唤醒并试图锁定与之相关的互斥锁。条件变量要和互斥锁结合使用。

**条件变量的声明和初始化**

通过声明pthread_cond_t 类型的数据，并且必须先初始化才能使用。

初始化方法也有两种：

- 利用内部定义的常量

- 利用函数pthread_cond_init(cond, attr)， 其中attr 由pthread_condattr_init() 和 pthread_condattr_destroy() 创建和销毁

可以用pthread_cond_destroy()销毁一个条件变量
