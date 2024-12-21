dasp_all 函数的参数说明

/**

 * @brief dasp 的整体操作

 *

 * @param[out]  filename        matrix.mtx 路径

 * @param[out]  csrValA         存储矩阵A的非零元素值

 * @param[out]  csrRowPtrA      存储矩阵A的行指针

 * @param[out]  csrColIdxA      存储矩阵A的列索引

 * @param[out]  X_val           输入向量

 * @param[out]  Y_val           输出向量

 * @param[out]  order_rid       0

 * @param[in]   rowA            矩阵A的行数

 * @param[in]   colA            矩阵A的列数

 * @param[in]   nnzA            存储矩阵的非零元素个数

 * @param[in]   NUM             4

 * @param[in]   threshold       0.75

 * @param[in]   block_longest   256

 * @param[out]  dasp_pre        DASP 预处理的时间 0

 * @param[out]  dasp_time       DASP 运行总时间 0

 * @param[out]  dasp_gflops     DASP 带宽 0

 * @param[out]  dasp_bandwidth  bandwith 0

 *

 * @return 0

 */

常量

block_longest 256

warpNum_short 4

groupNum 1

局部变量

struct timeval t1, t2, pre_t1, pre_t2; // 测时间的

short_row_1 长度为 1 为多少行

short_row 2 长度为 2 为多少行

short_row_3 长度为 3 为多少行

short_row_4 长度为 4 为多少行

row_zero 长度为 0 为多少行

row_long 长行为多少行 （长度大于等于 256 ）

row_block 中等行为多少行

rowloop 执行中等行计算时，表示要由一个warp计算的行块数

short_rid_1  长度为 1 的每行 行id （后开辟的数组）

short_rid_2 长度为 2 的每行 行id （后开辟的数组）

short_rid_3 长度为 3 的每行 行id （后开辟的数组）

short_rid_4 // 长度为 4 的每行 行id （后开辟的数组）

zero_rid 长度为 0 的每行 行id （后开辟的数组）

long_rid 长行的每行 行id （后开辟的数组）

long_rpt 长行的对应每行的行长 （后开辟的数组）

ridA 中等行的每行 行id （后开辟的数组）

rptA 中等行的对应每行的行长 （后开辟的数组）

nnz_short 短行的非零元的元素个数

common_13 更新前是短行1总行数和短行3总行数的最小值，（在按8行为一行块进行划分，总行块数大于等于

                        16时

                        更新，否则如果没有大于等于 16行块的话，common_13就是0）跟新后是BLockSize * 4 的整数

                        倍，

                        短行1的总行数，和短行3的总行数都要小的值，这种操作通常用于优化计算和确保数据对齐

                        此后，短行 1的剩余总行数也更新了，短行3的剩余总行数也更新了

                        

short_block13  短行1&3拼接一起，8个这样的行算一个行块，一共有多少个这样的行块

half_short_row_2    2个短行 2 一起计算，一共多少个行块

short_block22     2&2拼接一起算作一个行，然后按 按8 行算一个行块，一共有这样的 行块

short_row_34     元素个数是 3 和 4 的行是一起计算的，一共有多少个行数 为3和4的行

short_block34   按 8 行为一行块进行分块，一共有多少个3 or 4 组合为一行块的的行块

block13_per_threadblock      每个线程块处理block13 的数量 16 ，一个SM有4个warp，一个

                                                    warp处理

                                                    2个 行块

block22_per_threadblock

block34_per_threadblock

threadblock13      一个线程块处理 16个block13  一共需要需要多少个线程块数

threadblock22

threadblock34

fill0_nnz_short13   需要处理的短行 1&3 总元素个数，包括填充0

fill0_nnz_short34   需要处理的短行 3or4 总元素个数，包括填充0

fill0_nnz_short22  需要处理的短行 2&2 总元素个数，包括填充0

fill0_nnz_short    需要处理的短行元素总个数

short_val  需要处理的短行元素 ，后开辟出来的 fill0_nnz_short大小的（数组）

short_cid 需要处理的短行元素对应的列id， 后开辟出来的 fill0_nnz_short大小的（数组）
