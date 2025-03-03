# ---

# 算子五：`nn.ConvTranspose3d` 3D 转置卷积算法

## 功能：

​在由多张输入平面组成的输入图像上应用三维转置卷积操作。​转置卷积操作对每个输入值与可学习的卷积核进行逐元素相乘，然后将所有输入特征平面的输出相加。对三维输入数据进行上采样，增加空间维度（深度、高度和宽度）。

## 输入参数：

- **输入通道数（in_channels）**：输入数据的通道数。
- **输出通道数（out_channels）**：输出数据的通道数。
- **卷积核大小（kernel_size）**：卷积核在每个维度上的大小(核的深度，核的高度，核的宽度)。
- **步长（stride）**：决定卷积核在输入数据上滑动的步幅，通常也用于控制输出的尺寸。
- **填充（padding）**：控制输入数据周围的零填充量（`dilation * (kernel_size - 1) - padding` 点数两侧的隐式零填充量），用于控制输出的尺寸。
- **输出填充（output_padding）**：在输出数据中补充的填充量，用于调整输出尺寸。
- **分组（groups）** ： 从输入通道到输出通道的阻塞连接数，默认值为1。控制输入输出的连接。in_channels 和 out_channels 都必须能被 groups 整除
  - groups = 1 所有输入都卷积到所有输出
  - groups = 2 该操作等效于并排有两个卷积层，每个卷积层看到一般的输入通道并产生一半的输出通道，并且随后将两者连接起来
  - groups = in_channels 每个输入通道都与自己的一组过滤器（大小为 out_channels/in_channels）进行卷积。
- **偏差（bias）** ：如果 `true` ,则向输出添加可学习的偏差，默认值为 True
- **扩张（dialation）** ：控制内核元素之间的间距，默认值为 1

参数 `kernel_size` 、 `stride` 、 `padding` 、 `output_padding` 可以是：

（1）单个int情况下：深度，高度，宽度的尺寸使用相同的值

（2）三个整数的tuple情况下：第一个 `int` 用于深度维度，第二个 `int` 用于高度维度，第三个 `int` 用于宽度维度

## 输出参数：

- 批次大小（N）：由输入数据决定,样本数量

- 输出通道数（C_out）: 通过 out_channels 参数设置

- 输出深度（D_out）

- 输出高度（H_out)

- 输出宽度（W_out）

## 工作量：

3个月

## 方法逻辑：

D_out​=(D_in​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1

H_out​=(H_in​−1)×stride[1]−2×padding[1]+dilation[1]×(kernel_size[1]−1)+output_padding[1]+1

W_out​=(W_in​−1)×stride[2]−2×padding[2]+dilation[2]×(kernel_size[2]−1)+output_padding[2]+1

# 算子六：torch.linalg.inv

## 功能：

如果一个矩阵存在逆矩阵，计算逆矩阵。如果矩阵不可逆，则抛出 RuntimeError 异常

支持 float，double， cfloat 和 cdouble 数据类型的输入。还支持矩阵批次，如果 A 是一批矩阵，则输出对应批次的逆矩阵

## 输入参数：

A(Tensor) : （* ，n ，n） 的张量，* 是由可逆矩阵组成的零个或者多个批量的维度

## 输出参数：

out(Tensor) ：输出张量。

## 工作量：

4个月

## 方法逻辑：

若输入矩阵是一个n * n 的对称正定矩阵，可以使用递归 Cholesky 分解算法来计算其逆矩阵。通过Cholesky 分解，得到下三角矩阵 L , 使得 A = L * L^T 。 把AX看成 L * (L^T * L) = L * Y， 求 L * Y = I，得到 Y，求 L ^ T * X = Y, 得到 X，X 即为 A 的逆矩阵 。在昇腾上实现 Cholesky 分解算法需要考虑到昇腾芯片的特性。

若输入矩阵不是n * n 的对称正定矩阵，则可以使用 LU 分解算其逆矩阵。首先进行行列选主元 LU 分解：根据公式 PA = LU , 将源矩阵A 分解为单位下三角矩阵 L，上三角矩阵U 和 置换矩阵P；接着进行三角矩阵求逆：对L矩阵求逆得到其逆矩阵 L ^ -1 将 U 矩阵的转置矩阵求逆后再转置得到 U ^ -1 ; 最后记性矩阵相乘：将矩阵 U ^ -1 和矩阵 L^ -1 相乘，并根据矩阵P将矩阵乘法结果进行列变换得到源矩阵 A ^ -1。
