# CUDA

# 1.WHY CUDA? WHY NOW?

## 1.1WHY CUDA

CUDA时建立在NVIDIA的GPUs上的一个通用并行计算平台和编程模型，基于CUDA编程可以利用GPUs的并行计算引擎来更加高效地解决比较复杂的计算难题。近年来，GPU最成功的一个应用就是深度学习领域，基于GPU的并行计算已经成为训练深度学习模型的标配。



## 1.2CPU、GPU之间的关系

GPU并不是一个独立运行的计算平台，而需要与CPU协同工作，可以看成是CPU的协处理器，因此我们说GPU并行计算时，其实是指的基于CPU+GPU的异构计算构架。	

在异构计算架构中，GPU与CPU通过PCIe总线连接在一起来协同工作，CPU所在位置称为为主机端（host），而GPU所在位置称为设备端（device），如下图所示。

![img](https://pic3.zhimg.com/80/v2-df49a98a67c5b8ce55f1a9afcf21d982_1440w.jpg)

### 1.2.1区别

|              | CPU                      | GPU                      |
| ------------ | :----------------------- | ------------------------ |
| cores        | 运算核心较少             | 运算核心很多             |
| 运算任务类型 | 实现复杂的逻辑运算       | 数据并行的计算密集型任务 |
| 线程类型     | 重量级，上下文切换开销大 | 轻量级                   |

​	因此，基于CPU+GPU的异构计算平台可以优势互补，CPU负责处理逻辑复杂的串行程序，而GPU重点处理数据密集型的并行计算程序，从而发挥重大功效。

![img](https://pic2.zhimg.com/80/v2-2959e07a36a8dc8f59280f53b43eb9d1_1440w.jpg)

# 2.Getting start

## 2.1CUDA编程基础模型介绍

在CUDA中，**host**和**device**是两个重要的概念：

- host：CPU及其内存
- device：GPU及其内存

CUDA程序中既包含host程序，又包含device程序，它们分别在CPU和GPU上运行。同时，host与device之间可以进行通信，这样它们之间可以进行数据拷贝。典型的CUDA程序的执行流程如下：

1. 分配host内存，并进行数据初始化；
2. 分配device内存，并从host将数据拷贝到device上；
3. 调用CUDA的核函数在device上完成指定的运算；
4. 将device上的运算结果拷贝到host上；
5. 释放device和host上分配的内存。

## 2.2核函数kernel

以上流程中最重要的一个过程时调用CUDA的核函数来执行并行计算，`kernel`是CUDA中一个重要的概念，`kernel`是在device中并行执行的函数，核函数用`__global__`符号声明，在调用时需要用`<<<grid,block>>>`来指定kernel要执行的线程数量，在CUDA中，每一个线程都要执行核函数，并且每个线程会分配一个唯一的线程号`thread_ID`这个ID值可以通过核函数的内置变量`threadIdx`来获得。

由于GPU实际上是异构模型，所以需要区分host和device上的代码，在CUDA中是通过函数类型限定词来区别host和device上的函数，主要的三个函数类型限定词如下：

- `__global__`：在device上执行，从host中调用（一些特定的GPU也可以从device上调用），返回类型必须是`void`，不支持可变参数，不能成为类成员函数。主要用`__global__`定义的kernel是异步的，这意味着host不会等待kernel执行完而会直接执行下一步。
- `__device__`：在device上执行，仅可以从device中调用，不可以和`__global__`同时用。
- `__host__`：在host上执行，仅可以从host中调用，一般省略不写，不可以和`__global__`同时用，但可和`__device__`同时用，此时函数会在device和host都编译。

## 2.3核函数的层次结构

要深刻理解kernel，必须要对kernel的线程层次结构有一个清晰的认识。kernel在device上执行时实际上是启动很多线程。

- **网格**（grid）：一个kernel所启动的所有线程称为一个网格。同一个网格是上的线程共享相同的全局内存空间，grid是线程结构的第一层次。
- **线程块**（block）：网格又可以分为很多线程块，一个线程块里面包括很多线程，block是线程结构的第二层次。
- ![img](https://pic1.zhimg.com/80/v2-aa6aa453ff39aa7078dde59b59b512d8_720w.jpg)



线程两层组织结构如上图所示，这是一个grid和block均为二维的线程组织。grid和block的定义为`dim3`类型的变量，`dim3`可以看成是包含三个无符号整数（x，y，z）成员的结构体变量，在定义时，缺省值初始化为1。因此grid和block可以灵活地定义为1-dim，2-dim以及3-dim结构，对于图中结构（主要水平方向为x轴，竖直方向为y轴），定义的grid和block如下所示，kernel在调用时也必须通过执行配置`<<<grid,block>>>`来指定kernel所使用的线城市及结构：

- grid：大小`(3,2)`
- block：大小`(5,3)`
- kernel：名称`kernel_fun`

```c++
dim3 grid(3,2);
dim3 block(5,3);
kernel_fun <<< grid, block >>>(prams...);
```

因此，一个线程需要两个内置的坐标变量`(blockIdx, threadIdx)`来唯一标识，它们都是`dim3`类型变量，其中`blockIdx`指明线程所在grid中的位置，而`threadIdx`指明线程所在block中的位置，如图中的Thread(1, 1)满足：

```c++
blockIdx = dim3(1,1)
threadIdx = dim3(1,1)
```

### 2.3.1如何获取一个线程的ID

一个线程块上的线程是放在同一个流式多处理器（SM）上的，但是单个SM的资源有限，这导致线程块中的线程数是有限制的，现代GPUs的线程块可支持的线程数可达1024个。

有时候，我们要知道一个线程在block中的全局ID，就必须要知道线程块（block）的组织结构，这是通过线程的内置变量`blockDim`来获得的。

- `gridDim`：获取网格各个维度的大小。

- `blockDim`：获取线程块各个维度的大小。

对于一个2-dim的grid$(D_x,D_y)$，线程块（x，y）的ID值为（x+y*$D_x$）,如果3-dim的grid（$D_x, D_y,D_z$），线程块（x, y, z）的ID值为（ $x+y * D_x + z*D_x*D_y$）。

对于一个2-dim的block$(D_x,D_y)$，线程（x，y）的ID值为（x+y*$D_x$）,如果3-dim的block（$D_x, D_y,D_z$），线程（x, y, z）的ID值为（ $x+y * D_x + z*D_x*D_y$）。

### 2.3.2核函数实现矩阵加法

kernel的这种线程组织结构天然适合vector，matrix等运算，如我们将上图2-dim结构实现两个矩阵的加法，每个线程负责处理每个位置的两个元素相加，代码如下所示。线程块大小为（16，16），

然后将N*N大小的矩阵均分为不同的线程块来执行加法运算。

```c++
//Kernel definitions
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}
int main()
{
	...
    // Kernel blocks'config
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd <<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```





