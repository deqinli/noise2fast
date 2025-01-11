# 1、noise2fast 论文的C++代码实现
## 无监督深度学习图像降噪C++版本，另外Python版本参考论文noise2fast经典论文

此工程环境如下：
   + vs2022
   + libtorch2.5.1+cuda12.1
   + opencv4.8.0
   + cuda12.1
   + cudnn8.9.7.29_cuda12

包含头文件夹（需对应自己的目录）：
   + libtorch2.5.1_debug\include
   + libtorch2.5.1_debug\include\torch\csrc\api\include
   + NVIDIA GPU Computing Toolkit\CUDA\v12.1\include
   + opencv480\include
   + opencv480\include\opencv2

库文件夹（需对应自己的目录）：
+ libtorch2.5.1_debug\lib
+ NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64
+ opencv480\lib（此OpenCV库是我自己编译的，也可以使用opencv官方的库）

链接库文件：
+ asmjit.lib
+ c10.lib
+ c10_cuda.lib
+ cpuinfo.lib
+ dnnl.lib
+ fbgemm.lib
+ fmtd.lib
+ kineto.lib
+ libprotobufd.lib
+ libprotobuf-lited.lib
+ libprotocd.lib
+ pthreadpool.lib
+ sleef.lib
+ torch.lib
+ torch_cpu.lib
+ torch_cuda.lib
+ XNNPACK.lib
+ cuda.lib
+ cudart.lib
+ cudnn.lib
+ cudnn_adv_infer.lib
+ cudnn_adv_infer64_8.lib
+ opencv_world480d.lib

在【链接器】->【命令行】->【其他选项】：添加如下代码（如果没有这一句，程序不会使用cuda计算）
+ /INCLUDE:"?ignore_this_library_placeholder@@YAHXZ" 

# 2、python版本
见  [noise2fast](https://github.com/jason-lequyer/Noise2Fast)

# 3、原论文
见工程中文件 “ noise2fast.pdf ” 或 [原文](https://arxiv.org/pdf/2108.10209v1)
