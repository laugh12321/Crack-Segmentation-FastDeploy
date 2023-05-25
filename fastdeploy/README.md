# FastDeploy C++库在Xmake上的使用方式 (Windows)


## 1. 安装Xmake

可以从以下[链接](https://github.com/xmake-io/xmake/releases)下载安装Xmake，安装完成后，可以在命令行中使用 `xmake` 命令。

Windows 10 建议下载 `xmake-master.win64.exe`，双击安装即可。

## 2. 下载FastDeploy C++ 预编译库 

可以从以下链接下载编译好的 FastDeploy Windows 10 C++ SDK，解压至 `fastdeploy`。

```
https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-1.0.4.zip
```

## 3. 创建 Xmake 工程使用 C++ SDK

### 3.1 修改 `xmake.lua` 文件

在工程目录下 `xmake.lua` 文件中新增内容如下：

``` lua

-- CUDA
add_includedirs("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include")
add_linkdirs("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/lib")

-- FastDeploy C++ SDK (fastdeploy opencv yaml-cpp)
add_includedirs("fastdeploy/include", "fastdeploy/third_libs/install/opencv/build/include", "fastdeploy/third_libs/install/yaml-cpp/include")
add_linkdirs("fastdeploy/lib", "fastdeploy/third_libs/install/opencv/build/x64/vc15/lib", "fastdeploy/third_libs/install/yaml-cpp/lib")
add_links("fastdeploy", "opencv_world3416", "yaml-cpp")

```

### 3.2 编译工程

只需要进入项目根目录下，然后执行 `xmake` 命令就可以完成编译。

### 3.3 运行工程

在执行可执行文件之前，首先需要拷贝所有的dll到exe所在的目录下。

进入SDK的根目录，执行以下命令：

``` bash
# info参数为可选参数，添加info参数后会打印详细的安装信息
# bin 为dll安装目录
fastdeploy_init.bat install %cd% bin info
```

之后，执行 `xmake run` 即可运行工程。