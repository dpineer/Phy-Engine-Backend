<p align="left">
  <img src="documents/images/logo.png" alt="phy engine logo"/>
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE.md)
[![Language](https://img.shields.io/badge/language-c++23-red.svg)](https://en.cppreference.com/)

# Phy Engine

Phy Engine 是一个电路仿真引擎，支持混合数字和模拟仿真以及 Verilog 模块。该引擎速度快、精度高，未来将作为电路引擎集成到 Physics Lab 中。

## 特性

* **混合信号仿真**: 支持数字和模拟电路的统一仿真
* **Verilog 支持**: 原生支持 Verilog 模块和数字逻辑仿真
* **高性能**: 针对速度优化，支持 CUDA 加速
* **高精度**: 使用先进的数值方法获得精确的仿真结果
* **跨平台**: 支持多种平台编译，包括 WebAssembly、原生桌面和移动平台
* **Physics Lab 集成**: 专为与 Physics Lab 无缝集成而设计

## 构建说明

### 前置条件
* C++23 兼容编译器 (GCC 13+, Clang 16+)
* CMake 3.20 或更高版本
* CUDA 支持: CUDA Toolkit 11.0 或更高版本
* WebAssembly 支持: Emscripten SDK

### 构建原生平台版本

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 构建 WebAssembly 版本

```bash
# 使用提供的 Dockerfile 以确保环境一致
docker build -t phy-engine-wasm src/targets/wasm32-emscripten/
# 或直接使用 emscripten
emcmake cmake -B build-wasm -DCMAKE_BUILD_TYPE=Release
emmake make -C build-wasm -j$(nproc)
```

## 使用方法

### 作为库使用
Phy Engine 可以作为 C++ 库集成到其他项目中。包含 `include/phy_engine/` 中的头文件并链接编译后的库。

### 作为共享库/DLL 使用
引擎可以编译为共享库，通过 C ABI 使用，使其可通过 FFI 从其他语言访问。

### Web 使用
编译为 WebAssembly 后，引擎可通过 JavaScript 绑定在 Web 应用中使用。

## 项目结构

* `benchmark/` - 性能基准测试和比较测试
* `custom/` - 自定义头文件和配置
* `documents/` - 中文文档文件
* `fuzz/` - 模糊测试工具
* `include/` - 头文件和第三方依赖
* `src/` - 源代码和构建目标
* `test/` - 包含 32+ 个测试类别的综合测试套件

## 文档

详细文档可在 `documents/docu.zh_CN/` 中找到：
* [00_头文件索引.md](documents/docu.zh_CN/00_头文件索引.md) - 头文件索引
* [01_快速开始.md](documents/docu.zh_CN/01_快速开始.md) - 快速开始指南
* [02_核心概念与数据结构.md](documents/docu.zh_CN/02_核心概念与数据结构.md) - 核心概念和数据结构
* [03_电路求解与分析类型.md](documents/docu.zh_CN/03_电路求解与分析类型.md) - 电路求解和分析类型
* [04_网表操作API.md](documents/docu.zh_CN/04_网表操作API.md) - 网表操作 API
* [05_模型库清单.md](documents/docu.zh_CN/05_模型库清单.md) - 模型库清单
* [06_数字逻辑与Verilog子集.md](documents/docu.zh_CN/06_数字逻辑与Verilog子集.md) - 数字逻辑与 Verilog 子集
* [07_PhysicsLab互操作.md](documents/docu.zh_CN/07_PhysicsLab互操作.md) - PhysicsLab 互操作
* [08_C_ABI_与共享库.md](documents/docu.zh_CN/08_C_ABI_与共享库.md) - C ABI 与共享库
* [09_文件格式_工具_FAQ.md](documents/docu.zh_CN/09_文件格式_工具_FAQ.md) - 文件格式、工具和 FAQ
* [10_Options_与配置参考.md](documents/docu.zh_CN/10_Options_与配置参考.md) - Options 与配置参考
* [11_API_逐函数参考.md](documents/docu.zh_CN/11_API_逐函数参考.md) - API 逐函数参考

## 测试

项目包含综合测试套件，涵盖 32+ 个不同测试类别：
* 模块功能
* 网表操作
* 电路求解
* 模型实现
* 数字逻辑仿真
* Verilog 编译
* DLL/共享库接口
* 数值方法
* CUDA 加速
* Physics Lab 包装器
* RISC-V 仿真
* 游戏实现 (俄罗斯方块、贪吃蛇)

## 性能

引擎针对性能进行了优化：
* 使用 Eigen 库进行高效的稀疏矩阵运算
* CUDA 加速并行计算
* 使用 fast_io 库进行快速 I/O 操作
* 优化的内存管理

## 贡献

我们欢迎对 Phy Engine 项目的贡献。请参阅文档和测试文件以了解有关代码结构和开发过程的更多信息。

## 许可证

本项目根据 Apache 2.0 许可证授权 - 详见 [LICENSE.md](LICENSE.md) 文件。

## 贡献者
* [MacroModel](https://github.com/MacroModel)
