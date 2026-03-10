# Phy-Engine 共享库接口使用文档

## 概述
Phy-Engine 是一个物理仿真引擎，编译后生成 `libphyengine.so` 共享库，可通过 C ABI 接口被 Linux 程序调用。

## 编译产物
- `libphyengine.so` - 主要的物理引擎共享库
- `verilog2plsav` - Verilog 到 PhysicsLab .sav 文件转换工具
- `verilog2penl` - Verilog 到 PENL(leveldb) 导出工具

## 实际API接口

### 枚举类型

#### 仿真类型 (phy_engine_analyze_type)
- `PHY_ENGINE_ANALYZE_OP` = 0 (工作点分析)
- `PHY_ENGINE_ANALYZE_DC` = 1 (直流分析)
- `PHY_ENGINE_ANALYZE_AC` = 2 (交流分析)
- `PHY_ENGINE_ANALYZE_ACOP` = 3 (交流工作点分析)
- `PHY_ENGINE_ANALYZE_TR` = 4 (瞬态分析)
- `PHY_ENGINE_ANALYZE_TROP` = 5 (瞬态工作点分析)

#### 数字状态 (phy_engine_digital_state)
- `PHY_ENGINE_D_L` = 0 (低电平)
- `PHY_ENGINE_D_H` = 1 (高电平)
- `PHY_ENGINE_D_X` = 2 (未知)
- `PHY_ENGINE_D_Z` = 3 (高阻态)

#### 元件代码 (phy_engine_element_code)
线性元件:
- `PHY_ENGINE_E_RESISTOR` = 1 (电阻) - 属性: r
- `PHY_ENGINE_E_CAPACITOR` = 2 (电容) - 属性: C
- `PHY_ENGINE_E_INDUCTOR` = 3 (电感) - 属性: L
- `PHY_ENGINE_E_VDC` = 4 (直流电压源) - 属性: V
- `PHY_ENGINE_E_VAC` = 5 (交流电压源) - 属性: Vp, freq(Hz), phase(deg)
- `PHY_ENGINE_E_IDC` = 6 (直流电流源) - 属性: I
- `PHY_ENGINE_E_IAC` = 7 (交流电流源) - 属性: Ip, freq(Hz), phase(deg)

受控源:
- `PHY_ENGINE_E_VCCS` = 8 (电压控制电流源) - 属性: G
- `PHY_ENGINE_E_VCVS` = 9 (电压控制电压源) - 属性: Mu
- `PHY_ENGINE_E_CCCS` = 10 (电流控制电流源) - 属性: alpha
- `PHY_ENGINE_E_CCVS` = 11 (电流控制电压源) - 属性: r

其他元件:
- `PHY_ENGINE_E_SWITCH_SPST` = 12 (单刀单掷开关) - 属性: cut_through (0/1)
- `PHY_ENGINE_E_PN_JUNCTION` = 13 (PN结) - 属性: Is,N,Isr,Nr,Temp,Ibv,Bv,Bv_set(0/1),Area
- `PHY_ENGINE_E_TRANSFORMER` = 14 (变压器) - 属性: n
- `PHY_ENGINE_E_COUPLED_INDUCTORS` = 15 (耦合电感) - 属性: L1,L2,k

数字逻辑元件:
- `PHY_ENGINE_E_DIGITAL_INPUT` = 200 (数字输入) - 属性: state (0=L,1=H,2=X,3=Z)
- `PHY_ENGINE_E_DIGITAL_OUTPUT` = 201 (数字输出)
- `PHY_ENGINE_E_DIGITAL_OR` = 202 (或门)
- `PHY_ENGINE_E_DIGITAL_YES` = 203 (是门)
- `PHY_ENGINE_E_DIGITAL_AND` = 204 (与门)
- `PHY_ENGINE_E_DIGITAL_NOT` = 205 (非门)
- `PHY_ENGINE_E_DIGITAL_XOR` = 206 (异或门)
- `PHY_ENGINE_E_DIGITAL_XNOR` = 207 (同或门)
- `PHY_ENGINE_E_DIGITAL_NAND` = 208 (与非门)
- `PHY_ENGINE_E_DIGITAL_NOR` = 209 (或非门)
- `PHY_ENGINE_E_DIGITAL_TRI` = 210 (三态门)
- `PHY_ENGINE_E_DIGITAL_IMP` = 211 (蕴含门)

- `PHY_ENGINE_E_DIGITAL_NIMP` = 212 (非蕴含门)

组合逻辑元件:
- `PHY_ENGINE_E_DIGITAL_HALF_ADDER` = 220 (半加器)
- `PHY_ENGINE_E_DIGITAL_FULL_ADDER` = 221 (全加器)
- `PHY_ENGINE_E_DIGITAL_HALF_SUBTRACTOR` = 222 (半减器)
- `PHY_ENGINE_E_DIGITAL_FULL_SUBTRACTOR` = 223 (全减器)
- `PHY_ENGINE_E_DIGITAL_MUL2` = 224 (2位乘法器)
- `PHY_ENGINE_E_DIGITAL_DFF` = 225 (D触发器)
- `PHY_ENGINE_E_DIGITAL_TFF` = 226 (T触发器)
- `PHY_ENGINE_E_DIGITAL_T_BAR_FF` = 227 (T反相触发器)
- `PHY_ENGINE_E_DIGITAL_JKFF` = 228 (JK触发器)

### 主要函数接口

#### 电路创建
```c
void* create_circuit(int* elements,           // 元件代码数组
                     size_t ele_size,        // 元件数量
                     int* wires,             // 连接线数组 [ele1,pin1,ele2,pin2]
                     size_t wires_size,      // 连接线数组大小(必须是4的倍数)
                     double* properties,     // 属性数组
                     size_t** vec_pos,       // 输出: 元件位置信息
                     size_t** chunk_pos,     // 输出: 元件块位置信息
                     size_t* comp_size);     // 输出: 元件总数
```

```c
void* create_circuit_ex(int* elements,               // 元件代码数组
                        size_t ele_size,            // 元件数量
                        int* wires,                 // 连接线数组 [ele1,pin1,ele2,pin2]
                        size_t wires_size,          // 连接线数组大小
                        double* properties,         // 属性数组
                        char const* const* texts,   // Verilog源码文本数组
                        size_t const* text_sizes,   // 文本大小数组
                        size_t text_count,          // 文本数量
                        size_t const* element_src_index,  // 元件源码索引
                        size_t const* element_top_index,  // 元件顶层模块索引
                        size_t** vec_pos,           // 输出: 元件位置信息
                        size_t** chunk_pos,         // 输出: 元件块位置信息
                        size_t* comp_size);         // 输出: 元件总数
```

#### 电路销毁
```c
void destroy_circuit(void* circuit_ptr,    // 电路指针
                     size_t* vec_pos,     // 元件位置信息
                     size_t* chunk_pos);  // 元件块位置信息
```

#### 仿真控制
```c
int circuit_set_analyze_type(void* circuit_ptr, uint32_t analyze_type_value);  // 设置分析类型
int circuit_set_tr(void* circuit_ptr, double t_step, double t_stop);           // 设置瞬态分析参数
int circuit_set_ac_omega(void* circuit_ptr, double omega);                    // 设置交流分析角频率
int circuit_set_temperature(void* circuit_ptr, double temp_c);                 // 设置温度
int circuit_set_tnom(void* circuit_ptr, double tnom_c);                       // 设置标称温度
int circuit_set_model_double_by_name(void* circuit_ptr, size_t vec_pos, size_t chunk_pos, 
                                     char const* name, size_t name_size, double value);  // 按名称设置模型参数
int circuit_analyze(void* circuit_ptr);                                        // 执行分析
int circuit_digital_clk(void* circuit_ptr);                                    // 数字时钟更新
```

#### 数据采样
```c
int circuit_sample(void* circuit_ptr,      // 电路指针
                   size_t* vec_pos,       // 元件位置信息
                   size_t* chunk_pos,     // 元件块位置信息
                   size_t comp_size,      // 元件总数
                   double* voltage,       // 电压输出数组
                   size_t* voltage_ord,   // 电压顺序数组
                   double* current,       // 电流输出数组
                   size_t* current_ord,   // 电流顺序数组
                   bool* digital,         // 数字状态输出数组
                   size_t* digital_ord);  // 数字状态顺序数组
```

```c
int circuit_sample_u8(void* circuit_ptr,      // 电路指针
                      size_t* vec_pos,       // 元件位置信息
                      size_t* chunk_pos,     // 元件块位置信息
                      size_t comp_size,      // 元件总数
                      double* voltage,       // 电压输出数组
                      size_t* voltage_ord,   // 电压顺序数组
                      double* current,       // 电流输出数组
                      size_t* current_ord,   // 电流顺序数组
                      uint8_t* digital,      // 数字状态输出数组(字节格式)
                      size_t* digital_ord);  // 数字状态顺序数组
```

#### 数字状态设置
```c
int circuit_set_model_digital(void* circuit_ptr, size_t vec_pos, size_t chunk_pos, 
                              size_t attribute_index, uint8_t state);  // 设置数字模型状态
```

#### 综合分析
```c
int analyze_circuit(void* circuit_ptr,      // 电路指针
                    size_t* vec_pos,       // 元件位置信息
                    size_t* chunk_pos,     // 元件块位置信息
                    size_t comp_size,      // 元件总数
                    int* changed_ele,      // 改变的元件数组
                    size_t* changed_ind,   // 改变的索引数组
                    double* changed_prop,  // 改变的属性数组
                    size_t prop_size,      // 属性数组大小
                    double* voltage,       // 电压输出数组
                    size_t* voltage_ord,   // 电压顺序数组
                    double* current,       // 电流输出数组
                    size_t* current_ord,   // 电流顺序数组
                    bool* digital,         // 数字状态输出数组
                    size_t* digital_ord);  // 数字状态顺序数组
```

## 使用方法

### 1. 链接共享库
```bash
gcc -o my_program my_program.c -lphyengine
```

### 2. 在 C/C++ 程序中使用
```c
#include <phy_engine/dll_api.h>

// 使用引擎功能
```

### 3. 运行时加载
```c
#include <dlfcn.h>

void* handle = dlopen("./libphyengine.so", RTLD_LAZY);
// 加载函数指针并使用
dlclose(handle);
```

## 注意事项
- 确保运行时能找到共享库（LD_LIBRARY_PATH 或 rpath）
- 引擎支持 OpenMP 并行加速
- 可选 CUDA 加速（需编译时启用）
- LevelDB 支持用于高效网表存储
