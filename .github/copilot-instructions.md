# unified_bench 开发规范（GitHub Copilot 指令）

本文件是 `d:\WorkSpace\unified_bench`（多后端推理基准测试工具，C++ / CMake，支持
ONNX Runtime / MNN / NCNN / TFLite / LiteRT / QNN SDK 等）的 Copilot 编码规范。
所有代码改动必须遵循以下规则；与代码相关的说明文档同步维护（见第 9 节）。

---

## 1. 语言要求（代码全英文，聊天全中文）

- **所有源代码文件（.c/.cpp/.h/.hpp/.hxx 等）中的注释、日志字符串、变量名、函数名、
  宏名必须全部使用英文**。
- 禁止在源代码文件中出现任何中文字符，包括中文标点（如 `—`、`：`、`，`、`。`、
  `《`、`》`、`年`、`月`、`日` 等）。编码建议统一 UTF-8（避免 MSVC C4819 警告）。
- 聊天回复（解释、建议、待办项、文档说明）使用中文，但**代码块内必须全英文**。
- 生成的文件内容（如 CSV 表头、模型名）也应为英文。

## 2. 模块化编程

- 功能按模块拆分（类、命名空间、独立文件），便于维护与升级。
- 每个模块有清晰接口定义（见 `include/*.hpp`），避免全局变量；需要共享的状态放在
  显式的注册表/管理器类中（如 `BackendRegistry`）。
- 优先 RAII 与智能指针（`std::unique_ptr`、`std::shared_ptr`）管理资源。
- 新后端应实现 `include/backend_interface.hpp` 中的 `IBackend` 接口，并在
  `src/backend_registry.cpp` 注册，而非散落各处。

## 3. 内存安全

- 禁止裸 `new`/`delete`（除非极特殊原因并注释说明），动态内存使用智能指针或容器。
- 所有数组/向量访问必须做边界检查（优先 `at()` 或先校验索引再 `[]`）。
- 避免 C 风格字符串函数（`strcpy`、`sprintf`、`strcat`），改用 `std::string`、
  `snprintf`、`std::ostringstream`。
- 所有资源（文件句柄、DLL/so 句柄、QNN/DML 对象、锁）必须保证异常路径也能正确释放
  （RAII 或析构函数中统一清理，参照 `ONNXBackend::Cleanup()`）。
- 分配失败（`malloc`/`new` 返回空）必须检查并返回失败，禁止直接解引用。

## 4. 日志系统

工程已有轻量级日志（`include/log.hpp` / `src/log.cpp`，`Logger` 静态类），改动须保持
以下约定：

- **a. 多等级日志**：`LogLevel::OFF/DBG/INFO/WARN/ERR`。默认级别为 `WARN`；
  高频调试日志必须用 `LOGD`（`LogLevel::DBG`），并保证在 `INFO` 级别下可整体关闭以
  降低开销（`Logger::level` 运行时切换，命令行 `--log-level` 控制）。
- **b. 日志格式**（目标统一格式）：
  ```
  date time programe_name [pid.tid] log_level file_name:line_number @fun_name log_string
  ```
  例如：`2026-08-05 15:00:00 unified_bench [1234.56] INFO onnx_backend.cpp:128 @ConfigureEP ...`
- **c. PC 平台日志落盘位置**：**推荐：程序所在目录为主，临时目录为兜底**。
  理由分析：
  - 程序所在目录（exe 同目录）：日志与程序/结果文件在一起，用户容易找到；本工具通常
    运行在用户可写目录（如 `d:\WorkSpace\...` 或构建输出目录），无权限问题；便于与
    `summary.csv` 一并归档分析。风险：若安装在 `Program Files` 等只读目录则无法写入。
  - 临时目录（`%TEMP%` / `/tmp`）：永远可写，但日志分散、可能被系统清理、用户难以
    定位，且与本工具"结果落盘"的习惯不一致。
  - 结论：优先写程序所在目录；若打开失败（权限/只读），降级写临时目录，并输出一条
    WARN 说明实际落盘路径。
  - **i. 日志文件大小限制 50MB**：纯文本足够。达到上限后轮转（如备份为
    `xxx.log.1` 后重开新文件），避免无限增长。
- **d. Android 平台**：默认输出到 logcat（`__android_log_print`，tag 用
  `prog_name`）。logcat 会自动附加日期/时间/pid/tid/tag 前缀，**不要在消息中再次获取
  并打印这些前缀，避免重复**。
- **e. 日志写入失败的降级行为**：文件打开/写入失败时不得崩溃；降级策略为
  回退到 stderr 控制台输出，并设置标志位避免每次写都重试报错（如每分钟最多重试一次）；
  若连 stderr 也不可用，则静默丢弃并计数，绝不进入死循环或抛出异常。

## 5. 跨平台兼容性

- **目标操作系统**：Windows / Linux / Android。
  - Windows：`_WIN32`/`_WIN64` 分支（DLL 加载、`CreateFile`/`GetSystemInfo`、
    `AddVectoredExceptionHandler`、WMI 等）。
  - Linux：`__linux__`（`dlopen`/`dlsym`、`posix_memalign`、`dma_buf` 等）。
  - Android：`__ANDROID__`/`__android__`（logcat、QNN/NNAPI/GPU 委托、NDK 构建）。
  - 平台相关代码必须用预处理分支隔离，避免在某平台引入编译/链接依赖（参照
    `platform.hpp`、`device_info.cpp`）。
- **编译器**：MSVC（VS 2022）、MinGW、GCC、Clang。
  - 避免 MSVC 特有扩展；使用 C++17 标准特性；`snprintf`、`strcasecmp` 等在不同平台
    名称不同，需提供封装（参照 `platform.hpp` 中的 `stricmp_` 等）。
  - 保证代码在 x86 / x64 / ARM64 均编译通过（本工具同时产出 win-x86 / win-x64 /
    android-arm64 构建）。
  - 新增代码必须通过 `cmake --build build/win-x64 --config Release` 与
    `build/win-x86 --config Release` 验证。

## 6. 显式检查所有返回值

- **所有系统调用与自定义函数必须显式检查返回值**（不得 `(void)` 丢弃）。
- 若返回异常值（`nullptr`、`-1`、`false`、非 0 状态码），必须记录错误日志，格式统一为：
  ```
  "open %s failed. due to %s, %d", file_path, strerror(errno), errno
  ```
  其中 `errno` 为系统错误码，`strerror` 提供可读描述。
- 对 ORT C API（返回 `OrtStatus*`）：统一走 `OrOk()` 辅助函数（`src/onnx_backend.cpp`），
  失败时日志输出 `ONNX: <what> failed: <msg>` 并写入 `last_error_`，禁止静默忽略。
- 检查失败后必须向上返回失败（`return false` / 错误码），让调用方（`BenchmarkRunner`）
  标记后端失败（`avg=0, accel=-1`），**绝不静默降级为 CPU 并上报误导性结果**。

## 7. 多线程安全

- 涉及多线程（如音频/推理线程与 UI 线程、跨后端共享输入）时，必须使用适当的同步机制
  （`std::mutex`、`std::atomic`、`std::condition_variable`）。
- 避免死锁：优先 `std::lock_guard` / `std::unique_lock`，并保持一致的加锁顺序。
- 实时/高频路径（如推理主循环、GPU 回调节点）中禁止加锁或仅使用无锁结构，避免竞态。
- 共享数据（如 `InputProvider`、`ResultCollector`）需明确线程归属或用锁保护。

## 8. 编码规范与可读性

- **a. 函数返回值设计要合理**：不应只有 `0`/`-1`。布尔语义用 `bool`；多状态用枚举；
  需携带原因时用 `bool + out 参数`（如 `last_error_`）或 `std::optional`/错误码枚举，
  返回值的含义要在头文件注释中说明。
- **b. 所有 `if/else/for/while/switch/case` 都必须加花括号 `{}`**，包括每个
  `case/default` 分支体；`else if` 保持链式但 if 体必须加括号。
  合规性检查：`python tools/utils/check_braces.py`（无参运行默认扫 `src/` 与
  `include/`，即仓库根目录执行即可）。退出码 `0`=合规、`1`=有违规、`2`=**扫描未
  完成**（路径不存在/无文件扫到/读取失败）。**只有退出码 0 才算通过**——输出
  `TOTAL: 0` 但退出码为 2 表示"什么都没扫到"，是假绿灯，不得视为合规（历史上
  该脚本的默认路径硬编码到不存在的目录，长期打印 `TOTAL: 0` 却一个文件都没扫）。
  已接入 CI（`.github/workflows/ci.yml` 的 "Brace style check" 步骤）。
- **c. 命名与 const 正确性**：变量/函数用 `snake_case`，类型/类用 `PascalCase`，
  宏用 `UPPER_SNAKE_CASE`；避免魔法数字（用 `constexpr` 命名常量）；能用
  `const`/`constexpr` 的地方必须用；头文件加 `#pragma once`。
- 保持代码简洁可读：注释解释"为什么"而非"是什么"；函数不宜过长，单一职责。

## 9. 文档同步（docs/ 目录）

- 所有代码改动必须同步更新 `docs/` 下的对应文档，**不要让文档与代码脱节**。
- 各后端排查与修复记录维护在 `docs/*_DEBUG_LOG.md`（如 `docs/ONNX_DEBUG_LOG.md`、
  `docs/NCNN_DEBUG_LOG.md`、`docs/QNN_SDK_DEBUG_LOG.md`、`docs/TFLITE_NPU_DEBUG_LOG.md`
  等）。每次问题排查：记录现象、根因、解决方案、实测数据（avg/accel/diff）。
- 新增/修改构建脚本或依赖（如升级 DirectML.dll）时，同步更新对应文档与
  `tools/*.bat` 脚本注释。
- 结论性经验（如"DML 图融合会产生数值错误必须关闭"、"DirectML.dll 从
  onnxruntime.dll 同目录加载"）必须写入文档并标注日期，避免重复排查。
