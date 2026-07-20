# ONNX Runtime Backend 修复记录

> **日期**: 2026-07-20
> **范围**: `src/onnx_backend.cpp`, `src/benchmark_runner.cpp`

---

## 1. ORT_API_VERSION 硬编码 → 运行时动态获取

### 问题

`onnx_backend.cpp` 中 `base->GetApi(ORT_API_VERSION)` 硬编码编译时头文件的 API 版本号（如 22）。当运行时链接不同版本的 `libonnxruntime.so` 时，API 版本不匹配 → `GetApi` 返回 NULL 或行为异常。

### 修复

通过 `GetVersionString()` 获取运行时版本字符串（如 `"1.22.0"`），解析小版本号作为 API 版本：

```cpp
// 旧: 硬编码编译时版本号
ort_ = base->GetApi(ORT_API_VERSION);

// 新: 运行时动态获取
const char *ort_ver_str = base->GetVersionString(); // "1.22.0"
ort_api_ver_ = 1;  // fallback
if (ort_ver_str) {
    LOGI("ONNX Runtime version: %s", ort_ver_str);
    const char *dot = strchr(ort_ver_str, '.');
    if (dot) ort_api_ver_ = (uint32_t)atoi(dot + 1);  // 22
}
ort_ = base->GetApi(ort_api_ver_);
```

同时将两处 DML `GetExecutionProviderApi("DML", ...)` 中的 `ORT_API_VERSION` 替换为成员变量 `ort_api_ver_`。

### 关键改动

- 添加成员变量 `uint32_t ort_api_ver_ = 1`
- 三处调用全部改用运行时版本

---

## 2. 经验总结

| 问题 | 教训 |
|------|------|
| ORT_API_VERSION 硬编码 | 任何 SDK API 版本号都不应硬编码编译时常量，必须运行时获取 |
| 版本兼容性 | `GetVersionString()` 返回格式 `major.minor.patch`，minor == API version |
