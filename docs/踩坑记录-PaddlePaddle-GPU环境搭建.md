# PaddlePaddle GPU 环境搭建踩坑记录

> 环境：WSL2 (Ubuntu 24.04) / glibc 2.39 / CUDA 12.6 / NVIDIA 驱动 API 13.3 / GPU 算力 8.6
> 目标：在 PaddleDetection 上跑通 GPU 版 PaddlePaddle
> 最终可用组合：官方 CPython 3.10 + `paddlepaddle-gpu==2.6.2.post120` (cu120) + cuDNN 8.9.7

---

## 症状总览

按官方 `docs/tutorials/INSTALL_cn.md` 安装 `paddlepaddle-gpu==2.3.2` 后，`import paddle` 直接报错，随后每解决一个问题又冒出下一个。完整问题链如下。

---

## 坑 1：`_dl_sym, version GLIBC_PRIVATE` —— 老 Paddle 二进制不兼容新 glibc

### 报错
```
ImportError: .../paddle/fluid/core_avx.so: undefined symbol: _dl_sym, version GLIBC_PRIVATE
```
（前面那句 `Can not import avx core while this file exists...` 只是 Paddle 的提示，**不是** AVX 指令集问题。）

### 排查过程
- 一开始怀疑是 venv 用了 **uv 的 standalone Python**（`~/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu`，Clang 编译），换成官方 CPython 3.10（deadsnakes）后**报错依旧**。
- 清空 `LD_LIBRARY_PATH` 后仍然报同样的错 → 排除环境变量污染。
- `ldd core_avx.so` 显示它链接的是系统 `libc.so.6`（glibc 2.39）。

### 根因
`_dl_sym@GLIBC_PRIVATE` 是 glibc 的**内部私有符号**。glibc **2.34 起** libdl 并入 libc，该符号导出方式改变。**凡是针对 glibc < 2.34 编译、且引用了 `_dl_sym@GLIBC_PRIVATE` 的旧二进制，在 glibc ≥ 2.34 上一律无法加载。**

`paddlepaddle-gpu 2.3.2`（2022 年，manylinux2014 旧 glibc + CUDA 11.x）撞上 Ubuntu 24.04 的 glibc 2.39 → 硬性 ABI 不兼容。**换 Python、清环境变量、装 noavx 都无效**，坏的是 `.so` 二进制本身。

### 解决
升级到针对新 glibc 编译的 Paddle。`INSTALL_cn.md` 里写的 `2.3.2` 是**最低基线**（依赖表是 `>=2.3.2`），装更新版符合要求。

---

## 坑 2：cu123 源没有 2.6.2 / cu120 源版本号带后缀

### 报错
```
# cu123 源
ERROR: Could not find a version that satisfies the requirement paddlepaddle-gpu==2.6.2
(from versions: 3.0.0b0, ..., 3.1.0)

# cu120 源
ERROR: ... (from versions: 2.6.1.post120, 2.6.2.post120)
```

### 原因
- **cu123 源从 3.0 起步**，没有 2.6.x。
- 2.6.x 只在 cu120/cu118 源，且版本号带 CUDA 后缀 → 精确版本是 **`2.6.2.post120`**，不是 `2.6.2`。

### 解决
```bash
pip uninstall -y paddlepaddle-gpu paddlepaddle
pip install paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
```
> cu120 构建自带 CUDA 运行库，宿主机 CUDA 12.6 驱动向下兼容 12.0，可正常运行。
> 备选：直接用 cu123 源的稳定版 `3.1.0`（最贴合 CUDA 12.6，但 Paddle 3.x 与 2.x 有少量 API 差异）。

---

## 坑 3：`Cannot load cudnn shared library` —— GPU wheel 不带 cuDNN

### 报错
```
libcudnn.so: cannot open shared object file: No such file or directory
PreconditionNotMetError: Cannot load cudnn shared library. Cannot invoke method cudnnGetVersion.
```

### 原因
Paddle 的 GPU wheel **不打包 cuDNN**，需系统另装。且 **2.6.2 针对 cuDNN 8.x 编译**（找 `libcudnn.so.8`），装 cuDNN 9 无效。系统内 `find libcudnn.so*` 空 → 根本没装。

### 解决（pip 装，无需 sudo，装进 venv）
```bash
pip install "nvidia-cudnn-cu12==8.9.7.29"
# pip 包只给 libcudnn.so.8，Paddle 找 libcudnn.so，需建软链
CUDNN_DIR=$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__),'lib'))")
ln -s libcudnn.so.8 "$CUDNN_DIR/libcudnn.so"
```

---

## 坑 4：`libcuda.so: cannot open shared object file` + 段错误 —— WSL2 驱动路径

### 报错
```
libcuda.so: cannot open shared object file: No such file or directory
FatalError: `Segmentation fault` is detected by the operating system.
```

### 原因
`libcuda.so` 是 **NVIDIA 驱动**库（非 CUDA toolkit）。WSL2 下驱动库在 **`/usr/lib/wsl/lib/`**，不在 Paddle 默认搜索路径。ldconfig 只缓存了 `libcuda.so.1`，Paddle dlopen 找 `libcuda.so` 找不到。

### 解决
把 cuDNN 目录和 WSL 驱动目录都加进 `LD_LIBRARY_PATH`：
```bash
export LD_LIBRARY_PATH="$CUDNN_DIR:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
```

---

## 固化环境变量（免每次手动加）

把下面这段追加到 `.venv/bin/activate` 末尾，之后 `source .venv/bin/activate` 自动生效：
```bash
# --- Paddle GPU runtime libs (cuDNN + WSL libcuda) ---
export LD_LIBRARY_PATH="<绝对路径>/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
```

---

## 验证成功标志
```bash
python -c "import paddle; paddle.utils.run_check()"
```
```
device: 0, GPU Compute Capability: 8.6, cuDNN Version: 8.9.
PaddlePaddle works well on 1 GPU.
PaddlePaddle is installed successfully!
```

---

## 经验总结（换机器/重装避坑）

1. **别用 uv 的 standalone Python 装 Paddle** —— 会撞 `_dl_sym@GLIBC_PRIVATE`。用系统/官方 CPython（Ubuntu 24.04 需 deadsnakes PPA 装 3.10）。
2. **INSTALL_cn.md 的 2.3.2 是最低版本**，不是必须精确用。新系统（glibc ≥ 2.34 / CUDA 12.x）要装满足 `>=2.3.2` 的**新构建**，如 `2.6.2.post120`。
3. Paddle 官方源按 CUDA 分：`cu120`（2.6.x，版本号带 `.postXXX`）、`cu123`（3.0+）。
4. **GPU wheel 不带 cuDNN**，2.6.x 要配 **cuDNN 8.x**（不是 9），并建 `libcudnn.so` 软链。
5. **WSL2 驱动 `libcuda.so` 在 `/usr/lib/wsl/lib`**，必须加进 `LD_LIBRARY_PATH`。
