# InstantHMR on Windows（中文速查）

面向 Windows + NVIDIA 显卡的完整流程：一键配置环境 → 下模型 → 用三个 `.bat` 跑图 / 视频 / 摄像头。
更细的实测记录（性能数字、调参过程、踩过的坑）在 [`RUN.md`](RUN.md)。

> 只用三个命令就够：
> ```bat
> setup_windows.bat
> run_image.bat assets\test_zidane.jpg
> run_video.bat D:\videos\clip.mp4
> ```

---

## 1. 前置条件

| 项目 | 要求 |
| --- | --- |
| 系统 | Windows 10/11 x64 |
| Python | 3.10 ~ 3.12（**推荐 3.12**，用 `py -3.12` 启动；不要把系统里 3.13/3.14 当默认） |
| 显卡 | NVIDIA，驱动 525+（本机实测 RTX 3090 + CUDA 12.6 正常） |
| 磁盘 | 预留 ~5 GB（venv 里的 torch 约 3 GB，模型约 0.3 GB） |
| 网络 | 首次要访问 Hugging Face / PyPI；国内建议挂代理或换镜像 |

---

## 2. 一键配置：`setup_windows.bat`

双击运行，或命令行：

```bat
setup_windows.bat                                                     :: 全套：建 venv + 装依赖 + 下模型
setup_windows.bat --check                                             :: 只体检，不装东西
setup_windows.bat --proxy http://127.0.0.1:21882                      :: 走代理下载
setup_windows.bat --model medium --fit 0,1,2                          :: 换模型 / 多下几个 fit 图
```

它依次做四件事：

1. 没有 `.venv` 时用 `py -3.12 -m venv .venv` 建一个（已存在则复用）；
2. 装 **CUDA 版** torch / torchvision（`--index-url https://download.pytorch.org/whl/cu126`，
   失败自动回退 CPU 版并提示——检测器会因此跑在 CPU 上，慢 4 倍左右）；
3. 装 `requirements.txt`（onnxruntime-gpu、rfdetr、rerun-sdk、huggingface_hub 等）；
4. 调 `tools\prefetch_models.py` 把 ONNX 图预先下到 Hugging Face 缓存。

结束时打印体检结果：`torch 2.9.1+cu126 | cuda: True NVIDIA GeForce RTX 3090`、
onnxruntime 的 provider 列表（应含 `CUDAExecutionProvider`）、rfdetr/rerun 是否在位。

全部可重复执行；`--check` 用来确认当前环境是否已经可用。

---

## 3. 模型：下哪些、放哪里

demo 会自动从 [Hugging Face `momolesang/InstantHMR`](https://huggingface.co/momolesang/InstantHMR)
下载，落到缓存目录（`%USERPROFILE%\.cache\huggingface\hub`），**不需要手动摆到 `models\`**。
`setup_windows.bat` / `tools\prefetch_models.py` 只是把它提前做掉。

| 文件 | 用途 | 大小 | 许可 |
| --- | --- | --- | --- |
| `instanthmr-medium-commercial.onnx` | 默认主模型（网络，无 crop 细化） | 123.4 MB | 商用可 |
| `instanthmr-medium-commercial-fit1.onnx` | `--fit 1`：单个 crop 的运动学拟合 | 79.3 MB | 商用可 |
| `instanthmr-medium-commercial-fit2.onnx` | `--fit 2`：双 crop，根关节最稳（**推荐**） | 79.4 MB | 商用可 |
| `mhr-body.onnx` | MHR body 图，把姿态参数解码成 18,439 顶点 mesh | 12.7 MB（12.1 MiB） | 随仓库 LICENSE-SAM |
| `instanthmr-medium.onnx` / `-fit1` / `-fit2` | 精度更高的非商用档 | 93.7 / 79.3 / 79.4 MB | **SAM License，非商用** |
| `instanthmr-{nano,tiny,small}*.onnx` | 更小更快，精度更低 | 56 ~ 65 MB | 非商用 |

补充说明：

- **`-fitN` 图必须配 `--fit N`**，混用会报错或结果异常。
- 想要商用，只用 `medium-commercial`（其余档位都是 SAM License 非商用）。
- 需要 PyTorch 权重（微调用）时：`pytorch\medium-commercial\instanthmr-medium-commercial.pth`
  （129.4 MB）+ 同目录 `run_config.json`，然后用
  `python tools\export_onnx.py --ckpt <pth> --output models\instanthmr-medium-commercial.onnx`
  自己导出 ONNX（`--fit 1|2` 导出带拟合的图，需额外 `training\checkpoints\mhr_model.pt`）。
- 手动下载也行：`https://huggingface.co/momolesang/InstantHMR/resolve/main/<文件名>`，放进
  `models\` 后在命令里用 `--model models\xxx.onnx` 指定。

---

## 4. 三个 `.bat` 怎么用

三个脚本都做了同样的事：把 venv 里 `nvidia\{cudnn,cublas,cuda_runtime,cufft,curand,cuda_nvrtc,nvjitlink}\bin`
注入 `PATH`（CUDA EP 动态加载子库需要，否则报 DLL 找不到），检查模型是否存在，然后调 `demo.py`。
额外参数**原样追加在命令行后面**即可。

### 4.1 `run_image.bat` —— 单张图片

```bat
run_image.bat                                   :: 不带参数 = assets\test_zidane.jpg
run_image.bat D:\photos\a.jpg
run_image.bat D:\photos\a.jpg --no-mesh         :: 只看骨架
run_image.bat D:\photos\a.jpg --detector medium --max-persons 2
```

默认 `--fit 0`（单图没有时序信息，细化意义不大）、`--detector nano`、`--device cuda`。

### 4.2 `run_video.bat` —— 本地视频

```bat
run_video.bat D:\videos\clip.mp4
run_video.bat clip.mp4 --smooth
run_video.bat clip.mp4 --focal 880                :: 手机 26mm 等效，人物距离更接近真实
run_video.bat clip.mp4 --pose-filter alpha-beta   :: 换另一种姿态滤波
run_video.bat clip.mp4 --no-viewer --save-rrd assets\demo.rrd   :: 不弹窗，只存回放
```

默认用 `medium-commercial-fit2`（存在时）+ `--fit 2` + `--pose-filter one-euro --smooth`，
也就是根关节最稳的那套组合。本机实测 720×1280 手机视频 ≈ **70 ms/帧（约 14 fps）**。

### 4.3 `run_camera.bat` —— 本地摄像头

```bat
run_camera.bat            :: 默认 0 号摄像头
run_camera.bat 1          :: 第二个摄像头
run_camera.bat 0 --detector nano --max-persons 1
```

默认 `--detector-stride 3`（每 3 帧检一次人，中间帧用 2D 关键点跟框），采集循环 ~24 fps。
窗口里按 `Ctrl+C` 停止。

### 4.4 改默认值（不用改脚本）

脚本读这三个环境变量，先设置再运行即可：

```bat
set HMR_MODEL=E:\Code\python\InstantHMR\models\instanthmr-medium.fit2.onnx
set HMR_DETECTOR=nano
set HMR_DEVICE=cuda
run_video.bat clip.mp4
```

---

## 5. 常用参数

| 参数 | 作用 |
| --- | --- |
| `--model` | 模型名（`nano`/`tiny`/`small`/`medium`/`medium-commercial`）或 `.onnx` 路径 |
| `--fit 0\|1\|2` | 运动学拟合档位，必须与 `-fitN` 图匹配 |
| `--device cuda\|cpu\|tensorrt` | 主模型后端 |
| `--detector nano\|small\|medium\|base\|large` | 人体检测器，nano 最快 |
| `--detector-stride N` | 每 N 帧检一次人（实时场景 2~3） |
| `--pose-filter one-euro\|alpha-beta` | 平滑姿态参数 |
| `--smooth` | 平滑 `cam_trans`（根位置），可叠加 |
| `--focal <像素>` | 覆盖焦距，决定世界尺度（不填按对角线 1.05× 估） |
| `--max-persons N` | 每帧最多几人（默认 4） |
| `--no-mesh` / `--no-viewer` / `--save-rrd <文件>` | 只要骨架 / 不开窗口 / 存回放 |

---

## 6. 排错

| 现象 | 原因 / 处理 |
| --- | --- |
| `[ERROR] virtual env not found` | 先跑 `setup_windows.bat` |
| `DLL load failed ... cudnn64_9.dll` | 直接用三个 bat 跑（它们已注入 nvidia 的 bin）；自己在别处跑 `python demo.py` 时缺这段 PATH |
| 检测器 det 90~120 ms（明显比主模型慢） | torch 是 CPU 版：跑 `setup_windows.bat --check` 看 `cuda: False`，或重装 cu126 wheel |
| 日志 `Registering cuda ... failed` | onnxruntime-gpu 与 torch 的 CUDA 版本不匹配，装 `onnxruntime-gpu==1.26.*` + `torch 2.9.1+cu126` |
| 根关节位置乱跳 | 确认用的是 `-fit2` 图并带 `--fit 2`；再加 `--smooth`；`--focal` 给准值 |
| 下载模型卡住 / SSL 报错 | 挂代理后重试：`setup_windows.bat --proxy http://127.0.0.1:21882` |
| 想彻底重来 | 删掉 `.venv` 再跑 `setup_windows.bat`（模型缓存在别处，不会被删） |

---

## 7. 性能参考（本机 RTX 3090）

| 场景 | 检测器 | 主模型 | 端到端 |
| --- | --- | --- | --- |
| 单图（预热后） | ~30 ms | ~15 ms（fit2 图） | — |
| 视频 465 帧 720×1280 | 34.7 ms | 35.8 ms | ~70 ms/帧（14 fps） |
| 摄像头 640×480（stride 3） | ~36 ms | — | 采集 ~24 fps |

检测器 CUDA 快路径（`instanthmr/detector.py` 里的 `_cuda_predictor()`）是 90~120 ms → 25~36 ms 的关键；
想更极端可以把它改成 `compile=True` + fp16（单帧 16 ms，但首次构建要 ~141 s，不适合每次开 bat）。
