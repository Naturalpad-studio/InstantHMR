# InstantHMR 本地运行说明（本机实测通过）

本机环境：Windows + RTX 3090（24 GB）+ Python 3.12（`E:\Code\python\InstantHMR\.venv`）。
`medium-commercial` 权重已导出为 ONNX，推理走 onnxruntime CUDA，实测 2 人 16.8 ms（8.4 ms/人，fp32）。

> **关于"虚拟环境"**：`.venv` 是用系统里的 Python 3.12 创建的独立副本，所有包只装在
> `.venv\Lib\site-packages`，系统 Python（3.14/3.12/3.9）一个包都没动。
> 注意 PATH 里第一个 `python` 是豆包沙箱的 3.13 解释器（只有 numpy/opencv），
> 所以始终用 `.\.venv\Scripts\python.exe`，不要用裸 `python`。

## 0. 三个 bat 脚本（推荐，双击即可）

| 脚本 | 作用 | 用法 |
|---|---|---|
| `run_image.bat` | 图片 → Rerun 窗口 | `run_image.bat assets\test_zidane.jpg`（不带参数默认跑内置测试图） |
| `run_video.bat` | 本地视频 → Rerun 窗口 | `run_video.bat D:\videos\clip.mp4`（默认加 `--pose-filter one-euro`） |
| `run_camera.bat` | 本地摄像头实时 | `run_camera.bat`（默认 index 0 + `--detector-stride 3`） |

- 三者都已自动做两件容易忘的事：把 cuDNN/cuBLAS 的 DLL 目录加进 PATH、把 `.venv\Scripts`
  加进 PATH（Rerun viewer 的 `rerun.exe` 在那里，不加会报 "viewer not found"）。
- 模型/设备/检测器可用环境变量覆盖：`HMR_MODEL`、`HMR_DEVICE`（cuda/cpu/tensorrt）、`HMR_DETECTOR`（nano/small/medium/...）。
- 第一个参数之后可以继续追加 `demo.py` 的原生参数，会原样透传，例如
  `run_video.bat clip.mp4 --detector medium --detector-stride 3 --save-rrd out.rrd`。
- 本机实测（torch 已换 CUDA 版，检测器也走 GPU）：图片 3 人 detector ~30 ms / 模型 148 ms（首帧含预热）；
  视频 59 帧中位 detector 29 ms、模型 37 ms；真实 465 帧视频中位 detector 34.7 ms、模型 35.8 ms（整帧 ≈70 ms，约 14 fps）；
  摄像头 640×480 detector ~36 ms，采集循环跑到 ~24 fps（`--detector-stride 3`）。

## 1. 权重放在哪

`instanthmr-medium-commercial.pth` 是**训练/微调用的 PyTorch 权重**，demo 不能直接读它，
要先导出成 `.onnx`。它和 `run_config.json` 必须放在一起（导出时靠它恢复 `cliff_focal` 等设置）：

```
pytorch/medium-commercial/instanthmr-medium-commercial.pth
pytorch/medium-commercial/run_config.json          <- 从 HF 下载，缺了会导致深度方向静默出错
models/mhr-body.onnx                               <- 所有模型共用的人体图，已下载
models/instanthmr-medium-commercial.onnx           <- 由 .pth 导出，已生成
```

## 2. 环境（已装好）

```powershell
# 依赖已装在 .venv 里，无需重装。如需重建：
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision
.\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt timm onnx
```

`requirements.txt` 里的 `onnxruntime-gpu[cuda,cudnn]` 会拉约 1.5 GB 的 CUDA/cuDNN wheel。
**国内建议用清华源**（实测 2.5 MB/s，阿里云大包会中断）：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

## 3. 把 .pth 导出成 ONNX

```powershell
.\.venv\Scripts\python.exe tools\export_onnx.py `
    --ckpt "pytorch\medium-commercial\instanthmr-medium-commercial.pth" `
    --output "models\instanthmr-medium-commercial.onnx"
```

成功时会打印 `onnxruntime vs PyTorch, max abs diff: ... ~1e-6`，说明权重完整、配置匹配。
输出约 129 MB。

> **更准的做法**：HF 上直接有带 kinematic fit 的现成图（`instanthmr-medium-commercial-fit2.onnx`，79.4 MB），
> 不用 `mhr_model.pt`，也不用自己导。`--fit 2` 会把身体对到网络自己预测的 2D 关键点上，
> medium-commercial 的 EMDB-1 误差 72.2 → 68.3 mm，实测根关节深度抖动中位 18 → 12 mm，且比 fit0 更快
> （单帧 15 ms vs 29 ms）。`run_video.bat` 已默认用这张 fit2 图 + `--smooth`。
>
> 想要更高精度：`models\instanthmr-medium-fit2.onnx`（同机实测 3D→2D 重投影误差 5.5 → 3.8 px，
> 根深度最大跳变 121 → 78 mm，速度相同）。但它是**非商用授权（SAM License）**，
> 商用请继续用 `medium-commercial-fit2`。切换只需设环境变量：
> `set HMR_MODEL=models\instanthmr-medium-fit2.onnx` 再跑 `run_video.bat`。

## 4. 跑 demo

cuDNN 的子库是动态加载的，Windows 上要把 DLL 目录加进 PATH（每个新终端都要）：

```powershell
$sp = "E:\Code\python\InstantHMR\.venv\Lib\site-packages\nvidia"
$env:PATH = ((Get-ChildItem $sp -Directory | ForEach-Object { Join-Path $_.FullName "bin" } | Where-Object { Test-Path $_ }) -join ";") + ";" + $env:PATH
```

然后用导出的模型推理：

```powershell
.\.venv\Scripts\python.exe demo.py --image assets\test_zidane.jpg `
    --model models\instanthmr-medium-commercial.onnx --device cuda --detector nano
```

- `--detector nano`：第一次运行会从 repo.roboflow.com 下 366 MB 的 RF-DETR 权重到
  `%USERPROFILE%\.roboflow\models\`（国内很慢，建议用代理单线程下：见下）。
  `--detector` 可选 `nano/small/medium/base/large`，default 是 `medium`。
- `--no-viewer --save-rrd assets\demo.rrd`：不开 Rerun 窗口，只存录制（`rerun assets\demo.rrd` 回放）。
- 视频：`--video clip.mp4 --pose-filter one-euro`；摄像头：`--camera 0`。
- 只用骨架：`--no-mesh`；限制人数：`--max-persons 4`。

### 检测器权重下载（国内）

```powershell
$u = "https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth"
$f = "$env:USERPROFILE\.roboflow\models\rf-detr-nano.pth"
& curl.exe -x http://127.0.0.1:21882 -C - --retry 10 --retry-delay 5 --retry-all-errors -o $f $u
# 文件须为 366287238 字节
```

代理只能单连接（并发连接会被卡住），但单连接约 0.5 MB/s，够用。

## 5. 不依赖检测器的最小验证

自己有框时可直接调 API（这条路径不需要 RF-DETR）：

```python
import cv2
from instanthmr import InstantHMR

img = cv2.cvtColor(cv2.imread("assets/test_zidane.jpg"), cv2.COLOR_BGR2RGB)
hmr = InstantHMR("models/instanthmr-medium-commercial.onnx", device="cuda")
for p in hmr.predict(img, [[100, 190, 380, 700], [700, 165, 990, 700]]):
    p.joints_2d        # (70, 2) 像素
    p.joints_3d_cam    # (70, 3) 米，相机坐标系
    p.vertices_cam     # (18439, 3) 网格
    p.mhr_params       # (204,)
    p.shape_params     # (45,)
```

或用整条 pipeline（含检测器）：`PosePipeline("medium-commercial", fit=0, device="cuda")`。

## 6. 备注

- `--device cuda` 是 fp32 onnxruntime（8.4 ms/人）；论文里的 0.98 ms 是 fp16 + TensorRT。
  TensorRT 已随 `onnxruntime-gpu` 可用（provider 列表里可见 `TensorrtExecutionProvider`），
  但首次要现场编译引擎（几分钟），需要先 `pip install tensorrt-cu12==10.15.*`。
- `--fit 2`（默认的训练图版本）比纯网络更准（EMDB-1 68.3 vs 72.2 mm），但需要上面的 fit 图。
- 本机 `python` 默认指向一个 3.13 沙箱解释器，**不要**用它，用 `.\.venv\Scripts\python.exe`。
- **检测器已跑在 GPU 上**（2026-11-21 完成）：venv 内 torch 已换成 `2.9.1+cu126`、torchvision `0.24.1+cu126`，
  `torch.cuda.is_available()=True`（RTX 3090 / CUDA 12.6），RF-DETR 会自动把权重放到 `cuda:0`。
- 光把 torch 换成 CUDA 版，检测器仍有 **90–120 ms/帧**（rfdetr 默认 eager 路径，固定分辨率没开）。
  已在 `instanthmr/detector.py` 里加了快路径：CUDA 可用时用 `model.inference(compile=False, dtype="float32")`
  的快照（固定分辨率 + 预分配 buffer），单帧降到 **~25–35 ms**，构建只要 0.3 s；没有 CUDA 时自动回退原路径。
  （另测过 `inference(compile=True)` 的路子：fp16 可到 16 ms，但首次编译要 2–8 分钟，不适合每次启动的 bat 脚本。）
- 想进一步压到十几 ms，可选 `compile=True, dtype=torch.float16`（代价是启动多等 1–2 分钟）。
- 重装 CUDA 版 torch 的正确姿势（pip 直连阿里云的 2.5 GB wheel 会卡死，本机实测）：
  先用 curl **分块 range** 下 wheel，再本地装（`--no-index --find-links` 指向 wheel 目录）。
  注意 `torchvision 0.24.0+cu126` 硬依赖 `torch==2.9.0`，配 2.9.1 要用 **0.24.1**。
  回退到 CPU 版：`.\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision`。

