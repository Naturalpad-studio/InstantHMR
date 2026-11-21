# InstantHMR → iPhone（CoreML）可行性检查

实测时间：2026-11-21，在 Linux 云主机上做（coremltools 9.0 + torch 2.7.1 CPU）。
结论：**能转，但不是"一键转换"，需要两处改写；真机验证必须在 macOS 上做。**

## 1. 为什么不能走 ONNX

- `coremltools >= 7` **已经没有 ONNX 前端**（9.0 里 `coremltools.converters` 只剩 torch/tensorflow/sklearn/xgboost/libsvm）。
  也就是说 `instanthmr-*.onnx` 无法直接转成 CoreML，只能从 **PyTorch checkpoint** 走。
- Windows 也没有 coremltools 的 wheel（只有 macOS / Linux），所以转换只能在 Linux 或 macOS 上跑。

## 2. PyTorch 直接 trace 的两处阻塞（已解决）

| 现象 | 原因 | 处理 |
|---|---|---|
| `NotImplementedError: op '_native_multi_head_attention' not implemented` | torch 把 `nn.MultiheadAttention` 融合成一个算子 | trace 前 `torch.backends.mha.set_fastpath_enabled(False)` |
| `TypeError: only 0-dimensional arrays can be converted to Python scalars`（`int` op，位于 `self_attn`） | attention 里 `bsz * num_heads` 之类由 `aten::size` 派生，trace 成 `aten::Int` 节点 | 把 MHA 换成 `PlainAttn`（纯 matmul/softmax/unflatten/flatten），graph 里 `aten::Int` 归零 |

`tools/coreml/convert_to_coreml.py` 里就是这套改法，转换时它会打印 `trace diff` 和 `aten::Int` 计数做自检。

## 3. 实测结果（medium-commercial 主干图）

- 输入 `image (1,3,288,288) fp32` + `cliff_cond (1,3)`；输出 `mhr_params / shape_params / cam_trans / joints_2d`
- fp16：**62 MB**，fp32：124 MB；MIL 程序 2788 个算子，**没有自定义算子**
  （const/mul/add/cast/conv/linear/softmax/gather/slice_by_index/reshape…），可整体交给 ANE 调度
- 未验证项：CoreML 与 PyTorch 的数值一致性、真机延迟 —— 只能在 Mac 上跑 `MLModel.predict` 验证

## 4. 在 iPhone 上还缺什么

1. **人体图**（`mhr-body.onnx`，12.7 MB，18439 顶点）从 `MHRBody.from_rig(training/checkpoints/mhr_model.pt)` 构造后同样转 CoreML；
   注意 `mhr_model.pt` 需要从 MHR release 的 assets.zip 里取（约 700 MB），这一步本次未验证。
2. **带 fit 的图**（`FitNetwork`）要额外传 `training/kinematic_fit.py` 的 `MHRKinematics`，同样依赖上面的 `.pt`；
   图内是 fp16 的 LM 一步求解，CoreML 上能否保持精度要单独测。iOS 上建议先用 `fit 1` 省算力。
3. **检测器**换成 Vision 的 `VNDetectHumanRectanglesRequest`（或把 RF-DETR 另转一个 CoreML 检测头）。
4. **预处理**：Python 里的 crop/resize 与 `cliff_cond = (atan(dx/f), atan(dy/f), box_side/f)` 要在 Swift/Metal 里重写；
   `f` 用 `AVCameraCalibrationData` 的真实内参，比桌面端的 1.05×对角线假设更准（根位置会受益）。
5. **授权**：`nano/tiny/small/medium` 全是 SAM License（非商用），只有 `medium-commercial` 可商用。

## 5. 想继续的话

- 在 Mac 上：`xcrun coremlcompiler compile InstantHMRMediumCommercial.mlpackage out/` 编译，再用 Xcode 跑一次
  单帧数值比对（同一张裁剪图，PyTorch vs CoreML，看 `mhr_params` 的最大绝对误差）。
- 需要我做的下一步：把人体图 + fit 图的 `.mlpackage` 也转出来（要先下 `mhr_model.pt`），
  或把已转好的 mlpackage 打包上传给你。
