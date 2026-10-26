# rodent-act

小鼠悬尾实验 (tail-suspension test, TST) 视频分析 —— **Rust 实现**。

已实现：**多目标检测 → 多目标追踪 → 视频标记预览 → 0.5 s 窗口特征 → 分类**
（支持自训练 XGBoost JSON 模型，未训练时用可解释的速度阈值基线）。

---

## 1. 快速开始

```bash
# 1) 编译
cargo build --release

# 2) 下载 ONNX Runtime 动态库（仅 YOLO 后端需要）
curl -sL https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz \
  | tar xz --wildcards '*/lib/libonnxruntime.so*'
cp onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0 models/libonnxruntime.so

# 3) 看一段视频的基本信息
./target/release/rodent-act probe media/synthetic_mice.mp4

# 4) 检测 + 追踪 + 生成标记视频（默认 blob 后端，无需模型）
#    --reference 传入“空场”图（动物取走后的背景），强烈推荐
./target/release/rodent-act track \
  -i media/synthetic_mice.mp4 \
  --reference media/empty_ref.png \
  -o out/tracked.mp4 \
  --json out/tracks.json --csv out/tracks.csv \
  --min-hits 5

# 4b) 一步到位：追踪 + 0.5 s 窗口特征 + 不动/活动判定
./target/release/rodent-act analyze \
  -i media/synthetic_mice.mp4 --reference media/empty_ref.png \
  --windows out/windows.csv -o out/analyzed.mp4

# 5) 使用 YOLO 后端
ORT_DYLIB_PATH=models/libonnxruntime.so ./target/release/rodent-act track \
  -i media/mouse_cage_20s.mp4 --detector yolo --class-name any \
  -o out/yolo_tracked.mp4 --json out/yolo_tracks.json
```

`out/tracked.mp4` 是可以直接播放/预览的标注视频：每个目标一个固定颜色的
框、ID、置信度、运动轨迹，左上角 HUD 显示帧号 / 时间 / 目标数。

---

## 2. 架构

```
ffmpeg (子进程)              (Rust)
┌────────────┐   rgb24   ┌─────────────────────────────────────────┐
│ VideoReader│──────────▶│ 检测后端                                 │
└────────────┘           │  ├─ blob: 前景寄存器 + 形态学连通域      │
                         │  └─ yolo: ONNX Runtime YOLO11            │
                         └──────────────┬──────────────────────────┘
                                        │ Vec<Detection>
                         ┌──────────────▼──────────────────────────┐
                         │ MultiTracker                            │
                         │  卡尔曼(匀速)预测 → 匈牙利匹配 → 生命周期│
                         └──────────────┬──────────────────────────┘
                                        │ Vec<Track> (含历史轨迹点)
              ┌─────────────────────────┼──────────────────────────┐
              ▼                         ▼                          ▼
        draw::draw_frame          JSON / CSV 导出           (下一步) 0.5s 特征
        (Canvas, 矢量绘制)        (TrajectoryRecord)         → XGBoost
```

### 模块

| 文件 | 作用 |
|---|---|
| `src/video_io.rs` | ffmpeg 子进程解码/编码、`ffprobe` 读取元信息、letterbox 几何 |
| `src/preprocess.rs` | rgb24 → NCHW letterbox 张量；网络坐标 ↔ 原图坐标 |
| `src/detect/blob.rs` | 无模型检测：时间中值背景 + **前景寄存器**(长期静止区域也算目标) + 开闭运算 + 连通域 |
| `src/detect/motion.rs` | 轻量运动门控 / ROI 提议（YOLO 后端用于"按需检测"） |
| `src/detect/yolo.rs` | YOLO11 `[1,84,N]` 输出解码 + NMS |
| `src/detect/mod.rs` | ONNX Runtime 会话、**从模型 metadata 自动读取类名** |
| `src/track/kalman.rs` | 8 状态匀速卡尔曼滤波（`cx,cy,aspect,h,vx,vy,va,vh`） |
| `src/track/hungarian.rs` | 匈牙利算法（最短增广路，O(n³)，支持矩形代价矩阵） |
| `src/track/mod.rs` | 多目标追踪器：IoU+距离混合代价、门控、coasting、Tentative/Confirmed/Lost |
| `src/draw.rs` | 纯 RGB 缓冲区绘框/轨迹/文字（内置 5×7 点阵字体）、PNG/ASCII 预览 |
| `src/features.rs` | 0.5 s 窗口特征（骨架已就位，见第 5 节） |
| `src/classify.rs` | XGBoost 接口（骨架已就位，见第 5 节） |
| `src/pipeline.rs` | 端到端流水线 + 统计 + JSON/CSV 导出 |

---

## 3. 两个检测后端（关键设计）

### 3.1 `blob`（默认，适合固定机位 TST / 家笼）

* 每 4 px 池化成一格，用最近 25 帧的**逐格时间中值**作为静态背景；
* **前景寄存器**：一格若超过 60 帧没变化，且与长期背景不同，就判为目标
  —— 这样**完全静止的小鼠也不会被背景"吸收"**（这是纯背景减法的经典陷阱）；
* **空场参考帧**（`--reference ref.png` 或 `--reference frame:N`）：TST 里动物
  从第一帧就在画面中，时间中值背景**原理上无法**看见它。用一张"动物取走后的
  空场"图直接做差，两只（或更多）目标从第 0 帧起就全部可见 —— **生产环境推荐**；
* 3×3 开运算去噪 → 闭运算连接身体碎片（眼睛/耳朵/尾巴）→ 连通域 → 面积/长宽比/填充率过滤；
* 速度快（640×360 下 **~170 fps**），CPU 单核即可。

调参：

| 现象 | 调整 |
|---|---|
| 漏检（动物与地面太接近） | `--blob-threshold`（默认 22）调小 |
| 静止动物看不见 | 必须提供 `--reference`（空场图/帧） |
| 碎成多块 | 增大 `merge_gap`（源码 `BlobDetector::merge_gap`） |
| 误检地面纹理 | `--blob-min-area` 调大 / `--blob-threshold` 调大 |

**局限**：假设**相机固定**。若相机/视野漂移（例如手持或云台移动），
逐像素背景模型会失效 —— 请改用 `yolo` 后端，或先做视频稳像
（`ffmpeg -vf vidstabdetect/vidstabtransform`）。

### 3.2 `yolo`（适合杂乱场景 / 微调模型）

* ONNX Runtime（`ort`，`load-dynamic`，需要 `ORT_DYLIB_PATH`）；
* 运动门控 + ROI：场景不动时跳过推理，只在 ROI 上跑 CNN（`--refresh-every` 强制定期全图）；
* **类名从 ONNX metadata 自动读取**：微调后的单类模型无需额外配置，
  `--class-name mouse` 直接可用；
* 预置的 COCO `yolo11n` 对**俯视实验室白鼠几乎不触发 `mouse` 类**
  （实测最高分是 `cat/bear/bowl`），所以默认开启
  `--fallback-animal`：首帧全图检测为空时自动放宽到所有类别，并打印警告。

---

## 4. 关键算法细节

### 4.1 卡尔曼 + 匈牙利

```
预测:  x' = F x,  P' = F P Fᵀ + Q          (F = 匀速模型)
代价:  cost = 0.7·(1−IoU) + 0.3·(d / gate)  (gate = 2.5·轨迹高度)
门控:  IoU 或中心距离超出 → 1e6（禁配）
更新:  K = P Hᵀ (H P Hᵀ + R)⁻¹，约瑟夫形式保证数值稳定
```

* 类别不同的检测不会互相匹配；
* 未匹配轨迹 **coast**（仅预测）最多 `--max-age` 帧，超过才销毁；
* 新轨迹 `--min-hits` 帧后才报告（抑制闪烁误检）；
* 每条轨迹保留最近 900 个轨迹点，供特征提取使用。

### 4.2 输出格式（供特征提取）

```jsonc
{
  "input": "...", "width": 640, "height": 360, "fps": 25.0,
  "frames": 500, "duration_s": 20.0,
  "detect_calls": 500, "detections": 581,
  "tracks": [{ "id": 1, "class_id": 0, "start_frame": 20, "end_frame": 480,
               "n_points": 460, "n_measured": 430 }],
  "records": [{ "frame": 20, "track_id": 1, "x1": 12.0, "y1": 40.0,
                "x2": 180.0, "y2": 140.0, "score": 0.42, "measured": true }]
}
```

---

## 5. 下一步：0.5 s 窗口特征 + XGBoost

### 5.1 特征（`analytics/features.py` 已实现）

窗口默认 0.5 s（`fps*0.5` 帧），可 50% 重叠。距离按轨迹自身中位对角线归一化，
因此与相机距离无关：

| 特征 | 含义 |
|---|---|
| `speed_mean/std/max` | 质心速度（体长/秒） |
| `path_length` / `displacement` / `path_efficiency` | 路程、净位移、直线度 |
| `turn_abs_mean/max` | 航向变化 |
| `stationary_fraction` | 低于阈值的帧占比 |
| `micro_move_ratio` | 轻微但有动作的帧占比 |
| `bbox_area_cv` / `aspect_mean` / `aspect_std` | 体态形状变化 |
| `velocity_autocorr1` | 速度序列一阶自相关（运动规律性） |
| `n_jumps` / `valid_ratio` / `size_norm` | 数据质量与尺度 |

```bash
./analytics/.venv/bin/python analytics/extract_features.py \
  out/tracks.json out/windows.csv --fps 25 --width 640 --height 360
```

### 5.2 训练 + 推理

```bash
# 训练（按 track_id 分组交叉验证，避免同一次录制的窗口泄漏）
./analytics/.venv/bin/python analytics/train_xgb.py \
  data/windows_labeled.csv models/tst_clf.json --cv 5

# 用训练好的模型做整段视频分析
./target/release/rodent-act analyze -i trial.mp4 --reference empty.png \
  --clf models/tst_clf.json --windows out/windows.csv -o out/analyzed.mp4
```

Rust 侧自带 XGBoost JSON 推理器（`src/classify.rs`），无需 `libxgboost`，
单文件二进制即可部署；只支持二分类 logistic 目标。

`out/windows.csv` 列：`track_id, start_frame, end_frame, start_t, end_t,
immobile_prob, label, rii` + 18 个特征。`rii` 是文献里的经典判据
（0.5 s 内位移 < 1 px 的帧占比），便于与已发表结果对照。

---

## 6. 检测模型微调（推荐用于生产）

COCO 模型不认识俯视/悬尾的白鼠（实测最高分是 `cat/bear/bowl`），必须微调。
**几百帧标注**通常就够（YOLO11n 从预训练权重做单类微调，几十到一两百张也能出效果）。

### 6.1 数据准备（`analytics/make_dataset.py`）

标准 YOLO 格式，一张图一个同名 `.txt`：

```
<类 ID> <x_center> <y_center> <width> <height>     # 全部归一化到 0..1
```

四步，每步都可单独跑：

```bash
# ① 抽帧：按 1 fps 抽（长视频抽 300 帧约覆盖 5 分钟）
./analytics/.venv/bin/python analytics/make_dataset.py extract \
  --video TST.mp4 --out dataset/raw --fps 1 --frames 300 --max-side 960

# ② 预标注：用"比全图中位亮度更暗"的连通域自动生成初始框
#    --channels 是悬尾装置通道数；--roi 限制在装置区域内（重要！）
./analytics/.venv/bin/python analytics/make_dataset.py autolabel \
  --raw dataset/raw --channels 6 --roi 0.20,0.12,0.95,0.92

# ③ 人工修正：用 labelImg / Label Studio / CVAT / Roboflow 打开 dataset/raw
#    （预标注已经很接近，通常每帧只需 1-2 次点击的修正）

# ④ 划分训练/验证集并生成 data.yaml
./analytics/.venv/bin/python analytics/make_dataset.py split \
  --raw dataset/raw --out dataset --val-frac 0.2

# ⑤ 校验（缺标签、越界、框太少都会报出来）
./analytics/.venv/bin/python analytics/make_dataset.py check --dataset dataset
```

`--roi` 非常关键：不加的话，灯管、线缆、笼具、操作人员的手都会变成"鼠标"。
`--channels` 直接对应装置通道数（多通道 TST 常见 4/6/12 通道）。

### 6.2 标注要点（比工具更重要）

| 要点 | 说明 |
|---|---|
| **类别只有 1 个** | 单类 `mouse`，不要区分身体部位，模型学"整只动物" |
| **框要贴紧身体** | 含尾巴会让框忽长忽短，干扰追踪；尾巴建议单独决策或忽略 |
| **姿态要全** | 悬挂、攀爬、蜷缩、剧烈挣扎都要各标一些，否则只认静止姿态 |
| **跨视频** | 至少 2-3 个不同日期/动物的视频，只用一个视频会过拟合背景 |
| **按视频划分验证集** | 同一视频的相邻帧几乎相同，随机划分会虚高 mAP；`split` 已按文件划分 |
| **数量下限** | ≥ 300 框；多通道装置 6 通道 × 50 帧即可 |
| **难例优先** | 模糊、遮挡、贴壁、与背景同色的帧价值最高 |

### 6.3 训练 + 导出

```bash
./analytics/.venv/bin/python analytics/model_tools.py fine-tune \
  --data dataset/data.yaml \
  --base yolo11n.pt \
  --epochs 100 --imgsz 640 --batch 16 --device cpu \
  --out models/mouse.onnx

# 检查导出的类名（Rust 端会从 ONNX metadata 自动读取）
./analytics/.venv/bin/python analytics/model_tools.py names models/mouse.onnx
```

CPU 上 yolo11n + 640px + 100 epoch 约需数小时；有 GPU 时 `--device 0`。
训练日志里关注 **mAP50**（>0.9 即可用）和 **val 的 P/R**。

### 6.4 用微调后的模型跑 Rust

```bash
ORT_DYLIB_PATH=models/libonnxruntime.so ./target/release/rodent-act analyze \
  -i TST.mp4 --detector yolo --model models/mouse.onnx --class-name mouse \
  --windows out/windows.csv -o out/analyzed.mp4
```

单类模型也可用 `--class-name any`（全部类别通过），效果相同。

验证检测是否正常（打印某一帧所有候选）：

```bash
ORT_DYLIB_PATH=models/libonnxruntime.so \
  cargo run --release --example debug_detect -- TST.mp4 200 0.25 models/mouse.onnx
```

### 6.5 什么时候不用微调

* **固定机位 + 背景干净 + 动物从不动到动**：`blob` + `--reference` 空场图更快更省事；
* **多通道悬尾装置**：先试 `blob`，若通道间相互干扰或光照不均再微调；
* **相机移动 / 光照剧变**：必须用 YOLO（背景法原理失效）。

---

## 7. 开发

```bash
cargo test                 # 24 个单元测试：匈牙利/卡尔曼/追踪/NMS/blob/中值背景/特征/XGBoost
cargo run --example debug_detect -- media/x.mp4 40 0.05 [model.onnx]  # 打印该帧所有候选检测
cargo run --example debug_blob   -- media/x.mp4 40 22                 # 打印 blob 检测结果
```

调试开关：`RUST_LOG=debug`、`--show-rois`、`--preview-every 25`（终端 ASCII 预览）。

### 已知问题 / 注意

* `ort` 2.0-rc 的 `GraphOptimizationLevel::Level3` 在 ORT 1.22 动态库上会失败，
  因此固定使用 `Level1`（功能差别仅在于布局优化，不影响正确性）。
* 静态链接版 ORT 需要 glibc ≥ 2.38；本项目使用 `load-dynamic` 以支持 glibc 2.35。
* `blob` 后端要求相机固定；相机运动会使背景模型失效。
