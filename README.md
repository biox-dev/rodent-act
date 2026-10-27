# rodent-act

小鼠悬尾实验 (tail-suspension test, TST) 视频分析 —— **Rust 实现**。

已实现：**多目标检测 → 多目标追踪 → 视频标记预览 → 0.5 s 窗口特征 → 分类**
（支持自训练 XGBoost JSON 模型，未训练时用可解释的速度阈值基线）。

---

## 0. 前置依赖与安装

### 0.1 依赖总览

| 依赖 | 是否必需 | 说明 |
|---|---|---|
| `ffmpeg` + `ffprobe` | **必需** | 解码/编码/读取元信息，通过**子进程**调用（不链接任何 C 库） |
| Rust 工具链（edition 2024） | 必需 | 需要 **rustc ≥ 1.85** |
| ONNX Runtime 动态库 | 仅 `yolo` 后端 / `serve` 默认模型 | `ort` 以 `load-dynamic` 方式加载，需 `ORT_DYLIB_PATH` |
| GPU（可选） | 仅 `yolo` 后端加速 | 需 `--features cuda` 编译 + GPU 版 ORT 动态库 + CUDA/cuDNN，见 0.4 节 |
| Node.js ≥ 18 + npm | 仅构建前端 UI | 不构建前端时二进制仍可运行，但 `serve` 没有页面 |
| Python ≥ 3.9 + `scikit-learn` | 仅重新训练 XGBoost | 推理已由 `src/classify.rs` 原生实现，**运行期不需要 Python** |

> 没有 `ffmpeg` / `ffprobe` 时，任何视频命令都会在启动阶段报
> `failed to spawn ffmpeg (is it on PATH?)` 或 `failed to run ffprobe`，项目**没有内置降级解码器**。

### 0.2 安装 ffmpeg / ffprobe

两者的安装包必须**同源同版本**（`ffprobe` 通常随 `ffmpeg` 一起提供）：

```bash
# Debian / Ubuntu
sudo apt update && sudo apt install -y ffmpeg

# Fedora / RHEL / CentOS（RHEL 需先启用 EPEL + RPM Fusion）
sudo dnf install -y ffmpeg

# Arch / Manjaro
sudo pacman -S --needed ffmpeg

# macOS（Homebrew）
brew install ffmpeg

# Windows（任选其一，安装后确认 ffmpeg.exe / ffprobe.exe 在 PATH 上）
winget install Gyan.FFmpeg
# 或 choco install ffmpeg
# 或 scoop install ffmpeg

# 校验
ffmpeg -version | head -n 1
ffprobe -version | head -n 1
```

若 `ffmpeg` 不在 PATH 上而无法修改系统环境，可在启动前临时追加目录：

```bash
export PATH="/opt/ffmpeg/bin:$PATH"
```


### 0.3 下载 ONNX Runtime（仅 YOLO 后端需要）

```bash
cd <repo>/rodent-act
mkdir -p models

# 按 CPU 架构选择包名：onnxruntime-linux-x64-1.22.0 / onnxruntime-linux-aarch64-1.22.0 /
# onnxruntime-osx-arm64-1.22.0 / onnxruntime-win-x64-1.22.0 ...
curl -sL https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz \
  | tar xz --wildcards '*/lib/libonnxruntime.so*'
cp onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0 models/libonnxruntime.so

# 每次运行前（或写入 ~/.bashrc / 服务单元 Environment=）
export ORT_DYLIB_PATH="$PWD/models/libonnxruntime.so"
```

| 平台 | 包名后缀 | 库文件名 |
|---|---|---|
| Linux x86_64 | `onnxruntime-linux-x64` | `libonnxruntime.so.1.22.0` |
| Linux aarch64 | `onnxruntime-linux-aarch64` | 同上 |
| macOS (Apple Silicon) | `onnxruntime-osx-arm64` | `libonnxruntime.1.22.0.dylib` |
| Windows x64 | `onnxruntime-win-x64` | `onnxruntime.dll` |
| **Linux x86_64 GPU** | `onnxruntime-linux-x64-gpu` | 同上，另需 CUDA 12.8+ / cuDNN 9（见 0.4 节） |

> 静态链接 ORT 需要 glibc ≥ 2.38，本项目使用 `load-dynamic`，glibc 2.35 即可。


### 0.4 GPU 加速（可选，仅 `yolo` 后端）

**仅仅把 `models/libonnxruntime.so` 换成 GPU 版是没用的** —— 还需要用
`--features cuda` 重新编译，让二进制里编进 CUDA 执行提供程序（EP），
并在建会话时显式注册。三者缺一不可：

| 条件 | 检查方式 |
|---|---|
| ① 二进制带 GPU 支持 | `rodent-act track --help` 里有 `--device`，且 `cargo build --features cuda` 编过 |
| ② ORT 动态库是 GPU 版 | `onnxruntime-linux-x64-gpu-1.22.0`，不是 `-x64-` |
| ③ 机器有驱动 + CUDA 12.8+ + cuDNN 9 | `nvidia-smi` 能列出 GPU |

```bash
# ① 用 GPU 支持编译（CPU 机器上也能编译，只是运行时会走 CPU）
cargo build --release --features cuda

# ② 下载 GPU 版 ONNX Runtime（体积比 CPU 版大得多，约 250 MB+）
curl -sL https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz \
  | tar xz --wildcards '*/lib/libonnxruntime.so*'
cp onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime.so.1.22.0 models/libonnxruntime.so

# ③ 运行：auto = 有 GPU 就用，没有就用 CPU
export ORT_DYLIB_PATH="$PWD/models/libonnxruntime.so"
./target/release/rodent-act analyze -i TST.mp4 --detector yolo --model models/mouse.onnx \
  --device auto --windows out/windows.csv
```

`--device` 取值：

| 值 | 行为 |
|---|---|
| `auto`（默认） | GPU 可用就用 GPU，否则静默用 CPU |
| `cpu` | 强制 CPU |
| `cuda` | **强制 GPU**，不可用时报错退出（不会静默退回 CPU） |

判定结果写在日志和输出里：`detector ready ... device=cuda:0`，
`track` 命令最后的 `... device=cuda:0`，以及 JSON 的 `device` 字段。
`--gpu-device N` 选择第 N 块卡。

> 用了 `--device cuda` 却报
> `could not enable the CUDA execution provider`：
> 说明二进制带 CUDA 支持，但**加载到的 ORT 库不是 GPU 版**，或驱动/cuDNN 版本不匹配。
> 反之，`--device cuda` 报 `this build has no CUDA support` 则是编译时没加 `--features cuda`。

GPU 版 ORT 硬性要求 **CUDA 12.8 或更新 + cuDNN 9.x**（cuDNN 8 与 9 不兼容）。
如果机器上装过 PyTorch，可以直接复用它的 CUDA/cuDNN 库：

```bash
export LD_LIBRARY_PATH="$(python -c 'import nvidia,os;print(os.path.dirname(nvidia.__file__))')/cudnn/lib:$LD_LIBRARY_PATH"
```

**值不值得用 GPU**：`yolo11n` + 640 px 单帧推理在 CPU 上本来就很快，瓶颈常在
ffmpeg 解码和 CPU↔GPU 拷贝上；小模型、小批量的场景 GPU 收益有限。用
`blob` 后端时 GPU 完全不起作用。

**Web UI**：`serve` 固定用 `--device auto`，启动时会把
`gpu_build` / `gpu_available` / `device` 报在 `GET /api/info` 里。


---

## 1. 快速开始

前置依赖见第 0 节（**至少需要 `ffmpeg`/`ffprobe`**）。

```bash
# 1) 编译（前端 dist 会被嵌入二进制；先构建前端可得到单文件版本）
(cd frontend && npm install && npm run build) && cargo build --release

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
export ORT_DYLIB_PATH=models/libonnxruntime.so 
./target/release/rodent-act track \
  -i media/synthetic_mice.mp4 --detector yolo --class-name any \
  -o out/yolo_tracked.mp4 --json out/yolo_tracks.json

# 6) 浏览器可视化界面（实时标注帧 + 窗口时间轴，见 2.5 节）
./target/release/rodent-act serve
```

> `frontend/dist` 已通过 `build.rs` **嵌入二进制**，`serve` 不再需要旁边的
> 前端目录，拷走一个可执行文件即可部署。

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
| `src/pipeline.rs` | 端到端流水线 + 统计 + JSON/CSV 导出 + `FrameHook` 逐帧回调 |
| `src/web/` | **Axum 服务**：REST + SSE（日志/进度/标注帧）+ 前端静态托管 |
| `src/web/assets.rs` | 编译期嵌入的前端文件表（由 `build.rs` 生成）+ 路径校验 / MIME |
| `build.rs` | 遍历 `frontend/dist` 生成 `include_bytes!` 表，缺失时只告警不失败 |
| `frontend/` | **React + Vite UI**：选片 → 跑分析 → 实时看标注帧 → 回放 + 时间轴 + 曲线 |

---

## 2.5 可视化 Web UI（Axum + React）

除了 CLI，还有一个浏览器界面：选视频 → 后台跑分析 → **实时看到标注帧** →
完成后在任意视频播放器上叠加检测框，并查看 0.5 s 窗口时间轴与特征曲线。

前端构建产物在**编译期嵌入二进制**（`build.rs` → `src/web/assets.rs`），
所以部署只需要一个 `rodent-act` 可执行文件，没有 `frontend/dist` 也能打开 UI。

```bash
# ① 编译前端（只需一次，改了前端再跑一次，然后再重新 cargo build）
cd frontend && npm install && npm run build && cd ..

# ② 编译并启动（默认 http://0.0.0.0:8080，监听所有网卡）
ORT_DYLIB_PATH=models/libonnxruntime.so cargo build --release
./target/release/rodent-act serve
#    ├── --addr 0.0.0.0:8080      绑定的地址（只在本机访问可用 127.0.0.1:8080）
#    ├── --media-dir media        视频选择器扫描的目录
#    ├── --out-dir out            job 输出（CSV / 可选 mp4）
#    └── --frontend frontend/dist 可选：改用磁盘上的构建（无需重新编译）
```

`--frontend` 只在目录内**确有 `index.html`** 时才生效，否则回退到嵌入版本：

| 情况 | 行为 |
|---|---|
| 编译时 `frontend/dist` 存在 | 嵌入，单文件即可运行 |
| 编译时缺失 + 运行时给了 `--frontend` | 从磁盘服务（日志 `serving frontend from disk`） |
| 两者都没有 | 只提供 `/api`，前端 404 |

嵌入版 `index.html` 发 `cache-control: no-cache`，`/assets/*`（文件名带哈希）
发 `immutable`；未知路径回退到 `index.html` 以支持 React Router 直接刷新。
编译日志会打印 `embedded N frontend files (X KiB)`，缺失时有 `cargo:warning`。

### 前端开发模式（热更新）

```bash
cargo run --release -- serve            # 终端 A：后端 :8080
cd frontend && npm run dev              # 终端 B：Vite :5173，/api 自动代理到 8080
```

### 页面

| 路由 | 内容 |
|---|---|
| `/` | 选择视频 + 参数（检测后端、空场参考、bob 阈值、min_hits、窗口长度、XGB 模型） |
| `/jobs` | 任务列表：状态、进度、轨迹数、不动比例、耗时（每 3 s 刷新） |
| `/jobs/:id` | 运行中：SSE 标注帧流 + 进度条 + 实时日志；完成后：视频 + 检测框叠加（Canvas）、窗口时间轴（点击跳转）、`p(immobile)` 曲线、任意特征曲线、每条轨迹统计、窗口明细表 |

### API

所有接口在 `/api` 下，SSE 端点用 `EventSource` 订阅：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/info` | 版本、目录、ffmpeg 可用性、特征名 |
| `GET` | `/media` | 媒体目录中的视频/图片 |
| `GET` | `/media/{name}` | 支持 `Range` 的视频流（`<video>` 拖动所需） |
| `POST` | `/jobs` | 创建任务 `{"media": "...", "detector": "blob", "reference": "empty_ref.png", ...}` |
| `GET` | `/jobs` / `/jobs/{id}` | 任务列表 / 详情（含 summary） |
| `POST` | `/jobs/{id}/cancel` | 取消运行中的任务 |
| `GET` | `/jobs/{id}/events` | **SSE**：`log` / `progress` / `status` / `done` |
| `GET` | `/jobs/{id}/frames` | **SSE**：标注帧，`data` 为 base64 JPEG |
| `GET` | `/jobs/{id}/frame` | 最新一帧 JPEG（`204` = 还没有帧） |
| `GET` | `/jobs/{id}/records` | 每帧每个轨迹框（用于播放器叠加） |
| `GET` | `/jobs/{id}/windows` | 窗口特征 + 分类结果 |
| `GET` | `/jobs/{id}/logs` | 日志快照（`?since=N` 增量拉取） |
| `GET` | `/jobs/{id}/video` | 可选导出的标注 mp4 |

实现要点：

* 任务跑在 `spawn_blocking` 线程里，通过 `pipeline::FrameHook` 每帧回调；
  JPEG 编码按 15 fps 节流，且**只有浏览器连着帧流时才编码**（`FrameBus` 统计订阅者）。
* 帧订阅者计数必须从 **SSE body 的 stream 内部**增加 —— axum 的 handler 在
  把 body 交给 hyper 后就被 drop 了，在 handler 里加计数会立刻归零。
* 日志复用现有 `tracing`：`SseLayer` 把事件转发到 broadcast channel，
  同时写入每个 job 的日志缓冲区（刷新页面也能看到历史）。
* 若代理缓冲了 SSE，前端会退化为 REST 轮询任务状态，功能不受影响。

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
* **可选 CUDA 加速**：`--features cuda` 编译 + GPU 版 ORT 库，
  `--device auto|cpu|cuda` 选择（见 0.4 节），日志/输出里会写 `device=cuda:0`；
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
cargo test                 # 单元测试：匈牙利/卡尔曼/追踪/NMS/blob/中值背景/特征/XGBoost/嵌入静态资源/设备解析
cargo test --features cuda # 同上，额外编译 CUDA 路径（无 GPU 也能跑）
cargo run --example debug_detect -- media/x.mp4 40 0.05 [model.onnx]  # 打印该帧所有候选检测
cargo run --example debug_blob   -- media/x.mp4 40 22                 # 打印 blob 检测结果
```

改了前端后必须**重新 `cargo build`** 才会进二进制（`build.rs` 监听
`frontend/dist` 的变更）；日常调 UI 可用 `--frontend` 或 Vite dev server 热更新。

调试开关：`RUST_LOG=debug`、`--show-rois`、`--preview-every 25`（终端 ASCII 预览）。

### 已知问题 / 注意

* `ort` 2.0-rc 的 `GraphOptimizationLevel::Level3` 在 ORT 1.22 动态库上会失败，
  因此固定使用 `Level1`（功能差别仅在于布局优化，不影响正确性）。
* 静态链接版 ORT 需要 glibc ≥ 2.38；本项目使用 `load-dynamic` 以支持 glibc 2.35。
* `blob` 后端要求相机固定；相机运动会使背景模型失效。
* GPU 构建（`--features cuda`）的二进制在无 GPU 机器上仍可用（`--device auto` 走 CPU）；
  但 `--device cuda` 在没有 GPU 版 ORT 库时会直接报错——这是有意设计的，避免
  “以为在用 GPU，实际跑的是 CPU”。
