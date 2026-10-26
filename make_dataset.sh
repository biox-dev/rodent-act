#!/usr/bin/env bash
#
# make_dataset.sh —— 微调全过程，一步一步，命令写死，可直接逐条复制运行
#
# 目的：先跑通一次，理解每一步在做什么。跑通后再换成你自己的数据集。
#
# 全流程只有 10 步：
#   1-3  抽帧、预标注、看图
#   4    人工修正（唯一需要手动的一步）
#   5-6  划分训练/验证集、校验
#   7    训练 + 导出 ONNX（一条 model_tools.py fine-tune）
#   8-9  Rust 加载模型：单帧验证 + 跑完整视频
#   10   看结果
#
# 运行：  bash make_dataset.sh

cd /home/admin/workspace/rodent-act


# ===========================================================================
# 步骤 1：从视频抽帧
# ===========================================================================
# 抽 60 帧（约每秒 1 帧），缩放到最长边 960，存到 dataset/raw/
# 每张图会自动配一个空的 .txt（YOLO 格式要求同名标签文件）

./analytics/.venv/bin/python analytics/make_dataset.py extract \
  --video TST.mp4 \
  --out dataset/raw \
  --fps 1 \
  --frames 60 \
  --max-side 960

ls dataset/raw | head
echo "图片数: $(ls dataset/raw/*.jpg | wc -l)"


# ===========================================================================
# 步骤 2：自动预标注
# ===========================================================================
# 用"比全图中位亮度更暗"的连通域生成初始框，免去从零标注
#   --channels 6                最多 6 个目标（你的装置是 6 通道）
#   --roi 0.10,0.10,0.98,0.95   只保留中心在框内的目标（去掉灯管/线缆/手）
#
# 生成后务必人工检查！

./analytics/.venv/bin/python analytics/make_dataset.py autolabel \
  --raw dataset/raw \
  --channels 6 \
  --roi 0.10,0.10,0.98,0.95

cat dataset/raw/TST_000215.txt
echo "框总数: $(cat dataset/raw/*.txt | grep -c .)"


# ===========================================================================
# 步骤 3：可视化检查（看图！）
# ===========================================================================
# 红框 = ROI 范围，绿框 = 自动标注框
# 用眼睛确认：绿框是否套住每只鼠、有没有套到杂物

./analytics/.venv/bin/python - <<'PY'
import cv2, glob

img_path = sorted(glob.glob("dataset/raw/*.jpg"))[0]
img = cv2.imread(img_path)
h, w = img.shape[:2]

# 画 ROI（和步骤 2 的 --roi 保持一致）
x1, y1, x2, y2 = 0.10, 0.10, 0.98, 0.95
cv2.rectangle(img, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 0, 255), 2)

# 画标注框
for line in open(img_path.rsplit(".", 1)[0] + ".txt"):
    p = line.split()
    if len(p) != 5:
        continue
    _, xc, yc, bw, bh = p
    xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
    a = (int((xc - bw/2) * w), int((yc - bh/2) * h))
    b = (int((xc + bw/2) * w), int((yc + bh/2) * h))
    cv2.rectangle(img, a, b, (0, 255, 0), 2)

cv2.imwrite("out/roi_check.jpg", img)
print("看 out/roi_check.jpg：红框=ROI，绿框=标注")
PY

# 如果框不对：
#   框太少 → --roi 放大，或加 --thresh 30（更容易判为暗）
#   框太多 → --thresh 60（只取更暗的）


# ===========================================================================
# 步骤 3b：把【所有】图片的标注都画出来，存到 out/roi/
# ===========================================================================
# 一张一张看太慢，这里一次性把 dataset/raw/ 里每张图的标注都渲染成 jpg。
# 文件名和原图一致，方便对照修改。
# 画三样东西：
#   红框 = ROI 范围（超出 ROI 的目标不会被标注）
#   绿框 = 标注框本身
#   左上角文字 = 文件名 + 本帧框数（框数为 0 的帧要重点看，通常是漏检）

./analytics/.venv/bin/python - <<'PY'
import cv2, glob, os

os.makedirs("out/roi", exist_ok=True)

# ROI 和步骤 2 的 --roi 保持一致
x1, y1, x2, y2 = 0.10, 0.10, 0.98, 0.95

images = sorted(glob.glob("dataset/raw/*.jpg")) + sorted(glob.glob("dataset/raw/*.png"))
empty = 0

for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    # 画 ROI
    cv2.rectangle(img, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 0, 255), 2)

    # 画标注框
    n_box = 0
    lbl = os.path.splitext(img_path)[0] + ".txt"
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) != 5:
                continue
            _, xc, yc, bw, bh = p
            xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
            a = (int((xc - bw/2) * w), int((yc - bh/2) * h))
            b = (int((xc + bw/2) * w), int((yc + bh/2) * h))
            cv2.rectangle(img, a, b, (0, 255, 0), 2)
            n_box += 1
    if n_box == 0:
        empty += 1

    # 左上角写文件名和框数（用 "/" 分隔，避免和后缀连在一起看串）
    tag = f"{os.path.basename(img_path)}  |  boxes = {n_box}"
    cv2.putText(img, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    out_path = os.path.join("out/roi", os.path.basename(img_path))
    cv2.imwrite(out_path, img)

print(f"共导出 {len(images)} 张到 out/roi/")
print(f"其中框数为 0 的帧: {empty} 张（这些帧要重点检查是否漏标）")
PY

ls out/roi | head
echo "导出图片数: $(ls out/roi | wc -l)"


# ===========================================================================
# 步骤 4：人工修正（本脚本不自动化）
# ===========================================================================
# 用任意 YOLO 标注工具打开 dataset/raw/：
#   labelImg dataset/raw            （本地）
#   Label Studio / CVAT / Roboflow  （网页版，可用 docker 起）
# 把错的框拖正、缺的补上、多的删掉。预标注通常已经很接近。
#
# 修正完再往下走。这步产出的就是训练数据的"真值"。


# ===========================================================================
# 步骤 5：划分训练集 / 验证集，生成 data.yaml
# ===========================================================================
# 生成：
#   dataset/images/train  dataset/images/val
#   dataset/labels/train  dataset/labels/val
#   dataset/data.yaml     （Ultralytics 的训练配置）

./analytics/.venv/bin/python analytics/make_dataset.py split \
  --raw dataset/raw \
  --out dataset \
  --val-frac 0.2

cat dataset/data.yaml


# ===========================================================================
# 步骤 6：校验数据集
# ===========================================================================
# 检查缺标签、坐标越界、框数太少等问题

./analytics/.venv/bin/python analytics/make_dataset.py check --dataset dataset


# ===========================================================================
# 步骤 7：训练
# ===========================================================================
# 关键参数：
#   data        = dataset/data.yaml
#   project     = "$PWD/runs"   ← 必须是绝对路径（相对路径会被加 runs/detect/）
#   single_cls  = True          只有 1 个类别，别学 COCO 的 80 类
#   epochs 30   = 冒烟测试；正式训练改 100~200
#   imgsz 480   = 冒烟测试；正式训练改 640
#   device cpu  = 有 GPU 改 0

./analytics/.venv/bin/python - <<PY
from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")          # COCO 预训练权重作为起点
model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=480,
    batch=8,
    device="cpu",
    single_cls=True,
    project="$PWD/runs",      # ← 绝对路径，见上方说明
    name="tst",
    exist_ok=True,
    workers=0,
    verbose=True,
)
print("TRAIN_OK")
PY

# 训练产物：
#   runs/tst/weights/best.pt   最好的权重
#   runs/tst/weights/last.pt   最后一轮
#   runs/tst/results.csv       每轮指标（列 6 = mAP50）
ls -lh runs/tst/weights/
tail -1 runs/tst/results.csv


# ===========================================================================
# 步骤 8：导出 ONNX
# ===========================================================================
# Rust 端用 ONNX Runtime 推理，不认 .pt
# convert 会把类名写进 ONNX 的 metadata，Rust 端自动读取

./analytics/.venv/bin/python analytics/model_tools.py convert \
  runs/tst/weights/best.pt \
  --out models/mouse.onnx \
  --imgsz 480

# 确认类名 —— 打印 {0: 'item'} 之类的单类名字
# ⚠ 这个名字就是下面 Rust 端 --class-name 要填的值，别硬记成 "mouse"！
./analytics/.venv/bin/python analytics/model_tools.py names models/mouse.onnx


# ===========================================================================
# 步骤 9：Rust 加载模型 —— 单帧验证
# ===========================================================================
# 打印视频第 200 帧的所有候选检测，看模型有没有学到
# 同时把该帧存成 out/debug_frame.png

ORT_DYLIB_PATH=models/libonnxruntime.so \
  ./target/release/examples/debug_detect \
  TST.mp4 200 0.25 models/mouse.onnx


# ===========================================================================
# 步骤 10：Rust 加载模型 —— 跑完整视频
# ===========================================================================
# 检测 + 追踪 + 0.5s 窗口特征 + 不动/活动判定
#   --class-name item   必须等于上一步 names 打印出来的类名（这里是 item）
#   --detector yolo     走 ONNX Runtime 后端
#   （YOLO 后端不需要 --reference，它是语义检测）

ORT_DYLIB_PATH=models/libonnxruntime.so \
  ./target/release/rodent-act analyze \
  -i TST.mp4 \
  --detector yolo \
  --model models/mouse.onnx \
  --class-name item \
  --windows out/finetuned_windows.csv \
  -o out/finetuned.mp4 \
  --max-side 640 \
  --min-hits 3


# ===========================================================================
# 步骤 11：看结果
# ===========================================================================

head -5 out/finetuned_windows.csv
ls -lh out/finetuned.mp4 out/finetuned_windows.csv models/mouse.onnx

# 播放：      ffplay out/finetuned.mp4
# 看窗口表：  cat out/finetuned_windows.csv
