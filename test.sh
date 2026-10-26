#!/usr/bin/env bash
#
# test.sh —— rodent-act 端到端测试（所有命令写死，最直观的形式）
#
# 做的事：
#   1. 编译
#   2. 生成空场参考图  media/empty_ref.png
#   3. 生成测试视频    media/synthetic_mice.mp4
#   4. 检测 + 追踪（blob 后端）→  out/tracked.mp4
#   5. 0.5s 窗口分析（blob 后端）→  out/analyzed.mp4 + out/windows.csv
#   5b. 检测 + 追踪（YOLO 后端）→  out/yolo_tracked.mp4
#   6. 截图预览          out/preview.png
#
# 运行：  bash test.sh

cd /home/admin/workspace/rodent-act


# ============================================================ 1. 编译 ======
echo "=== 1. 编译 ==="

cargo build --release


# ================================================== 2. 生成空场参考图 ======
echo ""
echo "=== 2. 生成空场参考图 (纯背景色 0xD8D8D0) ==="

mkdir -p out media

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "color=c=0xD8D8D0:s=960x540:r=25:d=0.2" \
  -frames:v 1 media/empty_ref.png

echo "已生成 media/empty_ref.png"


# ===================================================== 3. 生成测试视频 ======
echo ""
echo "=== 3. 生成测试视频 (8 秒, 25fps, 960x540, 两只小鼠: 一动一静) ==="

analytics/.venv/bin/python tools/make_test_video.py media/synthetic_mice.mp4 8

ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of default=nw=1 media/synthetic_mice.mp4


# ================================================= 4. 检测 + 追踪 ===========
echo ""
echo "=== 4. 检测 + 追踪 → out/tracked.mp4 ==="

./target/release/rodent-act track \
  -i media/synthetic_mice.mp4 \
  --reference media/empty_ref.png \
  -o out/tracked.mp4 \
  --json out/tracks.json \
  --csv out/tracks.csv \
  --max-side 640 \
  --min-hits 5

echo "已生成 out/tracked.mp4 (带框、ID、轨迹的标注视频)"


# ============================================ 5. 0.5 秒窗口分析 =============
echo ""
echo "=== 5. 0.5s 窗口分析 → out/analyzed.mp4 + out/windows.csv ==="

./target/release/rodent-act analyze \
  -i media/synthetic_mice.mp4 \
  --reference media/empty_ref.png \
  --windows out/windows.csv \
  -o out/analyzed.mp4 \
  --max-side 640 \
  --window 0.5 \
  --min-hits 5

echo "已生成 out/analyzed.mp4 和 out/windows.csv"


# ==================================== 5b. YOLO 后端（ONNX Runtime）==========
echo ""
echo "=== 5b. YOLO 后端检测 → out/yolo_tracked.mp4 ==="
#
# 说明：这一步验证 *YOLO 推理链路* 能跑通，不代表检测质量可用。
#   - 需要 models/yolo11n.onnx + ORT 动态库（models/libonnxruntime.so）
#   - 预置 COCO 模型对合成的深色椭圆会把它们识别成 airplane/cat 之类，
#     所以这里用 --class-name any 让所有类别通过；
#   - 换成微调过的模型（analytics/model_tools.py fine-tune）后，
#     把 --class-name any 改成 --class-name mouse 即可。
#
# YOLO 后端不需要 --reference：它是语义检测，不依赖背景差。
#
# 若模型或动态库缺失，脚本会打印 SKIP 并继续，不中断后面的步骤。
#
# 换成微调过的模型时，改掉下面两行：
#   YOLO_ONNX=models/mouse.onnx
#   --class-name mouse

YOLO_ONNX=models/yolo11n.onnx

if [ ! -f "$YOLO_ONNX" ]; then
  echo "SKIP: 未找到 $YOLO_ONNX，先下载："
  echo "  curl -sL -o models/yolo11n.onnx \\"
  echo "    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"
elif [ ! -f models/libonnxruntime.so ]; then
  echo "SKIP: 未找到 models/libonnxruntime.so（ONNX Runtime 动态库）"
else
  echo "使用模型: $YOLO_ONNX"

  # 跑完整 8 秒（200 帧）。CPU 上 YOLO 约 8 fps，这一步需要 ~25 秒。
  # 想快速看链路是否通，临时改成 --max-frames 60（只处理前 2.4 秒）。
  ORT_DYLIB_PATH=models/libonnxruntime.so ./target/release/rodent-act track \
    -i media/synthetic_mice.mp4 \
    --detector yolo \
    --model "$YOLO_ONNX" \
    --class-name any \
    -o out/yolo_tracked.mp4 \
    --json out/yolo_tracks.json \
    --max-side 640 \
    --min-hits 1 \
    || echo "YOLO 步骤失败（见上面的错误）"

  echo "已生成 out/yolo_tracked.mp4 / out/yolo_tracks.json"
fi


ORT_DYLIB_PATH=models/libonnxruntime.so ./target/release/rodent-act track \
    -i TST.mp4 \
    --detector yolo \
    --model "$YOLO_ONNX" \
    --class-name any \
    -o out/my_yolo_tracked.mp4 \
    --json out/my_yolo_tracks.json \
    --max-side 640 \
    --min-hits 1


# ================================================ 6. 截一张图看看 ===========
echo ""
echo "=== 6. 从分析视频里截一帧 ==="

ffmpeg -hide_banner -loglevel error -y \
  -ss 3 -i out/analyzed.mp4 \
  -frames:v 1 out/preview.png

echo "已生成 out/preview.png"


# ==================================================== 7. 看结果 =============
echo ""
echo "=== 7. 窗口判定结果 (前 10 个窗口) ==="

cut -d, -f1-8 out/windows.csv | head -11


echo ""
echo "=== 产物 ==="
ls -lh out/


echo ""
echo "=== 下一步可以做的 ==="
echo "  播放视频:    ffplay out/analyzed.mp4        # blob 后端结果"
echo "  播放视频:    ffplay out/yolo_tracked.mp4    # YOLO 后端结果"
echo "  看整张表:    cat out/windows.csv"
echo "  看某帧截图:  xdg-open out/preview.png"
