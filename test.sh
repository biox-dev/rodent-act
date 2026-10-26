#!/usr/bin/env bash
#
# test.sh —— rodent-act 端到端测试（所有命令写死，最直观的形式）
#
# 做的事：
#   1. 编译
#   2. 生成空场参考图  media/empty_ref.png
#   3. 生成测试视频    media/synthetic_mice.mp4
#   4. 检测 + 追踪  →  out/tracked.mp4
#   5. 0.5s 窗口分析 →  out/analyzed.mp4 + out/windows.csv
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
echo "  播放视频:    ffplay out/analyzed.mp4"
echo "  看整张表:    cat out/windows.csv"
echo "  看某帧截图:  xdg-open out/preview.png"
