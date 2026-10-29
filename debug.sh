#!/usr/bin/env bash
#
# debug.sh —— 以 debug 构建启动 Web 服务（方便打断点 / 单步）
#
# 用法：
#   bash debug.sh                          # 默认 0.0.0.0:8080
#   bash debug.sh --addr 127.0.0.1:9000    # 其他参数原样传给 `serve`
#
# `ort` 是以 load-dynamic 方式链接的：ONNX Runtime 在**运行时**才加载，
# 必须先用 ORT_DYLIB_PATH 指向 libonnxruntime.so，否则 yolo 后端和
# `serve` 的默认模型都会报 `failed to load dynamic library`。
#
# 注意 $PWD 只有在仓库根目录才等于仓库路径，所以先 cd 到脚本所在目录。

set -euo pipefail
cd "$(dirname "$0")"

export ORT_DYLIB_PATH="$PWD/models/libonnxruntime.so"
export RUST_LOG="${RUST_LOG:-debug,ort=warn}"

if [ ! -f "$ORT_DYLIB_PATH" ]; then
  echo "警告：$ORT_DYLIB_PATH 不存在，YOLO 后端将无法加载 ONNX Runtime" >&2
  echo "      可解压 onnxruntime-linux-x64-* ，把 lib/libonnxruntime.so* 拷到 models/" >&2
fi

echo "ORT_DYLIB_PATH=$ORT_DYLIB_PATH"
echo "RUST_LOG=$RUST_LOG"

exec cargo run --bin=rodent-act -- serve "$@"
