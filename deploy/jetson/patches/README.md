# Patched sherpa-onnx for Paraformer streaming EOF fix

Pre-built aarch64 shared libraries for sherpa-onnx 1.12.28+cuda with a patch
that fixes Paraformer streaming ASR tail truncation.

## What the patch fixes

The stock sherpa-onnx `OnlineRecognizer` for Paraformer drops the last 1-3
characters because `IsReady()` requires a full 61-frame chunk. When
`input_finished()` is called with fewer remaining frames, they are never
decoded.

### Changes (in `online-recognizer-paraformer-impl.h`):

1. **IsReady()**: Returns `true` after `InputFinished()` if there are any
   remaining unprocessed frames (even < chunk_size).
2. **DecodeStream()**: Zero-pads the final partial chunk to `chunk_size_` so
   the encoder receives the expected tensor shape.
3. **CIF force-fire**: When processing the last chunk, if accumulated CIF
   energy > 0.5, forces emission of the final token.

## Build environment

- Base image: `dustynv/onnxruntime:1.20-r36.4.0`
- Platform: aarch64 (Jetson Orin NX, JetPack 6.2)
- CUDA: 12.6
- onnxruntime: 1.20.0 (system, with CUDAExecutionProvider)
- sherpa-onnx: v1.12.28 (git tag)
- Python: 3.10

## To rebuild from source

```bash
cd /opt/sherpa-onnx
# Apply patch
python3 /tmp/apply_patch.py  # or manually edit the .h file
# Build
mkdir -p build-cuda && cd build-cuda
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_GPU=ON -DSHERPA_ONNX_ENABLE_PYTHON=ON \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_BINARY=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DPYTHON_EXECUTABLE=/usr/bin/python3 ..
make -j4
```

Output: `build-cuda/lib/_sherpa_onnx.cpython-310-aarch64-linux-gnu.so`
