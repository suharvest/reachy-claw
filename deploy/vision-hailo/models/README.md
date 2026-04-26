# vision-hailo HEF models

Both files target HAILO8 (NOT HAILO8L). If your hardware is Hailo-8L, you'll
get a perf warning but it'll still work; for native h8l, replace with the
matching `_h8l` variants.

## scrfd_2.5g.hef (face detection)

- Source: Hailo Model Zoo official pre-compiled HEF
- URL: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/scrfd_2.5g.hef
- md5: `6b061112668b2738f7f12100b9012be2`
- Input: 640×640×3 UINT8
- Outputs: 9 tensors (3 FPN scales × {bbox, score, kps})
- Reported FPS (pure inference): 1057 on Hailo-8

To re-download:
```bash
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/scrfd_2.5g.hef \
     -O scrfd_2.5g.hef
```

## hsemotion_b0.hef (emotion classification)

- Source: compiled from `enet_b0_8_best_afew.onnx` (HSEmotion EfficientNet-B0)
  via Hailo Dataflow Compiler 2025-04 (see `convert.py` script kept in WSL2)
- md5: `fb155867327611d9a38ae038899463e5`
- Input: 224×224×3 UINT8 (ImageNet normalization baked in)
- Output: 8 softmax classes — `Anger Contempt Disgust Fear Happiness Neutral Sadness Surprise`

To recompile (on x86 + Hailo SW Suite docker):
```bash
# In WSL2 → docker → /local/shared_with_docker/
python convert.py    # outputs hsemotion_b0.hef
```

## Override at runtime

```bash
HEF_DETECT=/path/to/your_scrfd.hef HEF_EMOTION=/path/to/your_emotion.hef \
    python producer.py
```
