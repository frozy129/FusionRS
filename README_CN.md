# FusionRS

本仓库是论文 **FusionRS: A Large-Scale RGB–Infrared Remote Sensing Dataset for Dual-Modal Vision–Language Foundation Models** 的官方实现与发布入口。

仓库分为五部分：

- [`dataset/`](dataset/)：数据集卡、Datasheet、Croissant 元数据、公开索引、划分、配置、审计及重建工具。
- [`code/`](code/)：数据构建、CLIP/VLM 训练、推理、评测和验证代码。
- [`results/`](results/)：检索、消融、迁移、Caption/VQA、真实热红外 VQA 与配对统计结果。
- [`models/`](models/)：模型卡、加载配置、文件大小与 SHA-256。
- [`benchmark/`](benchmark/)：固定 Caption/VQA 清单、真实热红外 VQA、标注来源、评测协议及 scorer。

GitHub 保存代码、文档、小型 manifest 和可审计结果。经过核验的 RC4 CLIP 模型包（包括可加载权重和 IR-aware caption manifest）已公开发布在 [FusionRS-CLIP v1.0.0-rc4](https://github.com/frozy129/FusionRS-CLIP/releases/tag/v1.0.0-rc4)。完整数据资产仍受上游数据许可约束，用户可通过公开索引和文档化的重建路径获取。

快速验证：

```bash
python code/dataset/scripts/verify_public_redacted_index.py \
  --index dataset/manifests/fusionrs_public_index_v3_rc4_20260727.jsonl.gz
```

详细发布边界见 [`dataset/PUBLIC_UPLOAD_ALLOWLIST_20260727.md`](dataset/PUBLIC_UPLOAD_ALLOWLIST_20260727.md)，模型哈希见 [`models/MODEL_MANIFEST.json`](models/MODEL_MANIFEST.json)。
