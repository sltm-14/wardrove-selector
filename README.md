# Garment Classifier: CNN from Scratch vs. Fine-tuned ViT

Real-time garment classifier built to compare two fundamentally different approaches to
image classification: a **CNN designed and trained from scratch** (no transfer learning)
against a **Vision Transformer fine-tuned from a pretrained checkpoint**. Same dataset,
same 14 categories, same train/val/test split.

## Results

|               | CNN (from scratch) | ViT (fine-tuned)        |
| ------------- | ------------------ | ----------------------- |
| Parameters    | 397,198             | 85,809,422 (~216x more) |
| Test accuracy | 0.86                | 0.90–0.91               |
| Test macro-F1 | 0.77–0.78           | 0.85–0.88               |
| Training time | ~90 min (CPU)       | ~1 hr (Colab T4 GPU)    |

**Why macro-F1, not just accuracy:** the dataset has a ~55:1 imbalance between the
largest and smallest classes. Accuracy is dominated by the large classes and hides poor
performance on the small ones. Macro-F1 treats every class equally, which is a more
honest signal here.

**Where each model struggles:** both models' weakest classes are Tops, Sweatshirts,
Sweaters, and Jackets; visually similar upper-body garments that overlap significantly
in the confusion matrix. The ViT handles this ambiguity noticeably better than the CNN
(e.g. Sweaters F1: 0.56 → 0.72–0.78), but doesn't eliminate it. Some of that confusion
reflects genuine ambiguity in the category taxonomy itself, not a model failure (see
*Qualitative testing* below).

## Approach

### CNN (from scratch, no transfer learning)

5 blocks of `Conv2d(padding='same') → BatchNorm2d → ReLU → MaxPool2d`, channels doubling
each block (16→32→64→128→256), followed by Global Average Pooling, Dropout, and a linear
classifier. Written as an `nn.Module` subclass (`src/model.py`).

Key design decisions:

- **`padding='same'` on every conv**: separates feature extraction (conv, size-preserving)
  from downsampling (pooling, explicit); each layer has one responsibility.
- **Channels double each time spatial resolution halves**: compensates for the information
  lost when pooling shrinks the spatial map.
- **Global Average Pooling instead of a large fully-connected layer**: flattening the
  final 7×7×256 feature map directly would need a ~3.2M-parameter FC layer; more
  parameters than training images (~16k). GAP controls model size given a
  small/imbalanced dataset.
- **Dropout(0.3) + weight_decay(1e-4)**: added after the first unregularized run showed
  clear overfitting (train loss 0.229 vs. best val loss 0.530, a 0.30 gap). With
  regularization, the gap dropped to ~0.07–0.09.
- **Checkpointing on best validation loss, not final epoch**: validation loss never
  settles into a flat plateau (small val set, no LR scheduler), so the last epoch isn't
  necessarily the best one; checkpointing catches the true best epoch regardless.

### ViT (transfer learning)

`google/vit-base-patch16-224` (HuggingFace `transformers`), backbone frozen, only the
new 14-class classification head is trained (~10k trainable parameters out of 86M
total).

Key design decisions:

- **Backbone frozen**: fine-tuning all 86M parameters on ~16k training images would risk
  severe overfitting. Freezing keeps the trainable surface small, relying on the
  pretrained backbone's already-general visual features.
- **Same loss/optimizer/checkpointing strategy as the CNN**, for a fair comparison.
- **Trained on Google Colab (GPU)**: a ViT-Base forward pass is heavy; early CPU testing
  (a comparable-sized model) took multiple hours per epoch. Locally on CPU, this
  training run would not have been practical.

## Qualitative testing (own photos)

Both models were also tested informally on real, non-dataset photos (not a replacement
for the formal test set evaluation). This surfaced two real bugs before it surfaced
anything about the models:

1. The ViT's prediction function was using the CNN's plain resize/tensor transform
   instead of the HuggingFace `processor` the ViT was actually trained with. That was a
   preprocessing bug, not a model problem; it produced nonsense predictions (e.g. a
   t-shirt classified as shoes) until fixed.
2. Loading a GPU-trained checkpoint on a CPU-only session without `map_location`
   fails outright; an easy one-line fix once diagnosed.

Once fixed, the remaining disagreements were mostly the same Tshirt/Top ambiguity
already visible in the formal confusion matrix. This is evidence that the model's
"errors" reflect a real taxonomy ambiguity, not a new failure mode.

## Edge Deployment (NVIDIA Jetson Orin Nano)

The CNN (from scratch) was exported to ONNX and deployed on a Jetson Orin Nano
(JetPack 6.2, CUDA 12.6) to get real, measured evidence of edge inference, not
just a plan.

### Results

| Metric | Value |
| --- | --- |
| Export format | ONNX, opset 18 (requested 17; exporter fell back to 18 after an internal downgrade step failed, correct result either way) |
| ONNX vs. PyTorch prediction agreement | 70/70 (100%) |
| Max logit difference (ONNX vs. PyTorch) | 0.004997 |
| Inference latency (ONNX Runtime, CUDA EP) | 3.236 ms avg (±0.659 ms std; 100 runs, 50 warm-up) |
| Throughput (CUDA EP) | ~309 images/sec |
| Inference latency (TensorRT EP) | 1.922 ms avg (±0.080 ms std; 100 runs, 50 warm-up) |
| Throughput (TensorRT EP) | ~520 images/sec (**~1.7x faster than CUDA EP**) |
| TensorRT engine build time | 24.6 ms (included in the first, untimed run) |
| Power draw, idle (`VDD_IN`) | ~3.7 W |
| Power draw, under load (`VDD_IN`) | ~4.3 W avg, ~5.1 W peak (~+580 mW over idle) |
| Temperature, idle → under load | ~45-47°C → ~47.6°C max (no throttling observed) |

Power/thermal measured with `tegrastats` (built into JetPack). Two honest caveats:
`GR3D_FREQ` (GPU utilization) reads near 0% in most samples because each
inference takes ~1.9ms while `tegrastats` samples every 200ms — most samples
land between inferences, not during one. Also, part of the "load" window
includes Python process startup/import overhead (the benchmark script was
re-invoked 5 times), not purely isolated inference power.

Validated on a random sample of 70 images (5 per category, drawn from the full
labeled dataset, not exclusively the held-out `test_df`) — enough to confirm
the export preserves model behavior exactly; not intended as a formal
test-set re-evaluation (hence accuracy on this sample, 75.7% for both
PyTorch and ONNX, isn't directly comparable to the 86% reported above).

### Setup problems and fixes

Getting PyTorch running on the Jetson surfaced three separate environment
issues, each real and worth documenting:

1. **Version pinning backfired.** Constraining `torch<2.8` to dodge a
   missing-library error made pip silently fall back to a generic CUDA 13
   build incompatible with the Jetson's driver (`torch.cuda.is_available()`
   returned `False`). Fixed by reinstalling the correct Jetson-specific build
   (torch 2.11.0) from the `jetson-ai-lab` index without a version constraint.
2. **Missing `libcudss.so.0`.** Recent PyTorch builds for Jetson depend on
   cuDSS (CUDA Sparse Solver), which JetPack doesn't ship. Fixed by installing
   NVIDIA's cuDSS archive directly into `/usr/local/cuda`.
3. **pandas/numpy ABI mismatch.** Installing PyTorch pulled a newer numpy
   (2.2.6) into user site-packages, breaking the system's apt-installed pandas
   (built against an older numpy ABI). Fixed with `pip install --upgrade pandas`.

None of these are specific to this project — they're the standard friction of
running less-common ML frameworks on ARM64/edge hardware, exactly the kind of
debugging this deployment was meant to surface.

### What's next

- Phase 3 (stretch): rule-based outfit selector.

## Repository structure

```
notebooks/
  01_eda.ipynb                          Dataset exploration, category selection
  02_data_pipeline.ipynb                Dataset/DataLoader prototyping
  03A_cnn_from_scratch_class.ipynb      CNN: design, training, evaluation
  03B_cnn_inference.ipynb               CNN: inference on real photos
  04A_vit_finetuning_colab.ipynb        ViT: fine-tuning, evaluation (Colab)
  04B_vit_inference_colab.ipynb         ViT: inference on real photos (Colab)
src/
  data.py                               Dataset classes, get_dataloaders()
  model.py                              GarmentCNN (nn.Module)
  download_dataset.py                   Kaggle dataset download
```

Each notebook starts with an environment setup cell (`RUNTIME`, `DEVICE`); set those two
values and run the whole notebook top to bottom. It resolves paths and device
automatically for either a local machine or Google Colab.

## Dataset

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
(Kaggle, `paramaggarwal`): ~44k product images with metadata. Filtered down to 14
`articleType` categories most relevant to a wardrobe/outfit use case, ~20,400 images
after cleaning (removing rows with missing or corrupted images).
