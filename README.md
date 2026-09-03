# Wardrobe Selector — Garment Classifier: CNN from Scratch vs. Fine-tuned ViT

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

## What's next

- **Phase 2 (planned):** export both models to ONNX, deploy on a Jetson Nano, measure
  and compare inference latency before/after INT8 quantization.
- **Phase 3 (stretch):** a rule-based outfit selector on top of the classifier; no ML
  compatibility model, just structural rules by garment role and weather.
