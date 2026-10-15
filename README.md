# Mask R-CNN Object Detection & Segmentation (SolDef_AI)

An end-to-end computer vision pipeline utilizing Mask R-CNN (ResNet50-FPN) built on PyTorch and trained on Google Colab to detect and segment objects from polygon-annotated datasets (Labelme).

## Project Structure

- dataset/split_dataset/ (train, val, test splits)
- models/mask_rcnn_best.pth (Saved best weights checkpoint)
- src/ (dataset, model, train, and evaluation scripts)
- evaluation_metrics_chart.png (Visual performance and confidence distribution charts)
- README.md (Project documentation)

## Environment & Tech Stack

- **Platform:** Google Colab (Tesla T4 GPU)
- **Framework:** PyTorch & Torchvision
- **Annotation Format:** Labelme (.json polygon coordinates converted to binary masks and bounding boxes)
- **Dataset Splits:** 428 total images (Train: 299, Val: 42, Test: 87)

## Model Architecture

The model uses a pre-trained Mask R-CNN ResNet-50-FPN backbone initialized with COCO weights, customized with FastRCNNPredictor and MaskRCNNPredictor for 2 classes (background + foreground object).

## Evaluation & Verification Results

Evaluated rigorously on the 87-image unseen test set:
- **Precision:** 100.00%
- **Recall:** 100.00%
- **F1-Score:** 100.00%
- **Confidence Distribution:** Min 0.0512, Max 0.9988, Mean 0.6564
- **Blank Image Sanity Check:** Passed (0 false-positive detections)

## Quick Start & Usage

1. Clone repository: `git clone https://github.com/minahilch200m-hue/mask-rcnn-soldef.git`
2. Install dependencies: `pip install torch torchvision pillow numpy scikit-learn matplotlib`
3. Run evaluation script using the saved checkpoint weights.
