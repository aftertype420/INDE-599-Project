# INDE 599 Project: Multi-Class Rivet Detection and Shape Classification

This repository contains a computer vision pipeline for detecting rivet centers and classifying each rivet as one of three shape/condition classes:

- **Circle**: normal circular rivets
- **Oval**: elongated or oval rivets
- **Damaged**: damaged or defective rivets

The project was built for the INDE 599 final project. The main goal is to move beyond simple circle detection and create a system that can localize rivets, classify their shape/condition, and run in real time from a webcam.

---

## Project Summary

Traditional circle detection methods, such as the Hough Circle Transform, work best when every object is circular and visually clean. This project is more difficult because the rivets may be circular, oval, or damaged, and the system must detect the center of each rivet while also assigning the correct class.

The final approach uses a multi-task U-Net model. The model predicts:

1. A **segmentation mask** showing where background, circle rivets, oval rivets, and damaged rivets appear in the image.
2. Three **center heatmaps**, one for each rivet class, showing likely rivet-center locations.

The segmentation output helps the model understand object shape, while the heatmap output helps it place a center point on each rivet.

---

## Repository Contents

```text
INDE-599-Project/
├── README.md
├── rivet_pipeline_multiclass_autolabel.py
├── requirements.txt
├── .gitignore
├── docs/
│   └── results_summary.md
└── scripts/
    └── run_examples_windows.bat
```

The main file is:

```text
rivet_pipeline_multiclass_autolabel.py
```

It includes commands for automatic labeling, manual label correction, dataset building, U-Net training, validation/test evaluation, a Hough Circle baseline, and live webcam inference.

---

## Installation

This project was developed and tested with Python on Windows.

```bash
git clone https://github.com/aftertype420/INDE-599-Project.git
cd INDE-599-Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For a CUDA-enabled GPU, install the correct PyTorch build for your system from the official PyTorch installation instructions. The pipeline can also run on CPU by adding the `--cpu` flag to training, evaluation, or live inference commands.

---

## Dataset Layout

The pipeline expects images to be grouped by class folder. Each folder name corresponds to the rivet class used during training.

Example Windows folders:

```text
C:\Users\Will\Downloads\Photos-3-circle
C:\Users\Will\Downloads\Photos-3-oval
C:\Users\Will\Downloads\Photos-3-damaged
```

Each image may contain multiple rivets. The folder tells the script the class of the rivets in that image set.

---

## Main Workflow

### 1. Create or review labels

The project supports automatic labeling using classical image processing. This creates initial center labels and review images.

```bash
python rivet_pipeline_multiclass_autolabel.py autolabel ^
  --out "C:\Users\Will\Downloads\rivet_project_shapes" ^
  --class_dir circle="C:\Users\Will\Downloads\Photos-3-circle" ^
  --class_dir oval="C:\Users\Will\Downloads\Photos-3-oval" ^
  --class_dir damaged="C:\Users\Will\Downloads\Photos-3-damaged" ^
  --multi_pass
```

Optional manual correction can be done with:

```bash
python rivet_pipeline_multiclass_autolabel.py make_bad_csv --out "C:\Users\Will\Downloads\rivet_project_shapes" --prefill_suspects

python rivet_pipeline_multiclass_autolabel.py label ^
  --out "C:\Users\Will\Downloads\rivet_project_shapes" ^
  --bad_csv "C:\Users\Will\Downloads\rivet_project_shapes\bad_review_list.csv"
```

### 2. Build the training, validation, and test dataset

```bash
python rivet_pipeline_multiclass_autolabel.py build ^
  --out "C:\Users\Will\Downloads\rivet_project_shapes" ^
  --target_train 1000 ^
  --overwrite
```

This step resizes images to 512 x 384, creates segmentation masks, creates center heatmaps, splits the dataset into train/validation/test sets, and augments the training set.

### 3. Train the model

```bash
python rivet_pipeline_multiclass_autolabel.py train ^
  --out "C:\Users\Will\Downloads\rivet_project_shapes" ^
  --epochs 80 ^
  --batch 4
```

The best model checkpoint is saved as:

```text
C:\Users\Will\Downloads\rivet_project_shapes\checkpoints\best_unet_shapes.pt
```

The checkpoint is intentionally not committed to GitHub because model weight files can be large.

### 4. Evaluate the model

Recommended threshold settings from validation tuning:

```bash
python rivet_pipeline_multiclass_autolabel.py eval ^
  --out "C:\Users\Will\Downloads\rivet_project_shapes" ^
  --split test ^
  --heat_thresh 0.15 ^
  --mask_thresh 0.35 ^
  --match_dist 16
```

### 5. Run the live webcam demo

This project uses the laptop/webcam input for the final demo:

```bash
python rivet_pipeline_multiclass_autolabel.py live ^
  --out "C:\Users\Will\Downloads\rivet_project_shapes" ^
  --source 0 ^
  --heat_thresh 0.15 ^
  --mask_thresh 0.35
```

During live inference, the overlay displays:

```text
C = number of circle rivets
O = number of oval rivets
D = number of damaged rivets
total = total detections
FPS = live processing speed
```

---

## Model Details

The model is a U-Net style convolutional neural network with two output heads:

- **Segmentation head**: predicts four classes: background, circle, oval, damaged.
- **Center heatmap head**: predicts three class-specific center heatmaps: circle center, oval center, damaged center.

This design lets the model learn both the shape region and the center point of each rivet. This is useful because classification alone does not tell the system where each rivet is, and center detection alone does not describe the shape or damage class.

---

## Evaluation Metrics

The project reports several metrics:

- **True Positive (TP)**: predicted rivet center matches a ground-truth rivet of the same class within the matching distance.
- **False Positive (FP)**: predicted center does not match any ground-truth rivet.
- **False Negative (FN)**: ground-truth rivet was missed.
- **Precision**: of all predicted rivets, the percentage that were correct.
- **Recall**: of all real rivets, the percentage the model found.
- **F1 score**: balance between precision and recall.
- **Mean center error**: average pixel distance between predicted and true rivet centers after resizing to 512 x 384.
- **Mean IoU**: average segmentation overlap between predicted and ground-truth masks.
- **FPS**: offline or live processing speed in frames per second.

---

## Key Results

The final tuned model was evaluated on a held-out test split using:

```text
heat_thresh = 0.15
mask_thresh = 0.35
match_dist = 16 pixels
```

### Test Set Performance

| Metric | Value |
|---|---:|
| True Positives | 386 |
| False Positives | 47 |
| False Negatives | 20 |
| Precision | 0.8915 |
| Recall | 0.9507 |
| F1 Score | 0.9201 |
| Mean Center Error | 3.679 px |
| Median Center Error | 3.305 px |
| 95% Center Error | 7.432 px |
| Mean IoU | 0.5534 |
| Offline FPS | 32.02 |

### Per-Class Test Performance

| Class | Precision | Recall | F1 | Mean Error | IoU |
|---|---:|---:|---:|---:|---:|
| Circle | 0.891 | 0.948 | 0.918 | 4.08 px | 0.632 |
| Oval | 0.880 | 0.959 | 0.918 | 3.32 px | 0.537 |
| Damaged | 0.906 | 0.946 | 0.926 | 3.47 px | 0.491 |

### Baseline Comparison

A Hough Circle baseline was used for comparison. It detected circular centers only and performed much worse on the multi-class rivet task.

| Method | Precision | Recall | F1 | False Negatives | Mean Error |
|---|---:|---:|---:|---:|---:|
| Hough Circle Baseline | 0.6594 | 0.2241 | 0.3346 | 315 | 5.908 px |
| Multi-task U-Net | 0.8915 | 0.9507 | 0.9201 | 20 | 3.679 px |

The U-Net model substantially reduced missed detections and improved overall F1 score compared with the classical baseline.

---

## Threshold Tuning Result

A stricter validation setting produced high precision but missed many rivets:

```text
heat_thresh = 0.30
mask_thresh = 0.40
match_dist = 12
Validation F1 = 0.7660
Validation recall = 0.6351
```

Lowering the heat and mask thresholds increased recall while keeping precision high:

```text
heat_thresh = 0.15
mask_thresh = 0.35
match_dist = 16
Validation F1 = 0.9335
Validation recall = 0.9561
```

This tuned setting was selected for the final test evaluation and live webcam demo.

---

## Live Demo Notes

Use this command for the final presentation demo:

```bash
python rivet_pipeline_multiclass_autolabel.py live --out "C:\Users\Will\Downloads\rivet_project_shapes" --source 0 --heat_thresh 0.15 --mask_thresh 0.35
```

In the OpenCV window:

- Yellow labels/circles indicate **circle** rivets.
- Green labels/circles indicate **oval** rivets.
- Red labels/circles indicate **damaged** rivets.
- The top-left text shows the class counts, total detections, and FPS.
- Press `q` or `Esc` to close the live demo window.

---

## Limitations and Future Work

Current limitations:

- The model depends on the quality and consistency of the training labels.
- The dataset is relatively small and uses controlled class folders.
- Damaged rivets are harder to segment cleanly because damage can vary in shape and appearance.
- Live webcam performance can change with lighting, camera angle, motion blur, and distance from the part.

Future improvements:

- Collect more images under different lighting and camera angles.
- Add more defect classes if more failure modes are available.
- Export the model for faster deployment using ONNX or TorchScript.
- Add a simple GUI for non-technical users.
- Integrate the system with an automated inspection station.

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- tqdm
- U-Net architecture
- Image segmentation
- Center heatmap detection
- Real-time webcam inference

---

## Author

William Hong  
INDE 599 Final Project  
University of Washington
