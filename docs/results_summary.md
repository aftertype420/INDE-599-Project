# Results Summary

This file summarizes the main evaluation numbers for the multi-class rivet detection project.

## Final Thresholds

The final validation-tuned thresholds were:

```text
heat_thresh = 0.15
mask_thresh = 0.35
match_dist = 16 pixels
```

## Validation Threshold Tuning

| Setting | TP | FP | FN | Precision | Recall | F1 | Mean Center Error | Mean IoU | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict: heat 0.30, mask 0.40, match 12 | 275 | 10 | 158 | 0.9649 | 0.6351 | 0.7660 | 3.237 px | 0.6350 | 49.11 |
| Tuned: heat 0.15, mask 0.35, match 16 | 414 | 40 | 19 | 0.9119 | 0.9561 | 0.9335 | 3.214 px | 0.6032 | 52.43 |

The tuned threshold improved recall substantially while keeping precision above 91%.

## Test Set Results

| Metric | Value |
|---|---:|
| TP / FP / FN | 386 / 47 / 20 |
| Precision | 0.8915 |
| Recall | 0.9507 |
| F1 | 0.9201 |
| Mean center error | 3.679 px at 512 x 384 |
| Median center error | 3.305 px |
| 95% center error | 7.432 px |
| Mean IoU | 0.5534 |
| Offline FPS | 32.02 |

## Per-Class Test Results

| Class | Precision | Recall | F1 | Mean Center Error | IoU |
|---|---:|---:|---:|---:|---:|
| Circle | 0.891 | 0.948 | 0.918 | 4.08 px | 0.632 |
| Oval | 0.880 | 0.959 | 0.918 | 3.32 px | 0.537 |
| Damaged | 0.906 | 0.946 | 0.926 | 3.47 px | 0.491 |

## Hough Circle Baseline

| Method | TP | FP | FN | Precision | Recall | F1 | Mean Error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hough Circle baseline | 91 | 47 | 315 | 0.6594 | 0.2241 | 0.3346 | 5.908 px |
| Multi-task U-Net | 386 | 47 | 20 | 0.8915 | 0.9507 | 0.9201 | 3.679 px |

The Hough baseline missed most non-ideal rivets and had much lower recall. The U-Net model was better suited for a multi-class rivet inspection task because it learned class-specific shape information and center heatmaps.
