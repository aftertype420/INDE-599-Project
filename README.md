# AI-Based Detection of Low-Contrast Rivets (Midterm Baseline)

Classical OpenCV pipeline for low-contrast same-color rivet detection.

## Install
```bash
pip install -r requirements.txt
```

## Single image
```bash
python detect_rivets.py --image "C:/Users/Will/Desktop/INDE 599/Photos-3-001/20260503_230721.jpg" --output outputs --debug --show-coords
```

## Folder mode
```bash
python detect_rivets.py --folder "C:/Users/Will/Desktop/INDE 599/Photos-3-001" --output outputs --debug
```

## Useful tuning
- Increase `--hough-votes` to reduce false positives.
- Decrease `--hough-votes` if rivets are missed.
- Use `--select-roi` or `--roi x,y,w,h` if auto ROI is off.
