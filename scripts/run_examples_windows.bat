@echo off
REM Example Windows commands for the INDE 599 rivet project.
REM Edit these paths before running.

set SCRIPT=rivet_pipeline_multiclass_autolabel.py
set OUT=C:\Users\Will\Downloads\rivet_project_shapes
set CIRCLE=C:\Users\Will\Downloads\Photos-3-circle
set OVAL=C:\Users\Will\Downloads\Photos-3-oval
set DAMAGED=C:\Users\Will\Downloads\Photos-3-damaged

echo === Autolabel ===
python %SCRIPT% autolabel --out "%OUT%" --class_dir circle="%CIRCLE%" --class_dir oval="%OVAL%" --class_dir damaged="%DAMAGED%" --multi_pass

echo === Build dataset ===
python %SCRIPT% build --out "%OUT%" --target_train 1000 --overwrite

echo === Train model ===
python %SCRIPT% train --out "%OUT%" --epochs 80 --batch 4

echo === Evaluate test split ===
python %SCRIPT% eval --out "%OUT%" --split test --heat_thresh 0.15 --mask_thresh 0.35 --match_dist 16

echo === Run live webcam demo ===
python %SCRIPT% live --out "%OUT%" --source 0 --heat_thresh 0.15 --mask_thresh 0.35
