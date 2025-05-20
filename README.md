# ReWaIn-MTS

## Toward a new dataset: Mexican Traffic Signs “ReWaIn-MTS” for detection

## Daniela Bolaños-Flores, Tania Aglae Ramirez-delreal, Hamurabi Gamboa-Rosales, Guadalupe O. Gutierrez-Esparza

Various factors on the road can endanger the safety of drivers or pedestrians and cause high-impact accidents while driving. This is why traffic signs, as essential elements, provide crucial information on the condition of the road during the trip. We introduce the ReWaIn-MTS dataset, a practical tool that can support and advance research on traffic sign detection and classification in Mexico. Its applications are theoretical and have implications in the real world in autonomous conduction or assist driving. Convolutional neural networks (CNNs) have shown outstanding detection results compared to conventional methods. In this work, we used CNN-based machine learning techniques to categorize and detect Mexican traffic signs. The dataset, focused on traffic signs on the Mexican territory within the main urban roads in eight different cities, contains 2,283 road elements divided into 37 classes to train and validate algorithms. 

# Install

## Ultralytics

!pip install ultralytics --upgrade

# Download

## Dataset ReWaIn-MTS

https://drive.google.com/drive/folders/15vN2240gHq5qkaYTZNfmcGjpKzqBZh_b?usp=sharing

# Create the .yalm file 

```python
dataset_yaml = f"""
path: {base_path}

train: images/train
test: images/test
val: images/valid

nc: 37
names: [{', '.join(map(str, range(37)))}]
"""

# Save the file
with open(f"{base_path}/custom_dataset.yaml", "w") as f:
    f.write(dataset_yaml)

print(" File custom_dataset.yaml saved.")
```

# Dataset files structure

```python
dataset/
├── images/
│   ├── train/
│   ├── val/
│   ├── test/
├── labels/
│   ├── train/
│   ├── val/
│   ├── test/
├── custom_dataset.yaml
```

# YOLOv5

```python
!!git clone https://github.com/ultralytics/yolov5

%cd yolov5

!pip install -r requirements.txt
```

## Training YOLOv5

```python

from yolov5 import train, val

train.run(
    data='custom_dataset.yaml',    
    imgsz=640,
    batch=16,
    epochs=100,
    weights='yolov5s.pt',
    project='runs/train',
    name='yolo37_custom',
    exist_ok=True
)

# Validation
val.run(
    data='custom_dataset.yaml',
    weights='runs/train/yolo37_custom/weights/best.pt',
    imgsz=640,
    batch=16
)
```

## Testing YOLOv5 

```python
import torch
from yolov5 import val

results = val.run(
    data='custom_dataset.yaml',
    weights='runs/train/yolo37_custom/weights/best.pt',
    imgsz=640,
    batch_size=16,
    task='detect',
    half=False  
)

metrics_general = results[0]

mAP_0_5 = metrics_general[0]
mAP_0_5_95 = metrics_general[1]

print(f"mAP@0.5: {mAP_0_5}")
print(f"mAP@0.5:0.95: {mAP_0_5_95}")

```

# YOLOv8

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
```

## Training YOLOv8

```python
model.train(
    data='custom_dataset.yaml',
    imgsz=640,
    epochs=100,
    batch=16,
    project='runs/train',
    name='yolo8_custom',
    exist_ok=True
)

model = YOLO('runs/train/yolo8_custom/weights/best.pt')
model.val(
    data='custom_dataset.yaml',
    imgsz=640,
    batch=16
)
```

## Testing YOLOv8 

```python
model = YOLO('runs/train/yolo8_custom/weights/best.pt')

metrics = model.val(data='custom_dataset.yaml', split='test')

print(f"mAP@0.5:      {metrics.box.map50:.6f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.6f}")
```

# YOLOv11

```python
from ultralytics import YOLO

model = YOLO('yolov11n.pt')
```

## Training YOLOv11

```python
model.train(
    data='custom_dataset.yaml',
    imgsz=640,
    epochs=100,
    batch=16,
    project='runs/train',
    name='yolo11_custom',
    exist_ok=True
)

model = YOLO('runs/train/yolo11_custom/weights/best.pt')
model.val(
    data='custom_dataset.yaml',
    imgsz=640,
    batch=16
)
```

## Testing YOLOv11 

```python

model = YOLO('runs/train/yolo11_custom/weights/best.pt')

metrics = model.val(data='custom_dataset.yaml', split='test')

print(f"mAP@0.5:      {metrics.box.map50:.6f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.6f}")
```

# RTDETR

```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")
```

## Training RTDETR

```python
model.train(

    data='custom_dataset.yaml',
    imgsz=640,
    epochs=100,
    batch=16,
    project='runs/train',
    name='rtdetr',
    exist_ok=True
)
```

## Testing RTDETR 

```python

model = RTDETR('runs/train/rtdetr/weights/best.pt')

metrics = model.val(data='custom_dataset.yaml', split='test')

print(f"mAP@0.5:      {metrics.box.map50:.6f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.6f}")
```
