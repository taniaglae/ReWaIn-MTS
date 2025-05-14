# ReWaIn-MTS
Toward a new dataset: Mexican Traffic Signs “ReWaIn-MTS” for detection


Various factors on the road can endanger the safety of drivers or pedestrians and cause high-impact accidents while driving. This is why traffic signs, as essential elements, provide crucial information on the condition of the road during the trip. We introduce the ReWaIn-MTS dataset, a practical tool that can support and advance research on traffic sign detection and classification in Mexico. Its applications are theoretical and have implications in the real world in autonomous conduction or assist driving. Convolutional neural networks (CNNs) have shown outstanding detection results compared to conventional methods. In this work, we used CNN-based machine learning techniques to categorize and detect Mexican traffic signs. The dataset, focused on traffic signs on the Mexican territory within the main urban roads in eight different cities, contains 2,283 road elements divided into 37 classes to train and validate algorithms. 

# Install

## Ultralytics

!pip install ultralytics --upgrade

# Download

## Dataset ReWaIn-MTS

https://drive.google.com/drive/folders/15vN2240gHq5qkaYTZNfmcGjpKzqBZh_b?usp=sharing

## For YOLOv5

```python
!!git clone https://github.com/ultralytics/yolov5

%cd yolov5

!pip install -r requirements.txt
```

# Create the .yalm file for train the model

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

