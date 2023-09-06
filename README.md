# Licence-Plate-Detection-using-TensorFlow-Lite

Convert your YOLO pretrained model to tflite (in my case final.pt) 
```
yolo export model=final.pt format=tflite
```

Predict your data using
```
yolo predict task=detect model=path_to_tflite_model imgsz=640 source='video.mp4/img.jpg to be predicted'
```

Your result will be stored in runs\detect\predict
