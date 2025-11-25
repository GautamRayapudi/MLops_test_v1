from ultralytics import YOLO
model = YOLO("yolo11n.pt")

# 2. Train
results = model.train(
    data="data.yaml",  # your dataset yaml
    epochs=500,
    imgsz=640,
    lr0=0.001,
    batch=8
)
