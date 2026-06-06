from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="dataset/images",
    save=True,
    show=True,   # Opens result windows
    conf=.9
)