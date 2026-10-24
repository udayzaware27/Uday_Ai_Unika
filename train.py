from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS["tensorboard"] = True

model = YOLO('yolo26m') # Pretrained backbone

model.train(
    data="construction-ppe/construction-ppe/data.yaml",
    
    epochs=100,
    imgsz=1280,#640,
    batch=7,
    workers=25,
    patience=80,
    device=0,
    amp=True,
    project="/home/aiunika-5090/Downloads/Uday/Models",
    name="constructionv1_yolo26x-10oct",
    resume=False,
    verbose=False

    # # ==== Optimization Parameters ====
    # lr0=0.0008,
    # lrf=0.1,
    # momentum=0.94,
    # weight_decay=0.0005,

    # # ==== Augmentation control ==== yolo settings tensorboard=True
    # degrees=5.0,
    # translate=0.08,
    # scale=0.25,
    # shear=0.0,
    # perspective=0.0,
    # flipud=0.0,
    # fliplr=0.5,
    # hsv_h=0.018,
    # hsv_s=0.5,
    # hsv_v=0.4,
)