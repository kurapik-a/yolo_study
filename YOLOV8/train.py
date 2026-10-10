from ultralytics import YOLO

def main():
    # 方式2：加载预训练权重微调
    model = YOLO("yolov8s.pt")

    model.train(
        data="score_data.yaml",
        epochs=10,
        imgsz=640,
        batch=4,
        device=0,
        workers=8,
        optimizer="SGD",
        project="runs/train",
        name="finetune_yolov8s"
    )

if __name__ == "__main__":
    main()