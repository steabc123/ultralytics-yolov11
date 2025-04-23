from ultralytics import YOLO

if __name__ == '__main__':

    # 加载模型
    model = YOLO(model=r'archive/best.pt')  #yolo11n.pt
    # 进行推理 datasets/data/test/images/* ultralytics/assets/*
    model.predict(source=r'archive/test_images/image_s3r2_kiit_1.jpeg',     # source是要推理的图片路径这里使用yolo自带的图片 ultralytics/assets/bus.jpg
                  save=True,    # 是否在推理结束后保存结果
                  show=True,    # 是否在推理结束后显示结果
                  project='runs/detect',  # 结果的保存路径
                  )

