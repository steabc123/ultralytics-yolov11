# from ultralytics.models import YOLO
# import os
# import multiprocessing
# # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
# if __name__ == 'main':
#     multiprocessing.freeze_support()
#     model = YOLO(model='yolo11n.pt')
#     model.train(
#         data='datasets/data/data.yaml',
#         epochs=50,  # 根据数据集适当调整
#         batch=8,  # 优化显存利用
#         device='0',  # 多 GPU 时可改为 '0,1'
#         imgsz=640,  # 保持默认
#         workers=16,  # 根据 CPU 核心数优化
#         cache=False,  # 数据集小且内存足够时开启
#         amp=True,  # 混合精度
#         mosaic=False,  # 数据增强,建议开启
#         project='run/train',
#         name='exp'
#     )

# from ultralytics import YOLO
#
# if __name__ == 'main':
#     # 加载模型
#     model = YOLO("yolo11n.pt")
#
# # 训练模型
#
#     train_results = model.train(
#         data="data/data.yaml",  # 数据集 YAML 路径
#         epochs=10,  # 训练轮次
#         imgsz=640,  # 训练图像尺寸
#         device="cuda:0",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
# )

# 评估模型在验证集上的性能
#     metrics = model.val()

# 在图像上执行对象检测
#     results = model("path/to/image.jpg")
#     results[0].show()

# 将模型导出为 ONNX 格式
# path = model.export(format="onnx")  # 返回导出模型的路径

from ultralytics import YOLO
import multiprocessing
#  C:\Users\Steven\AppData\Roaming\Ultralytics 设置
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    #
    # # Train the model
    # results = model.train(data="datasets/data/data.yaml", epochs=50, imgsz=640, workers=1,
    #                       batch=8, device="cuda:0", cache=False, amp=True, mosaic=False)

    # model = YOLO(model=r'ultralytics/cfg/models/11/yolo11n.yaml')
    # 指定用于训练的模型文件。接受指向 .pt 预训练模型或 .yaml 配置文件。对于定义模型结构或初始化权重至关重要
    model = YOLO("yolo11n.pt")
    results = model.train(
        # 决定是否从预处理模型开始训练。可以是布尔值，也可以是加载权重的特定模型的字符串路径。提高训练效率和模型性能
        pretrained=r'runs/detect/train2/weights/best.pt',  # yolo11n.pt
        data=r'datasets/data/data.yaml',
        # 数据集配置文件的路径（例如 coco8.yaml).该文件包含特定于数据集的参数，包括训练和 验证数据类名和类数。
        epochs=50,  # 训练历元总数。每个历元代表对整个数据集进行一次完整的训练。调整该值会影响训练时间和模型性能。
        imgsz=640,  # 用于训练的目标图像尺寸。所有图像在输入模型前都会被调整到这一尺寸。影响模型精度和计算复杂度。
        workers=1,  # 加载数据的工作线程数（每 RANK 如果多GPU 训练）。影响数据预处理和输入模型的速度，尤其适用于多GPU 设置。
        batch=8,   # 批量大小有三种模式： 设置为整数（如 batch=16）、自动模式，内存利用率为 60%GPU (batch=-1），或指定利用率的自动模式 (batch=0.70).
        device="cuda:0",  # 指定用于训练的计算设备：单个GPU (device=0）、多个 GPU (device=0,1）、CPU (device=cpu)
                          # 或MPS for Apple silicon (device=mps).
        cache=False,  # 在内存中缓存数据集图像 (True/ram）、磁盘 (disk），或禁用它 (False).通过减少磁盘 I/O，提高训练速度，但代价是增加内存使用量。
        amp=True,  # 启用自动混合精度(AMP) 训练，可减少内存使用量并加快训练速度，同时将对精度的影响降至最低。
        mosaic=False,  # multi_scale	bool	False 在训练完成前禁用最后 N 个历元的马赛克数据增强以稳定训练。设置为 0 则禁用此功能。
        #  box	float	7.5	损失函数中边框损失部分的权重，影响对准确预测边框坐标的重视程度。
        # cls	float	0.5	分类损失在总损失函数中的权重，影响正确分类预测相对于其他部分的重要性。
        # dfl	float	1.5	分布焦点损失权重，在某些YOLO 版本中用于精细分类。
        # pose	float	12.0	姿态损失在姿态估计模型中的权重，影响着准确预测姿态关键点的重点
        # patience	int	100	在验证指标没有改善的情况下，提前停止训练所需的历元数。当性能趋于平稳时停止训练，有助于防止过度拟合。
    )
