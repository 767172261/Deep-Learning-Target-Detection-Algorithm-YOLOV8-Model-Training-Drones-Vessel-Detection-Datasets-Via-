深度学习目标检测算法YOLOV8模型训练无人机船只检测数据集通过训练的模型权重推理识别检测 船舶 船只 和人的检测
@[toc]


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/12794759b6bc4fa6a805c98ae5c5a85e.png)

### **航空无人机船只检测数据集**

#### **数据集概述**

| 项目 | 内容 |
|------|------|
| **数据集名称** | 航空船只检测数据集 |
| **图像来源** | 无人机航拍图像（手动收集） |
| **总图像数量** | 10,252 张 |
| **应用场景** | 海洋监测、非法船只识别、人员落水检测、海域安全监控 |
| **目标类别** | `Person`（人员）、`Boat`（中小型船只）、`Ship`（大型船舶/商船） |
| **数据划分** | 训练集：8,203 张<br>验证集：1,024 张<br>测试集：1,025 张 |
| **图像特点** | 大面积广角镜头，用于检测进入感兴趣区域的可疑或非法目标 |

---

#### **按数据集划分的标注统计表**

| 数据集类型 | 图像数量 | Person（标注框数） | Boat（标注框数） | Ship（标注框数） | 总标注框数 |
|-----------|----------|---------------------|------------------|------------------|-------------|
| **训练集** | 8,203    | 219                 | 48,550           | 920              | **49,689**  |
| **验证集** | 1,024    | 7                   | 5,890            | 143              | **6,040**   |
| **测试集** | 1,025    | 13                  | 5,247            | 109              | **5,369**   |
| **总计**   | **10,252** | **239**           | **59,687**       | **1,172**        | **61,098**  |

---

#### **类别分布汇总**

| 类别   | 中文含义     | 总标注框数 | 出现图像数估算（粗略） | 主要分布场景 |
|--------|--------------|------------|------------------------|----------------|
| Person | 人员         | 239        | < 100 张               | 可能用于落水人员或甲板活动检测 |
| Boat   | 中小型船只   | 59,687     | 广泛分布               | 常见于近海、渔区、非法捕捞监测 |
| Ship   | 大型船舶/商船 | 1,172      | 数百张                 | 远海航线、港口周边、交通监控 |

---

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1e6baba19670451d821af31709b61f5b.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7ba5359284c34a219dee04c1b3b112f3.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/70b91a84aacc4cdc8d3129e49fe81c38.png)
---

## 🛠️ 一、环境搭建（支持 GPU 加速）

### 1. 安装 NVIDIA 驱动和 CUDA

```bash
# 检查 GPU 是否可用
nvidia-smi
```

确保输出中显示的 GPU 型号和 CUDA 版本（如 12.4）。若未安装，请前往：
👉 [NVIDIA 官网下载驱动](https://www.nvidia.com/Download/index.aspx)

---

### 2. 安装 Anaconda

下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

安装后打开 **Anaconda Prompt**（Windows）或终端（Linux/macOS）

---

### 3. 创建 Python 虚拟环境

```bash
# 创建名为 maritime_yolo 的环境
conda create -n maritime_yolo python=3.10

# 激活环境
conda activate maritime_yolo
```

---

### 4. 安装 PyTorch（支持 GPU）

根据的 CUDA 版本选择安装命令，例如 CUDA 11.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

验证是否成功：

```python
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

---

### 5. 安装 YOLOv8 和其他依赖

```bash
# 安装 Ultralytics YOLOv8（官方最新版）
pip install ultralytics

# 安装 OpenCV 和可视化工具
pip install opencv-python opencv-python-headless matplotlib seaborn numpy pandas tqdm

# 日志与评估
pip install tensorboard scikit-learn
```

验证安装：

```bash
yolo version
# 输出应为：8.1.x 或更高
```

---

## 📁 二、数据集结构与 `data.yaml` 配置

### 文件结构建议

```
maritime_dataset/
├── images/
│   ├── train/        # 8203 张
│   ├── val/          # 1024 张
│   └── test/         # 1025 张
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

> ✅ 确保每张图像有对应的 `.txt` 标签文件（YOLO格式）：`class_id x_center y_center width height`（归一化坐标）

---

### ✅ `data.yaml` 内容

```yaml
# data.yaml
train: ./maritime_dataset/images/train
val: ./maritime_dataset/images/val
test: ./maritime_dataset/images/test

# 类别数量
nc: 3

# 类别名称（顺序必须与 label 中的 class_id 一致）
names: 
  - person
  - boat
  - ship
```

> 💡 保存为 `maritime_dataset/data.yaml`

---

## 🚀 三、训练代码：`train.py`

由于航拍图像中小目标多（尤其是远距离船只），我们将使用 **大输入尺寸 + 强数据增强 + 预训练模型 + 早停机制**。

```python
# train.py
from ultralytics import YOLO
import torch

# ================== 配置参数 ==================
DATA_CONFIG = 'maritime_dataset/data.yaml'         # 数据集配置文件
MODEL_NAME = 'yolov8x.pt'                          # 推荐使用 x-large 模型提升小目标性能
PROJECT = 'runs/maritime'                          # 项目保存路径
EXPERIMENT_NAME = 'yolov8x_maritime_monitoring'    # 实验名称
EPOCHS = 200                                       # 航拍数据复杂，建议充分训练
BATCH_SIZE = 16                                    # 根据显存调整（建议 ≥ 12GB GPU）
IMG_SIZE = 1280                                    # 提升分辨率以检测小船
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# ================ 加载预训练模型（迁移学习）===============
model = YOLO(MODEL_NAME)

# ================ 开始训练 ================
results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT,
    name=EXPERIMENT_NAME,
    exist_ok=True,
    device=DEVICE,
    workers=8,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    patience=50,                    # 早停：50轮无提升则停止
    save=True,
    save_period=10,
    cache=False,
    single_cls=False,
    close_mosaic=30,                # 最后30轮关闭 Mosaic
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,                   # 小角度旋转增强（适合航拍）
    translate=0.2,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,                     # 保持 Mosaic 增强
    mixup=0.2                       # 混合增强提升泛化
)

print("✅ 训练完成！")
print(f"最佳模型保存路径：{PROJECT}/{EXPERIMENT_NAME}/weights/best.pt")
```

> 📌 训练完成后，最佳模型保存在：
> ```
> runs/maritime/yolov8x_maritime_monitoring/weights/best.pt
> ```

---

## 🔍 四、推理代码：`detect.py`

```python
# detect.py
from ultralytics import YOLO
import cv2
import os

# ================== 配置参数 ==================
MODEL_PATH = 'runs/maritime/yolov8x_maritime_monitoring/weights/best.pt'
SOURCE = 'rtsp://your_drone_stream'  # 支持：图片、视频、RTSP流、摄像头
CONF_THRESHOLD = 0.3                 # 海上目标建议低阈值提高召回
IOU_THRESHOLD = 0.45
SHOW_RESULT = True
SAVE_RESULT = True
OUTPUT_DIR = 'runs/detect/maritime_output'
IMG_SIZE = 1280

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================ 加载模型 ================
model = YOLO(MODEL_PATH)

# ================ 推理 ================
results = model.predict(
    source=SOURCE,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    save=SAVE_RESULT,
    project=OUTPUT_DIR,
    name='detection',
    exist_ok=True,
    imgsz=IMG_SIZE,
    device=DEVICE,
    show=SHOW_RESULT,
    stream=True,
    verbose=True
)

# ================ 遍历结果 ================
for r in results:
    im_array = r.plot()
    if SHOW_RESULT:
        cv2.imshow('Maritime Surveillance', im_array)
        if cv2.waitKey(1) == ord('q'):  # 按 q 退出
            break

cv2.destroyAllWindows()
print(f"✅ 推理完成！结果保存在: {OUTPUT_DIR}")
```

> ✅ 支持输入类型：
> - 图片：`'drone_001.jpg'`
> - 视频：`'ocean_monitoring.mp4'`
> - 摄像头：`0`
> - RTSP 流：`'rtsp://...'`（无人机实时流）

---

## 📊 五、评估代码：`val.py`

```python
# val.py
from ultralytics import YOLO

# ================== 配置参数 ==================
MODEL_PATH = 'runs/maritime/yolov8x_maritime_monitoring/weights/best.pt'
DATA_CONFIG = 'maritime_dataset/data.yaml'
IMG_SIZE = 1280
BATCH_SIZE = 16
CONF_THRESHOLD = 0.001
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# ================ 加载模型 ================
model = YOLO(MODEL_PATH)

# ================ 在验证集上评估 ================
print("📊 正在评估验证集...")
val_metrics = model.val(
    data=DATA_CONFIG,
    split='val',
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    conf=CONF_THRESHOLD,
    iou=0.6,
    device=DEVICE,
    plots=True,
    save_json=True
)

# ================ 在测试集上评估 ================
print("📊 正在评估测试集...")
test_metrics = model.val(
    data=DATA_CONFIG,
    split='test',
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    conf=CONF_THRESHOLD,
    iou=0.6,
    device=DEVICE,
    plots=True,
    save_json=True,
    project='runs/val/test'
)

# ================ 打印关键指标 ================
print("\n" + "="*60)
print("🌊 航空船只检测模型评估结果")
print("="*60)
print("【验证集】")
print(f"mAP@0.5:        {val_metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:   {val_metrics.box.map:.4f}")

print("\n【测试集】")
print(f"mAP@0.5:        {test_metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:   {test_metrics.box.map:.4f}")

# 按类别输出 AP
class_names = ['person', 'boat', 'ship']
print("\n【测试集各类别 AP50】")
for i, name in enumerate(class_names):
    print(f"{name:8} AP50: {test_metrics.box.ap[i]:.4f}")

print("="*60)
```

---


