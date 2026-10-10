# YOLO 学习合集

YOLO 系列目标检测的学习与复现仓库，覆盖 YOLOv1 / v5 / v8 / v12 / v13 五个版本：

| 目录 | 性质 | 任务 | 说明 |
|---|---|---|---|
| [`YOLOV1/`](YOLOV1/) | **从零复现** | 摩托车头盔检测（4 类） | 基于 PyTorch 手写 YOLOv1，主干网络替换为改进的 ResNet50 |
| [`YOLOV5/`](YOLOV5/) | 框架源码 | — | ultralytics YOLOv5 完整代码库，用于对照学习 |
| [`YOLOV8/`](YOLOV8/) | 框架源码 + 微调 | 摩托车头盔检测（4 类） | ultralytics YOLOv8，含微调脚本与数据集配置 |
| [`YOLOV12/`](YOLOV12/) | 框架源码 + 训练 | 猫狗分类（2 类） | ultralytics YOLOv12，含训练脚本、数据集与预测输出 |
| [`YOLOV13/`](YOLOV13/) | 框架源码 + 训练 | 猫狗分类（2 类） | ultralytics YOLOv13，含 12 次训练实验记录 |

**两条任务主线：**

- **摩托车头盔佩戴检测**（4 类：`no helmet` / `motor` / `number` / `with helmet`）—— 见于 `YOLOV1` 与 `YOLOV8`
- **猫狗分类**（2 类：`cat` / `dog`）—— 见于 `YOLOV12` 与 `YOLOV13`

---

## 一、YOLOV1 —— 摩托车头盔检测

### 任务背景

检测骑乘场景中的四类目标，用于判断骑手是否规范佩戴头盔：

```python
VOC_CLASSES = ('no helmet', 'motor', 'number', 'with helmet')
```

| 类别 | 含义 |
|---|---|
| `no helmet` | 未戴头盔的头部 |
| `motor` | 摩托车 |
| `number` | 车牌 |
| `with helmet` | 已戴头盔的头部 |

### 模块结构

| 文件 | 职责 | 关键内容 |
|---|---|---|
| `new_resnet.py` | 主干网络 | 改进 ResNet50：`Bottleneck` 基本块 + `output_net` 输出块（1×1 → 3×3 **空洞卷积** → 1×1），输出 `7×7×(CLASS_NUM+10)` |
| `yoloLoss.py` | 损失函数 | YOLOv1 复合损失，含 `compute_iou` 实现，按坐标 / 置信度 / 类别分项 |
| `yoloData.py` | 数据加载 | `yoloDataset` 继承 `Dataset`，从 txt 列表文件读取，含坐标归一化与数据增强 |
| `train.py` | 训练 | 加载 torchvision 预训练权重、SGD 优化、逐 epoch 保存 |
| `predict.py` | 推理 | 前向 → `Decode` 解码 → `NMS` 非极大值抑制 |
| `write_txt.py` | 数据准备 | 解析 VOC 格式 XML，按 7:3 划分并生成 `voctrain.txt` / `voctest.txt` |
| `VOCdevkit/VOC2007/` | 数据集占位 | 内含 `必看.txt` 说明目录约定 |

### 网络结构

输入尺寸固定 **448×448**，主干沿用 ResNet50 的五个阶段，尾部替换：

```
输入 3×448×448
  → layer0 (7×7 conv, stride 2 + MaxPool)     → 64×112×112
  → layer1 (Bottleneck ×3)                     → 256×112×112
  → layer2 (Bottleneck ×4)                     → 512×56×56
  → layer3 (Bottleneck ×6)                     → 1024×28×28
  → layer4 (Bottleneck ×3)                     → 2048×14×14
  → layer5 (output_net ×3, 空洞卷积)           → 256×14×14
  → AvgPool2d(2)                               → 256×7×7
  → conv_end (3×3, 输出 CLASS_NUM+10 通道)      → 14×7×7
  → sigmoid → permute                          → 7×7×14
```

> YOLOv1 原始设计输出为 `7×7×30`（20 类 + 10）。本仓库将类别数改为 4，输出通道相应变为 `4 + 10 = 14`。
> 源码注释中仍保留 `7*7*30` 的写法，是原版遗留，阅读时注意区分。

### 损失函数

`yoloLoss(S=7, B=2, l_coord=5, l_noobj=0.5)`，各分项按权重合成：

| 分项 | 说明 | 权重 |
|---|---|---|
| `loc_loss` | 边界框中心坐标 + **宽高开方**（缓解大小框权重失衡） | `5` |
| `contain_loss` | 含目标格点中 IoU 较大框的置信度 | `1` |
| `no_contain_loss` | 含目标格点中舍去框的置信度 | `0.5` |
| `nooobj_loss` | 不含目标格点的置信度 | `0.5` |
| `class_loss` | 类别损失 | `1` |

置信度目标值取预测框与真值框的 **IoU**；两个候选框中 IoU 较大者负责回归坐标，另一个仅计入置信度损失。

### 训练配置

| 参数 | 值 |
|---|---|
| 设备 | `cuda` |
| batch_size | `2`（显存充足可调至 4 / 8 / 16 / 32） |
| learning_rate | `0.001` |
| epochs | `100` |
| 优化器 | SGD |
| 预训练 | torchvision `resnet50` 权重，按参数顺序逐位加载（跳过全连接层，`op_num <= 317`） |

### 使用流程

**使用 VOC2007 数据集：**

```
1. 下载 VOC2007 数据集，放入 VOCdevkit/VOC2007/ 目录
2. 运行 write_txt.py  → 生成 voctrain.txt 与 voctest.txt
3. 运行 train.py      → 开始训练，权重保存为 yolo.pth
4. 在 predict.py 中填写图片路径与权重路径，运行推理
```

**使用自建数据集**（推荐，与头盔检测任务对应）：

```
1. 将标注 xml 文件放入 VOCdevkit/VOC2007/Annotations/
2. 将图片 jpg 文件放入 VOCdevkit/VOC2007/JPEGImages/
3. 修改 write_txt.py 与 predict.py 中的 VOC_CLASSES
4. 修改 new_resnet.py、yoloData.py、yoloLoss.py 中的 CLASS_NUM
5. 运行 write_txt.py 生成列表文件
6. 运行 train.py 开始训练
```

> ⚠️ **`CLASS_NUM` 必须四处保持一致** —— `new_resnet.py`（决定输出通道）、`yoloData.py`、`yoloLoss.py`，以及 `predict.py` 中由 `VOC_CLASSES` 长度推导的值。任一处不匹配都会在加载权重时报维度错误。

### 注意事项

- **主干并非标准 ResNet50**：YOLOv1 的输入输出与分类网络不同，因此该结构在 ResNet50 基础上做了改动（尾部替换为空洞卷积输出块），两者权重不能完全通用。
- **VOC2007 需要较多轮次**：使用 VOC2007 训练约需 **200 epoch** 才能收敛；自建小数据集收敛更快。
- **数据集目录需自建**：`VOCdevkit/VOC2007/` 下仅保留说明文件，`Annotations/` 与 `JPEGImages/` 需自行创建（见 `必看.txt`）。

---

## 二、YOLOV5 —— 框架源码

`YOLOV5/` 为 ultralytics YOLOv5 的完整代码库，未作修改，用作对照学习：

```
YOLOV5/
├── detect.py / train.py / val.py / export.py   # 检测、训练、验证、导出
├── hubconf.py                                  # torch.hub 入口
├── classify/                                   # 分类任务
├── segment/                                    # 分割任务
├── models/                                     # 网络结构与配置
├── data/                                       # 数据集 yaml 与示例图片
└── utils/                                      # 通用工具
```

使用方法参见 [`YOLOV5/README.zh-CN.md`](YOLOV5/README.zh-CN.md)。

---

## 三、YOLOV8 —— 头盔检测微调

`YOLOV8/` 为 ultralytics YOLOv8 代码库（`ultralytics/yolo/` 的早期目录结构），在其上做了头盔检测的微调实验。

### 自定义文件

| 文件 | 说明 |
|---|---|
| `train.py` | 微调脚本：加载 `yolov8s.pt`，数据集指向 `score_data.yaml`，10 epoch，batch 4，SGD |
| `score_data.yaml` | 数据集配置：**nc=4**，类别 `motor` / `no helmet` / `number` / `with helmet` |
| `inference.py` | 推理脚本 |
| `export.py` | 模型导出 |

`score_data.yaml` 中数据路径指向本机 `D:\DeepLearning\Data\yolov5data`，在别处运行需改为实际路径。

> 与 `YOLOV1/` 是**同一任务的两种实现**：YOLOV1 是手写复现，YOLOV8 是框架微调。对照两者可直观体会从零实现与工程框架的差异。

---

## 四、YOLOV12 —— 猫狗分类训练

`YOLOV12/` 为 ultralytics YOLOv12 代码库，配套一次完整的猫狗检测训练实践。

### 自定义文件

| 文件 | 说明 |
|---|---|
| `train.py` | 从 `yolov12.yaml` 构建模型并加载 `yolov12n.pt`，50 epoch，batch 4，`close_mosaic=10` |
| `detect.py` | 推理脚本 |
| `data.yaml` | 数据集配置：**nc=2**，类别 `dog` / `cat` |
| `app.py` | 应用入口（来自教程示例） |
| `xml转txt.py` | 标注格式转换：VOC XML → YOLO TXT |
| `划分.py` | 数据集划分（train / val） |
| `logs/` | 各规模模型的 benchmark 结果：`yolov12{n,s,m,l,x}.csv` |

### 训练产出

`runs/detect/` 下保留了预测结果（图像），预测视频已按体积忽略。
数据集 `yolotest/`（dog / cat 图像）本地保留，仓库内未提交。

---

## 五、YOLOV13 —— 猫狗分类（含实验记录）

`YOLOV13/` 为 ultralytics YOLOv13 代码库，保留了较完整的训练实验过程。

### 自定义文件

| 文件 | 说明 |
|---|---|
| `train.py` | 从 `yolov13n-seg.yaml` 构建模型并加载 `yolov13n.pt`，10 epoch，batch 4 |
| `detect.py` | 推理脚本 |
| `data.yaml` | 数据集配置：**nc=2**，类别 `cat` / `dog` |
| `json_to_txt.py` | 标注格式转换：LabelMe JSON → YOLO TXT |
| `xml转txt.py` | 标注格式转换：VOC XML → YOLO TXT |
| `划分.py` | 数据集划分 |
| `yolov13n-seg.yaml` | 分割任务模型配置 |

### 训练实验记录

`runs/train/` 下保存了 **12 次训练实验**（`exp`、`exp2` ~ `exp12`），每次实验保留：

```
runs/train/expN/
├── args.yaml                  # 该次训练的全部超参数
├── results.csv                # 逐 epoch 的损失与指标（训练日志）
├── labels.jpg                 # 标注分布统计
├── labels_correlogram.jpg     # 标注相关性分析
└── train_batch{0,1,2}.jpg     # 训练批次可视化
```

> 各实验目录下的 `weights/`（`best.pt` / `last.pt`）已按体积忽略，需要时从本地备份取回。

`runs/detect/` 与 `runs/segment/` 保留了预测结果图像。

> ⚠️ **注意 `data.yaml` 的类别顺序**：`YOLOV12` 是 `['dog','cat']`，`YOLOV13` 是 `['cat','dog']`，两者相反。跨版本复用权重或对比结果时容易搞错，需留意。

---

## 六、关于 .gitignore

本仓库**只提交源码、配置与训练日志**，大体积二进制文件一律忽略。规则见 [`.gitignore`](.gitignore)：

| 忽略类别 | 具体内容 | 体积 |
|---|---|---|
| 模型权重 | `*.pt` / `*.pth` / `*.onnx` / `*.engine` 等 | 约 110 MB |
| 视频 | `*.avi` / `*.mp4`（含预测输出与示例视频） | 约 110 MB |
| 数据集 | `dataset/`、`yolotest/`、`**/images/{train,val}` | — |
| 安装包 | `*.whl`（含 2.6 GB 的 PyTorch 安装包） | 约 2.6 GB |
| 论文与压缩包 | `*.pdf` / `*.zip` | 约 40 MB |
| IDE 与缓存 | `.idea/` / `__pycache__/` 等 | — |

**刻意保留**的内容：

- `runs/train/exp*/` 下的 `args.yaml`、`results.csv` 与训练曲线图 —— 这是训练日志，是本仓库最有参考价值的部分
- `assets/` —— ultralytics 各版本 README 引用的配图（约 22 MB）

---

## 环境要求

| 依赖 | 版本 |
|---|---|
| Python | 3.7+（YOLOV12 使用 3.11） |
| PyTorch | 1.7.1（YOLOV1，已测试 1.11.0）；YOLOV12 使用 2.2.2+cu118 |
| torchvision | 与 PyTorch 匹配 |
| opencv-python / numpy / Pillow | 近期版本 |
| CUDA | 训练建议使用 GPU |

各版本另有独立依赖，见各自目录下的 `requirements.txt`。

---

## 数据集与权重

仓库中**不包含**训练数据与模型权重（体积原因，`.gitignore` 已排除）：

| 文件 | 位置 | 大小 |
|---|---|---|
| `yolov8s.pt` | `YOLOV8/` | 21.5 MB |
| `yolov12n.pt` | `YOLOV12/` | 5.3 MB |
| `yolov13n.pt` | `YOLOV13/` | 10.1 MB |
| `best.pt`（猫狗分类训练成果） | `YOLOV13/` | 5.7 MB |
| `runs/train/exp*/weights/` | `YOLOV13/` | 约 67 MB |

官方预训练权重可从 ultralytics 官方发布页下载；自训练产物请从本地备份取回。
`YOLOV1/README.md` 中另附有 VOC2007 数据集与自建数据集的网盘链接。

---

## 备注

- 训练时主干网络预训练权重由 `torch.utils.model_zoo` 在线下载。实际使用中发现 **VS Code 与 PyCharm 中的下载速度存在差异**，若长时间卡在下载阶段，可手动下载 `resnet50-19c8e357.pth` 后本地加载。
- `YOLOV1/` 的自写实现参考了 [YOLOV1-pytorch](https://blog.csdn.net/ing100/article/details/125155065) 的改进 ResNet50 主干思路，在此基础上适配了头盔检测任务。
- `YOLOV12/` 与 `YOLOV13/` 的 `train.py`、`detect.py` 等脚本源自公开教程（作者署名「挂科边缘」），非本人原创，此处仅作学习留档。
- 各目录内的 `data.yaml` / `score_data.yaml` 中的绝对路径均为本机路径，换机器运行需自行修改。

## 许可证

`YOLOV5/`、`YOLOV8/`、`YOLOV12/`、`YOLOV13/` 部分遵循其原始 **AGPL-3.0** 许可。
`YOLOV1/` 为学习复现代码，仅供学习交流。
