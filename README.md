# Yolo v26 Tensor导出流程及其不同设备的性能对比

# 实验结果

对于这次尝试，是我第一次编写GitHub文章并且复现最新的Yolo26，但是出于未知原因yolo26在使用TensorRTC++编写完之后出现检测框定位不准确的问题，经过多次修改也未见其成效，于是我将其换成Yolov8模型并成功导出、TensorRTC++编写和成功运行，期间未改变代码的主体，只是读入的模型变了，这也反向证明了C++主体是正确的，但对于v26就不知道是什么原因了。最后发现可能是如下原因：

**模型架构差异分析 (Model Architecture Discrepancy):**

为探究 YOLOv26 部署效果不佳的根本原因，我们引入了同为 "End-to-End" 架构的 **YOLOv10** 进行 ONNX 结构对比。

- **YOLOv10 ONNX 结构（见图）：** 包含了完整的 `TopK` 和 `Gather` 后处理算子，输出形状为 `1x300x6`，实现了真正的 NMS-Free。
- **YOLOv26 ONNX 结构：** 实验中导出的 v26 模型输出为 `1x84x8400`，缺失了端到端后处理子图。

**结论：** 官方宣传的 YOLOv26 NMS-Free 特性依赖于特定的导出参数（如 `end2end=True`）或特定版本的算子支持。在默认导出模式下，v26 回退到了传统输出格式，这导致了与 TensorRT 标准后处理管线的兼容性问题（置信度异常）。

## 1.导出流程

### 1.1 更新或者下载 `ultralytics`

* 推荐使用conda 创建独立环境 `conda create --name yolov26`，并进入环境 `conda activate yolov26`
* 使用 pip 下载 ultralytics `pip install ultralytics`
* 验证 ultralytics版本

  ```
  from ultralytics import YOLO
  print(YOLO("yolo26n.pt").info())  # 应该能下载并加载模型
  ```

### 1.2 导出为ONNX格式

在创建的环境下新建文件，运行下方代码，这里使用的opset为13，最新是18，这里由于本人的Jetson nano 为4GB，所有固定输入尺寸。这里由于我的Jetson是4G版本，所有我关闭了 dynamic

```
from ultralytics import YOLO

# 加载预训练模型（会自动下载）
model = YOLO("yolo26n.pt")  # 或 yolo26s.pt，根据你的显存选择

# 导出 ONNX（推荐参数）
success = model.export(
    format="onnx",      # 导出格式
    imgsz=640,          # 输入尺寸（YOLO26 检测默认 640）
    dynamic=False,       # 支持动态输入形状（batch/size 可变，加分项）
    simplify=True,      # 简化 ONNX 图，减少节点
    opset=18            # 最新 opset，兼容性最好
)

if success:
    print("ONNX 导出成功！文件：yolo26n.onnx")

```

导出成功后得到 `yolo26n.onnx`

对导出的 onnx 做一个检测，运行下方代码，得到输出 `ONNX model check: OK`即可，

```
import onnx

model = onnx.load("yolo26n.onnx")
onnx.checker.check_model(model)
print("ONNX model check: OK")

```

之后使用[Netron.app](https://netron.app)，在网站中打开导出的 onnx 文件，若是模型能够加载成功，没有任何红叉警告，所有算子都显示正常（绿色节点），输出节点上由于YOLO26 是端到端无 NMS，所以通常只有一个输出节点（叫 output 或 output0），形状类似 (1, 84+num_classes, num_anchors) 或 (1, num_boxes, 85)。满足这几个条件，就说明导出成功。

### 1.3 传输到Jetson上

我们要把通用的 ONNX 编译成针对 Orin 架构优化的 `.engine` 文件，提高速度，并验证Yolo26是否有改进

#### 1.3.1 Jetson验证trtexec是否存在

在终端输入 `trtexec --help `，如果有大量代码产生及正常存在，若显示 `bash: trtexec: command not found` ，则使用 `/usr/src/tensorrt/bin/trtexec --help` 即可

#### 1.3.2 使用trtexec 将 onnx 转化为 Tensor Engine

在终端输入下方指令，先去使用FP16 精度验证，之后可以开启 FP18。这里注意如果Jetson的显存较小，并且使用的是vs code的SSH远程服务器进行操作，在运行下方执行之前最好关闭vs code，使用SSH软件去进行操作。并且使用`sudo systemctl stop gdm` 关闭桌面节省显存，并在jtop下的 4MEM 中按英文C建清楚缓存,最终的显存大约在500～700之间即可运行命令。

```python
trtexec \
  --onnx=yolo26n.onnx \
  --saveEngine=yolo26n_fp16.engine \
  --fp16 \

```

#### 1.3.3 使用trtexec 自带的推理进行验证，结果写在2.4

在Jetson终端输入 

```
trtexec --loadEngine=yolov26n.engine --shapes=images:1x3x640x640 --iterations=100
```

### 1.4 使用C++编写推理代码，使用TesorRT直接操控GPU

构建文件树，注意这里在全部文件构建好之后再运行。

```
YOLO26-TensorRT-Cpp/
├── CMakeLists.txt          # 编译规则
├── main.cpp                # 主程序（读取视频、统计FPS）
├── yolo.h                  # 头文件（类定义）
├── yolo.cpp                # 核心实现（加载引擎、推理、后处理）
└── models/
    └── yolov26n.engine     # 你刚才导出的 TensorRT 引擎文件
```

#### 1.4.1 编写 CMakeLists.txt 编译文件

Jetson 预装了 CUDA、TensorRT 和 OpenCV，我们需要在编译文件中链接它们。

这里注意要修改为自己设备的路径，路径有问题的话可以询问AI

#### 1.4.2 编写核心头文件 yolo.h

定义一个 `YoloInfer` 类，封装复杂的 TensorRT 操作，对外只暴露 `infer()` 接口。

#### 1.4.3  编写 核心实现 yolo.cpp

这里是 **End-to-End NMS-Free** 的魔法发生地。注意 `postprocess` 部分，我们不需要写 IOU 计算，直接读结果！

#### 1.4.4  编写主程序，与python类似只是使用C++编写

这里的main.cpp为基础检测视频的代码，main_usb.cpp为使用usb检测平板上相同视频并显示在Jetson屏幕上的代码

#### 1.4.5 编译运行

在YOLO26-TensorRT-Cpp目录下运行下方代码

```
mkdir build
cd build
cmake ..
make -j4  # 使用4个核心编译
cd ../
./build/yolo26_app  #运行
```

## 2 推理性能对比

⚠️新增使用摄像头检测同一使用的对比。由于USB摄像头广角太大可能检测到视频之外的物体并且对于视频中的检测效果可能不是很好。而且设置的阈值也是

### 2.1 Mac cpu 推理，未开启Mac Gpu

运行文件 `v26_maccpu.py`，得到输出，多次运行下的数值可能不同，但相差不

| 视频检测`v26_maccpu.py`                                      | 实时USB摄像头检测`v26_mactorch_usb.py`                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Total frames      : 460<br/>Total time (s)    : 21.5<br/>Average FPS (all) : 21.39<br/>Mean instant FPS  : 39.22 | Total frames      : 308<br/>Total time (s)    : 20.6<br/>Average FPS (all) : 14.96<br/>Mean instant FPS  : 20.62 |

### 2.2 开启Mac的mps GPU推理

将 `v26_maccpu.py`其中的 `DEVICE = 'cpu'`，cpu改为mps即可，视频保存为

| 视频检测`v26_maccpu.py`，`DEVICE = 'cpu'`                    | 实时USB摄像头检测`v26_mactorch_usb.py`                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Total frames      : 460<br/>Total time (s)    : 22.4<br/>Average FPS (all) : 20.53<br/>Mean instant FPS  : 48.35 | Total frames      : 312<br/>Total time (s)    : 21.6<br/>Average FPS (all) : 14.44<br/>Mean instant FPS  : 21.50 |

结果表明，在 macOS 上使用 PyTorch MPS 后端进行 YOLO26 推理时，虽然单次推理速度（instant FPS）明显高于 CPU，但由于 CPU–GPU 同步、视频 I/O 以及 Python 调度开销，端到端 Average FPS 并未提升，甚至略有下降。这表明在轻量模型和逐帧视频推理场景下，通用 GPU 后端（MPS）并不一定优于 CPU。

其后可以采取其他措施去加速Mps的帧率

### 2.3 ONNX 在Mac上的纯Cpu推理

运行v26_maconnx.py ，得到结果为：

| 视频检测`v26_maconnx.py`                                     | 实时USB摄像头检测`v26_maconnx_usb.py`                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Total frames      : 460<br/>Total time (s)    : 28.7<br/>Average FPS (all) : 16.05<br/>Mean instant FPS  : 24.41 | Total frames      : 307<br/>Total time (s)    : 20.9<br/>Average FPS (all) : 14.71<br/>Mean instant FPS  : 20.42 |

可以看出使用onnx框架比pytorch 要慢上许多。

### 2.4 Jetson上trtexec 自带的推理运行onnx转化后的engine

运行命令得到的结果解读为

Throughput: 90.4251 qps ≈ **90 FPS**
端到端的延迟：mean = 12.2366 ms
GPU Compute Time：mean = 11.0188 ms

### 2.5 运行C++ Tensor RT并且编译之后的程序

注意这里如果部署的Jetson显存较小，需要关闭vs code的SSH远程连接和Linux的桌面节省内存，内存占用大约在1.7G左右，由于手持摄像头可能检测的不是很好但是对于内存的占用确实低

#### 2.5.1 Jetson Orin TensorRT (C++) 推理，视频保存失败再次运行一下

| **视频检测**，视频为result_jetson_trt.mp4                    | Usb实时检测，视频为output_usb.mp4                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Total frames      : 460<br/>Total time (s)    : 40.1235<br/>Average FPS (all) : 11.4646<br/>Mean instant FPS  : 40.3066 | Total frames      : 309<br/>Total time (s)    : 22.9845<br/>Average FPS (all) : 13.4439 <br/>Mean Inference FPS: 13.9675 |

在CSI上的运行结果

但是由于保存视频占用CPU，以及一些需要使用CPU处理的函数、Resize, 归一化, HWC转CHW等都是需要使用CPU，之后修改main.cpp，关闭视频保存、绘制预测框和FPS绘制、关闭视频显示输出。修改main.cpp后需要在build下重新编译make -j4.

#### 2.5.2  Jetson Orin TensorRT (C++) 推理，关闭视频保存，

这次的显示使用同样减少到不到1GB，在使用CSI的情况下显存的使用同样不大

| **视频检测**                                                 | Usb实时检测                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Total frames      : 460<br/>Total time (s)    : 13.8549<br/>Average FPS (all) : 33.2012<br/>Mean instant FPS  : 43.1828 | Total frames      : 303<br/>Total time (s)    : 22.7868<br/>Average FPS (all) : 13.2971 )<br/>Mean Inference FPS: 13.7724 |

可以看到关闭视频保存后平均帧率远超Mac上的MPS。运行时间大幅度减少，平均帧数也是最高的，说明了使用更加底层的C++ TensorRT 的有效性。

还可以参照yolo官方的导出进行对照，精力有限就先做到这里。

## 3.⚠️ Limitations & Future Work / 局限性与未来展望

**关于 YOLOv26 的适配说明：** 本项目已成功构建了基于 C++ 和 TensorRT 的高性能通用推理管线，并在 **YOLOv8n** 上验证了其 **30+ FPS** 的工业级稳定性。

然而，在适配实验性的 **YOLOv26** 模型时，我们观察到 ONNX 导出模型在 TensorRT 推理中存在置信度分布异常（Confidence Misalignment）的问题。这可能是由于 v26 特有的端到端（End-to-End）结构在 ONNX 算子转换时的兼容性差异导致的。

限于时间与精力，我未能完美解决 v26 的后处理对齐问题，对此深表遗憾。但我相信这套 C++ 底层架构是正确的。**如果有开发者对 YOLOv26 的 TensorRT 部署感兴趣，非常欢迎提交 PR (Pull Request) 来完善这一部分！** 希望这个项目能成为大家探索嵌入式 AI 的一块垫脚石



\## 🤝 Acknowledgements 

* Ultralytics for the YOLO framework. 
* NVIDIA for TensorRT support.

