# imgshadowrm_py

This is a sample implementation of an image shadow removal node in **ROS2 Jazzy** / **Ubuntu 24.04**. The node subscribes to shadow images from the `/image_topic` and publishes deshadowed images to `/processed_image`.

## 1. Dataset
Due to time limitations and for convenience, two public datasets are used in this project:
- **SRD** (Shadow Removal Dataset)
- **ISTD** (Illumination and Shadow Dataset)

## 2. Model Development
To perform inference on the CPU, a classical model [**DeShadowNet**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qu_DeshadowNet_A_Multi-Context_CVPR_2017_paper.pdf) is explored. Additionally, computer vision methods were tested as a benchmark for comparison.

## 3. CPU Acceleration Strategies
To accelerate the processing on CPU, several strategies are employed:
- **Quantization**: Dynamic Quantization is implemented using `torch.quantization.quantize_dynamic`.
- **Inference with ONNX**: Using `onnxruntime`, inference latency decreased from 7.2s to 0.16s on a CPU (tested on an AMD 7000 series processor).
- **Future Work**:
  - Revise the model structure (e.g., use a lighter backbone network).
  - Implement the inference node in **C++** instead of Python to optimize pre-processing and post-processing.

## 4. Training Process and Results
The training process was conducted on an **NVIDIA 4060 8GB GPU** within 3 hours. The dataset consisted of approximately 2600 images for training and 500 images for testing (from the SRD dataset).

### Metrics:
- **Begin Matrix** (Epoch 2):
  - **RMSE**: 65.23
  - **PSNR**: 11.99
  - **SSIM**: 0.39
  - **LPIPS**: 0.60
- **End Matrix** (Epoch 86):
  - **RMSE**: 25.52
  - **PSNR**: 20.49
  - **SSIM**: 0.66
  - **LPIPS**: 0.19

## 5. Results
### Image 1 from SRD Test Set:
![image 1](https://github.com/user-attachments/assets/34c917d9-6cd9-467e-a200-35b5031f5c9d)

### Image 2:
![image 2](https://github.com/user-attachments/assets/18f9b322-c4d4-470d-91d0-a4a9a65385e9)

### Image 3:
![image 3](https://github.com/user-attachments/assets/43d1409e-23fd-4687-86e6-7f1e4d2ad5a5)

### Image 4:
![image 4](https://github.com/user-attachments/assets/ad5c21f5-2fdd-43f8-88ee-e1769a48bda4)
