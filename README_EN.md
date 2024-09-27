[[Japanese](https://github.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP)/English]

> [!WARNING]
> The dataset used to train the model was collected by ourselves.<br>
> The recognition rate may drop significantly depending on the background, clothing color, and skin color.<br>
> The author(Takahashi) is not responsible for any damage or problems that may occur when using the model.

> [!NOTE]
> Although we have made it compatible with illustrations, <br>
> the accuracy is not very good due to the insufficient number of datasets.

# Skin-Clothes-Hair-Segmentation-using-SMP
It is a model that performs semantic segmentation of 3 classes (skin, clothes, hair).<br>
I am using [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch).<br>
<img src="https://user-images.githubusercontent.com/37477845/132933990-717324f1-2d74-4060-8b67-0bef06058ebe.gif" width="45%">　<img src="https://user-images.githubusercontent.com/37477845/132933998-eae87d48-a98f-43e1-a31c-983e1dea9c1d.gif" width="45%"><br>
<img src="https://github.com/user-attachments/assets/c7cd1491-50cf-4eab-aeea-6b5ab061be56" width="45%">　

In this repository, the following three types of model pth files and onnx files are prepared.
* DeepLabV3+<br>Encoder：timm-mobilenetv3_small_100<br>Parameters：2.16 million
* DeepLabV3+<br>Encoder：timm-mobilenetv3_large_100<br>Parameters：4.71 million

# Requirement 
* torch 1.9.0 or later
* torchsummary 1.5.1 or later
* albumentations 1.0.3 or later
* matplotlib 3.2.2 or later
* onnx-simplifier 0.3.6 later
* opencv-python 3.4.2 or later
* onnxruntime 1.8.1 or later

# Requirement(demo_onnx_xxxx.py)
If you want to use only demo_onnx_simple.py or demo_onnx_image_overlay.py, please install the following two.
* opencv-python 3.4.2 or later
* onnxruntime 1.8.1 or later

# Dataset
I use a total of 452 images taken by ourselves and collected from the Internet.<bR>
The dataset is private.<br><br>
Annotation is performed using [GrabCut-Annotation-Tool](https://github.com/Kazuhito00/GrabCut-Annotation-Tool), and the class assignment is as follows.<br>
* Class ID 1：Skin
* Class ID 2：Clothes
* Class ID 3：Hair

# Training
Open [[Colaboratory]Skin_Clothes_Hair_Segmentation_using_SMP.ipynb]([Colaboratory]Skin_Clothes_Hair_Segmentation_using_SMP.ipynb) in Google Colaboratory and run from top to bottom.<br>
I've committed a few datasets so that my notebook can run, but it's just a learning sample.<br>
It does not reach the accuracy of the model published in this repository.

# Demo
Here's how to run the demo.
```bash
python demo_onnx_simple.py
```
```bash
python demo_onnx_image_overlay.py
```
* --device<br>
Camera device number<br>
Default：0
* --video<br>
Video file ※If you specify a video, it will be executed in preference to the camera.<br>
Default：None
* --image<br>
image file ※If you specify a image, it will be executed in preference to the camera.<br>
Default：None
* --width<br>
Width at the time of camera capture<br>
Default：960
* --height<br>
Height at the time of camera capture<br>
Default：540
* --mirror<br>
Whether to display the image like a mirrorか<br>
Default：unspecified
* --model<br>
Use model<br>
Default：'02.model/DeepLabV3Plus(timm-mobilenetv3_small_100)_452_2.16M_0.8385/best_model_simplifier.onnx'
* --score<br>
Semantic segmentation threshold<br>
Default：0.5

# Reference
* [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [GrabCut-Annotation-Tool](https://github.com/Kazuhito00/GrabCut-Annotation-Tool)

# Author
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)
 
# License 
Skin-Clothes-Hair-Segmentation-using-SMP is under [MIT License](LICENSE).

The image of the woman image is taken from [Free Material Pakutaso](https://www.pakutaso.com).
