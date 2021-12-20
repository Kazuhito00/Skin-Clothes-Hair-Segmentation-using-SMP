[Japanese/[English](https://github.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP/blob/main/README_EN.md)]

# ⚠Attention⚠
モデルを訓練するために使用したデータセットは自前で収集したものです。<br>
背景や服の色、肌の色によって認識率が大きく下がる可能性があります。<br>
モデルの使用に際し、損害や問題が発生しても、作成者(高橋)は一切責任を負いません。

# Skin-Clothes-Hair-Segmentation-using-SMP
3クラス(肌、服、髪)のセマンティックセグメンテーションを実施するモデルです。<br>
[Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch)を使用しています。<br>
<img src="https://user-images.githubusercontent.com/37477845/132933990-717324f1-2d74-4060-8b67-0bef06058ebe.gif" width="45%">　<img src="https://user-images.githubusercontent.com/37477845/132933998-eae87d48-a98f-43e1-a31c-983e1dea9c1d.gif" width="45%"><br>

本リポジトリでは、以下3種類のモデルのpthファイルとonnxファイルを用意しています。
* DeepLabV3+<br>エンコーダー：timm-mobilenetv3_small_100<br>パラメータ数：約216万
* PAN(Pyramid Attention Network)<br>エンコーダー：timm-mobilenetv3_small_100<br>パラメータ数：約102万
* U-Net++<br>エンコーダー：timm-mobilenetv3_small_100<br>パラメータ数：約371万

# Requirement 
* torch 1.9.0 or later
* torchsummary 1.5.1 or later
* albumentations 1.0.3 or later
* matplotlib 3.2.2 or later
* onnx-simplifier 0.3.6 later
* opencv-python 3.4.2 or later
* onnxruntime 1.8.1 or later

# Requirement(demo_onnx_xxxx.py)
demo_onnx_simple.py や demo_onnx_image_overlay.py のみを使用する場合は以下2つをインストールしてください。
* opencv-python 3.4.2 or later
* onnxruntime 1.8.1 or later

# Dataset
自前で撮影した画像やインターネット上から収集した画像を合計452枚使用しています。<bR>
データセットは非公開です。<br><br>
アノテーションは[GrabCut-Annotation-Tool](https://github.com/Kazuhito00/GrabCut-Annotation-Tool)を用いて実施しており、クラスの割り当ては以下の通りです。<br>
* クラスID 1：肌
* クラスID 2：服
* クラスID 3：髪

# Training
Google Colaboratoryで [[Colaboratory]Skin_Clothes_Hair_Segmentation_using_SMP.ipynb]([Colaboratory]Skin_Clothes_Hair_Segmentation_using_SMP.ipynb) を開き、上から順に実行してください。<br>
ノートブックが実行できるように、数枚のデータセットをコミットしてありますが、あくまで学習サンプルなので、<bR>
本リポジトリで公開しているモデルの精度には及びません。

# Demo
デモの実行方法は以下です。
```bash
python demo_onnx_simple.py
```
```bash
python demo_onnx_image_overlay.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --file<br>
動画ファイルの指定 ※動画指定時はカメラより優先<br>
デフォルト：None
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --mirror<br>
画像を反転表示するか否か<br>
デフォルト：指定なし
* --model<br>
使用するモデル<br>
デフォルト：'02.model/DeepLabV3Plus(timm-mobilenetv3_small_100)_452_2.16M_0.8385/best_model_simplifier.onnx'
* --score<br>
セマンティックセグメンテーションの判定閾値<br>
デフォルト：0.5

# Reference
* [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [GrabCut-Annotation-Tool](https://github.com/Kazuhito00/GrabCut-Annotation-Tool)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Skin-Clothes-Hair-Segmentation-using-SMP is under [MIT License](LICENSE).

また、女性の画像は[フリー素材ぱくたそ](https://www.pakutaso.com)様の写真を利用しています。
