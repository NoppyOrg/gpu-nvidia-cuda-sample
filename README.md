# gpu-nvidia-cuda-sample
NVIDA製GPUを利用しCUDAプラットフォームを利用したGPUのサンプルグログラム


## 説明
#### GPU/ML専用チップ系のインスタンス概略(2023/1現在)(*1)
| Class   | Instance Type   | GPU/Custom Chip     | CPU           | 料金(東京)(*2)  | Remark       |
| ------- | --------------- | ------------------- | ------------- | ------------- | ------------ |
| NVIDIA  | Amazon EC2 P3   | NVIDIA V100         | Intel Xeon    | 4.19〜42.8USD |              |
|         | Amazon EC2 P4   | NVIDIA A100         | Intel Xeon    | 44.9USD       |              |
|         | Amazon EC2 G4dn | NVIDIA T4           | Intel Xeon    | 0.71〜10.6USD |              |
|         | Amazon EC2 G5g  | NVIDIA T4G          | AWS Graviton2 | 0.57〜3.7USD  |              |
|         | Amazon EC2 G5   | NVIDIA A10G         | AMD EPYC      | 1.46〜23.6USD |              |
| AMD     | Amazon EC2 G4ad | AMD Radeon Pro V520 | AMD EPYC      | 0.51〜4.68USD |              |
| AWS     | Amazon EC2 Inf1 | AWS Inferentia      | Intel Xeon    | 0.31〜6.38USD | 推論専用チップ |
|         | Amazon EC2 Inf2 | AWS Inferentia2     | Intel Xeon    | 未提供         | 推論専用チップ |
|         | Amazon EC2 Trn1 | AWS Trainium        | Intel Xeon    | 未提供         |トレーニング専用|

- (*1)インスタンスのスペックについては、[Amazon EC2のインスタンスタイプ説明のページ](https://aws.amazon.com/jp/ec2/instance-types/#Accelerated_Computing)を参照
- (*2)インスタンスの価格は、東京リージョンのOS:Linuxの場合のオンデマンド料金(2023/1現在)になります。料金表記はスペースの関係で四捨五入し簡易表示しています。
- (*3) NVIDIA系インスタンスの詳細については、[こちら](./documents/instances.md)を参照ください。

#### 本サンプルプログラムで利用するEC2インスタンス
このサンプルプログラムではNVIDIA製CPUを利用した最低限の動作確認のため、NVIDIAのGPUとx86系CPUとの組み合わせで最もコストの安い`g4dn.xlarge`インスタンスを利用する。

#### AMIイメージ

## 手順
### GPU環境準備(EC2 + AmazonLinux2)
#### EC2インスタンス作成

- 



#### CUDAセットアップ

