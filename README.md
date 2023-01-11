# gpu-nvidia-cuda-sample
NVIDA製GPUを利用しCUDAプラットフォームを利用したGPUのサンプルグログラム


## 説明
### 実行環境(EC2インスタンス)
#### 本サンプルプログラムで利用するEC2インスタンス
このサンプルプログラムではNVIDIA製CPUを利用した最低限の動作確認のため、NVIDIAのGPUとx86系CPUとの組み合わせで最もコストの安い`g4dn.xlarge`インスタンスを利用する。

- GPUまたはカスタムチップを搭載したインスタンスの詳細については、[こちら](./documents/instances.md)を参照ください。

#### AMIイメージ
今回は、LinuxベースでCUDAを利用したGPUサンプルを作成するため、以下の構成とする。
- ベースOS: `RHEL9(64-bit(x86))`を利用(CUDAパッケージはAmazonLinux2身サポートのため、サポートしているRHELを利用)
- NVIDIAドライバ: `Teslaドライバ`を利用(CUDAと一緒にインストール)
- GPUプラットフォーム: `CUDA`を利用

## 手順
### GPU環境準備(EC2 + AmazonLinux2)
#### EC2インスタンス作成
マネージメントコンソールで下記インスタンスを作成する。
- OS Image: `RHEL9(64-bit(x86))`を選択
- Instance type: `g4dn.xlarge`を選択
- Key pair: 利用したいキーペアを指定
- ネットワーク
    - VPC: 利用したいVPCを指定。特に指定がなければデフォルトVPCを利用
    - セキュリティーグループ: ssh接続ができるようInboundでTCP Port22を許可する
- ストレージ: `1x 30GiB gp3`に変更(デフォルトの8GiBではインストールに必要な容量が足りないため変更)
- そのの他: デフォルト設定

#### CUDAセットアップ
##### OS環境確認
作成したEC2インスタンスにec2-userでSSHログインし、以下作業を進める。
- NVIDIA GPU確認
```shell
lspci | grep -i nvidia
````
`00:1e.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)`のようにNVIDIA T4が表示されればOK

- OSアーキテクチャの確認
```shell
uname -m && cat /etc/*release
```
`x86_64`で`RHEL9`が表示されればOK

##### 事前準備
- 前提ツールのインストール
```shell
sudo yum -y install wget git
```


- サードパーティー用yumリポジトリの有効化
```shell
sudo dnf install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm

sudo subscription-manager repos --enable=rhel-9-for-x86_64-appstream-rpms
sudo subscription-manager repos --enable=rhel-9-for-x86_64-baseos-rpms
sudo subscription-manager repos --enable=codeready-builder-for-rhel-9-x86_64-rpms
```

##### CUDAのインストール
CUDAのダウンロードページで下記条件のインストール手順に従う
- [OS:Linux, Architecture:x86_64 Distribution:RHEL Version9 Installer Type:rpm(local)](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=RHEL&target_version=9&target_type=rpm_local)


なお2023/1現在の最新CUDAの場合は以下手順になる。
```shell
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-rhel9-12-0-local-12.0.0_525.60.13-1.x86_64.rpm
sudo rpm -ivh cuda-repo-rhel9-12-0-local-12.0.0_525.60.13-1.x86_64.rpm
sudo dnf clean all

sudo dnf -y module install nvidia-driver:latest-dkms
sudo dnf -y install cuda
```

##### 事後処理
- PATH環境変数の設定
```shell
mkdir ~/.bashrc.d/
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' > ~/.bashrc.d/env_for_cuuda
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc.d/env_for_cuuda
```

- 設定の反映
設定後、一度ログアウト&ログインして設定を反映させる。

- パスの確認
```shell
nvcc --version
```

#### NVIDIAドライバーの動作確認
```shell
cat /proc/driver/nvidia/version
```
#### CUDAを利用したプログラムのbuild動作テスト
[CUDAのサンプルプログラム](https://github.com/nvidia/cuda-samples)をbuildできるかを確認する。
- コードをclone
```shell
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
```
- buildする
```shell
make
```
- ビルドしたサンプルの動作テスト
```shell
./bin/x86_64/linux/release/deviceQuery
```
以下のような結果が表示されれば成功
```
./bin/x86_64/linux/release/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Tesla T4"
  CUDA Driver Version / Runtime Version          12.0 / 12.0
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 14960 MBytes (15686828032 bytes)
  (040) Multiprocessors, (064) CUDA Cores/MP:    2560 CUDA Cores
  GPU Max Clock rate:                            1590 MHz (1.59 GHz)
  Memory Clock rate:                             5001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 30
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.0, CUDA Runtime Version = 12.0, NumDevs = 1
Result = PASS
```

## サンプルコード
- ディレクトリを移動してmakeする
```sh
cd src
make
```

- GPUを利用しない場合
```sh
./cpu_sample.out
```
- GPUを利用した場合
```sh
./cuda_sample.out
```

## リファレンス
- CUDAセットアップ
    - [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
    - [CUDA Toolkit 12.0 Downloads](https://developer.nvidia.com/cuda-downloads?target_os=Linux)
    -  [EC2 G4インスタンスのAmazon Linux 2にNVIDIA CUDAをインストールしてみた](https://dev.classmethod.jp/articles/install-nvidia-cuda-on-ec2-g4-amazon-linux-2/)
- CUDAプログラミング
    - [CUDAを使ってGPUプログラミングに挑戦してみた。](https://nonbiri-tereka.hatenablog.com/entry/2017/04/11/081601)