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
CUDAのダウンロードページで下記条件のインストール手順に従う。ここではrpmパッケージをネットワーク経由でインストールする方式としている。(なおrpm(local)やrunfile(local)の場合は、ディスク容量を30GB以上にしないとディスク容量不足になる)
- [OS:Linux, Architecture:x86_64 Distribution:RHEL Version9 Installer Type:rpm(network)](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=RHEL&target_version=9&target_type=rpm_network)


なお2023/1現在の最新rpm(network)のCUDAセットアップ手順は以下になる。
```shell
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
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
### サンプルコードの動かし方
CUDAの触りを理解するために、シンプルなサンプルプログラムを用意しています。
- CUDAをセットアップした環境でコードをcloneする
```sh
git clone https://github.com/NoppyOrg/gpu-nvidia-cuda-sample.git
```
- ディレクトリを移動する
```sh
cd gpu-nvidia-cuda-sample
cd src
```
- ファイル説明
    - `cuda_sample.cu` : CUDAを利用したプログラムのシンプルなサンプル
    - `cpu_sample.c` :  上記処理を通常のCPUで処理したサンプル
- makeする
```sh
make
```

- (実行) GPUを利用しない場合
```sh
./cpu_sample.out
```
- (実行) GPUを利用した場合
```sh
./cuda_sample.out
```

### サンプルコードの説明
該当コードは[こちら](src/cuda_sample.cu)

- 処理概要
    - 要素数がそれぞれ6億個の4つの配列(arr1, arr2, arr3, sum_arrの)がある
    - GPUで、６億個の要素それぞれに対して、`sum_arr = arr1 + arr2 + arr3`の計算を行う
#### CUDAコードのポイント
- (1)GPUのデバイス側メモリの確保: 下記でGPU処理で利用する、GPU側のメモリをmallocする
    ```c
        cudaMalloc((void **)&d_arr1, n_byte);
        cudaMalloc((void **)&d_arr2, n_byte);
        cudaMalloc((void **)&d_arr3, n_byte);
        cudaMalloc((void **)&d_sum_arr, n_byte);
    ```
- (2)メインメモリ(ホスト)からデバイス(GPU側のメモリ)へのデータ転送:
    ```c
        cudaMemcpy(d_arr1, arr1, n_byte, cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr2, arr2, n_byte, cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr3, arr3, n_byte, cudaMemcpyHostToDevice);
    ```
- (3)GPUでの処理実行
    ```c
    sum_of_array<<<(N + 255ULL) / 256ULL, 256ULL>>>(d_arr1, d_arr2, d_arr3, d_sum_arr);
    ```
- (4)GPUでの処理結果をメインメモリ(ホスト)に戻す
    ```c
    cudaMemcpy(sum_arr, d_sum_arr, n_byte, cudaMemcpyDeviceToHost);
    ```
- GPU処理ロジック
    - 通常のCPU処理ではfor文で回しているが、CUDAでは`sum_of_array<<<(N + 255ULL) / 256ULL, 256ULL>>>`の部分で処理をgridとtreadに分割して、パラレルで処理させているイメージになる。
    - `blockIdx.x * blockDim.x + threadIdx.x`の部分で、gridとthread番号から、配列内のどの要素の処理かを計算している
    ```c
    __global__ void sum_of_array(float *arr1, float *arr2, float *arr3, float *sum_arr)
    {

        unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
        *(sum_arr + i) = *(arr1 + i) + *(arr2 + i) + *(arr3 + i);

        return;
    }
    ```

## リファレンス
- CUDAセットアップ
    - [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
    - [CUDA Toolkit 12.0 Downloads](https://developer.nvidia.com/cuda-downloads?target_os=Linux)
    -  [EC2 G4インスタンスのAmazon Linux 2にNVIDIA CUDAをインストールしてみた](https://dev.classmethod.jp/articles/install-nvidia-cuda-on-ec2-g4-amazon-linux-2/)
- CUDAプログラミング
    - [NVIDIA CUDAプログラミングの基本 Part1 ソフトウェアスタックとメモリ管理](https://http.download.nvidia.com/developer/cuda/jp/CUDA_Programming_Basics_PartI_jp.pdf)
    - [NVIDIA CUDAプログラミングの基本 Part2 カーネル](https://http.download.nvidia.com/developer/cuda/jp/CUDA_Programming_Basics_PartII_jp.pdf)
    - [CUDAを使ってGPUプログラミングに挑戦してみた。](https://nonbiri-tereka.hatenablog.com/entry/2017/04/11/081601) (ただしリンク先のプログラムは正常に動かないので注意)