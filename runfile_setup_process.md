
##### GCCのセットアップ
```shell
sudo yum -y install gcc
```

##### OSの最新化
```shell
sudo yum -y update
sudo reboot
```

##### カーネルヘッダとカーネル開発ツールのインストール
再起動後に以下を実行する。
```shell
#レポジトリの有効化
sudo amazon-linux-extras install epel

#カーネルヘッダとカーネル開発ツールのインストール
sudo yum -y install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
```

##### CUDAのインストール
CUDAのダウンロードページで下記条件のインストール手順に従う
- [OS:Linux, Architecture:x86_64 Distribution:RHEL Version7 Installer Type:runfile(local)](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=RHEL&target_version=7&target_type=runfile_local)

2023/1時点の最新バージョンの場合は以下の手順になる。(上記リンク先の最新の手順に従うこと)
```shell
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run
```

実行すると以下の問い合わせがあるのでそれぞれ入力する。
- End User License Agreement: `accept`と入力しリターン
- CUDA Installer: デフォルトのままで`Install`をリターン


##### CUDA用のパス設定