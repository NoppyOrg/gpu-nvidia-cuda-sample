
# AWSのGPU/ML系専用チップ インスタンス整理（詳細）
## インスタンス概略(2023/1現在)
| Class   | Instance Type   | GPU/Custom Chip     | CPU           | 料金(東京)     | Remark       |
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


## NVIDIA系 インスタンス詳細
### P3インスタンス
| Instance Type | GPU model   | CPU model                 | GPUs | GPU Memory| vCPU | Memory |  料金(東京)   |
| ------------- | ----------- | ------------------------- | ---- | --------- | ---- | ------ | ------------ |
| p3.2xlarge	| NVIDIA V100 | Xeon Broadwell E5-2686v4  |  1   | 16GiB     | 8    | 61GiB  | 4.194 USD    |
| p3.8xlarge	| NVIDIA V100 | Xeon Broadwell E5-2686v4  |  4   | 64GiB     | 32   | 64GiB  | 16.776 USD   |
| p3.16xlarge	| NVIDIA V100 | Xeon Broadwell E5-2686v4  |  8   | 128GiB    | 64   | 128GiB | 33.552 USD   |
| p3dn.24xlarge	| NVIDIA V100 | Xeon Skylake 8175         |  8   | 256GiB    | 96   | 768GiB | 42.783 USD   |

### P4インスタンス

| Instance Type | GPU model   | CPU model                 | GPUs | GPU Memory| vCPU | Memory |  料金(東京)   |
| ------------- | ----------- | ------------------------- | ---- | --------- | ---- | ------ | ------------ |
| p4d.24xlarge  | NVIDIA A100 | Xeon Cascade Lake P-8275CL|  8   | 320GiB    | 96   | 1152GiB| 44.92215 USD |
| p4de.24xlarge | NVIDIA A100 | Xeon Cascade Lake P-8275CL|  8   | 640GiB    | 96   | 1152GiB| -            |


### G4dnインスタンス

| Instance Type | GPU model   | CPU model                 | GPUs | GPU Memory| vCPU | Memory |  料金(東京)   |
| ------------- | ----------- | ------------------------- | ---- | --------- | ---- | ------ | ------------ |
| g4dn.xlarge	| NVIDIA T4   | Xeon Cascade Lake P-8259L |  1   | 16GiB     | 4    | 16GiB  | 0.71 USD     |
| g4dn.2xlarge	| NVIDIA T4   | Xeon Cascade Lake P-8259L |  1   | 16GiB     | 8    | 32GiB  | 1.015 USD    |
| g4dn.4xlarge	| NVIDIA T4   | Xeon Cascade Lake P-8259L |  1   | 16GiB     | 16   | 64GiB  | 1.625 USD    |
| g4dn.8xlarge	| NVIDIA T4   | Xeon Cascade Lake P-8259L |  1   | 16GiB     | 32   | 128GiB | 2.938 USD    |
| g4dn.12xlarge	| NVIDIA T4   | Xeon Cascade Lake P-8259L |  4   | 64GiB     | 48	| 192GiB | 5.281 USD    |
| g4dn.16xlarge	| NVIDIA T4   | Xeon Cascade Lake P-8259L |  1   | 16GiB     | 64   | 256GiB | 5.875 USD    |
| g4dn.metal	| NVIDIA T4   | Xeon Cascade Lake P-8259L |  8   | 128GiB    | 96   | 384GiB | 10.562 USD   |

### G5gインスタンス

| Instance Type | GPU model   | CPU model                 | GPUs | GPU Memory| vCPU | Memory |  料金(東京)   |
| ------------- | ----------- | ------------------------- | ---- | --------- | ---- | ------ | ------------ |
| g5g.xlarge    | NVIDIA T4G  | AWS Graviton2             |	1    | 16GiB     | 4    | 8GiB   | 0.5669 USD   |
| g5g.2xlarge   | NVIDIA T4G  | AWS Graviton2             |	1    | 16GiB     | 8    | 16GiB	 | 0.7505 USD   |
| g5g.4xlarge   | NVIDIA T4G  | AWS Graviton2             |	1    | 16GiB     | 16   | 32GiB	 | 1.1176 USD   |
| g5g.8xlarge   | NVIDIA T4G  | AWS Graviton2             |	1    | 16GiB     | 32   | 64GiB  | 1.8519 USD   |
| g5g.16xlarge  | NVIDIA T4G  | AWS Graviton2             |	2    | 32GiB     | 64   | 128GiB | 3.7039 USD   |
| g5g.metal     | NVIDIA T4G  | AWS Graviton2             |	2    | 32GiB     | 64   | 128GiB | 3.7039 USD   |


### G5インスタンス

| Instance Type | GPU model   | CPU model                 | GPUs | GPU Memory| vCPU | Memory |  料金(東京)   |
| ------------- | ----------- | ------------------------- | ---- | --------- | ---- | ------ | ------------ |
| g5.xlarge	    | NVIDIA A10G | AMD EPYC 7R32             | 1    | 24GiB     | 4    | 16GiB  | 1.459 USD    |
| g5.2xlarge    | NVIDIA A10G | AMD EPYC 7R32             | 1    | 24GiB     | 8	| 32GiB  | 1.75776 USD  |
| g5.4xlarge    | NVIDIA A10G | AMD EPYC 7R32             | 1    | 24GiB     | 16	| 64GiB  | 2.35528 USD  |
| g5.8xlarge    | NVIDIA A10G | AMD EPYC 7R32             | 1    | 24GiB     | 32	| 128GiB | 3.55033 USD  |
| g5.16xlarge   | NVIDIA A10G | AMD EPYC 7R32             | 1    | 24GiB     | 64	| 256GiB | 5.94042 USD  |
| g5.12xlarge   | NVIDIA A10G | AMD EPYC 7R32             | 4    | 96GiB     | 48	| 192GiB | 8.22609 USD  |
| g5.24xlarge   | NVIDIA A10G | AMD EPYC 7R32             | 4    | 96GiB     | 96	| 384GiB | 11.81123 USD |
| g5.48xlarge   | NVIDIA A10G | AMD EPYC 7R32             | 8    | 192GiB    | 192	| 768GiB | 23.62246 USD |




- *1 料金は全て、オンデマンド料金、東京リージョン、 OS:Linuxの条件。
