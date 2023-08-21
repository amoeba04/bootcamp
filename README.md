# 이미지 생성모델 파헤치기

- 일정: 

8/21(월) ~ 8/23(수) Theory: Autoencoder, CycleGAN, Stable Diffusion
8/24(목) Practice

- 강의자료: 공지 메일 참고

## 8/24(목) Practice
### Autoencoder (Training, Inference)
- 학습 및 추론
CUDA_VISIBLE_DEVICES=0 python3 Autoencoder.py --batch_size 128 --epochs 5 --latent_dims 10 --lr 1e-3 --use_gpu
- 사전학습 모델 추론
CUDA_VISIBLE_DEVICES=0 python3 Autoencoder.py --batch_size 128 --epochs 5 --latent_dims 10 --lr 1e-3 --use_gpu --pretrained

### CycleGAN (Training, Inference)
- 학습
```
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 4 --epochs 2 --lr 2e-4 --sample_interval 100 --data_path ./horse2zebra --output_dir ./experiments_myname
```

- 추론 (이미지 생성)
```
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 1 --eval --output_dir ./experiment_ori
```

Download pre-trained weight: [Dropbox link](https://www.dropbox.com/scl/fi/4lbvos2n85clcye0qhx1l/checkpoint_latest.pth?rlkey=osbb73vc1phuv34wamixgfxby&dl=0)
### Stable Diffusion (Inference)
- 학습 및 추론: ```StableDiffusion/std_inference.ipynb``` 내 실행

### Frechet Inception Distance (FID)
- 계산
```
CUDA_VISIBLE_DEVICES=0 python fid_score.py /path/to/real/dataset /path/to/fake/dataset
```
Real Image 분포와 Fake Image 분포 간의 거리를 계산하여 Fake Image가 Real Image와 얼마나 가까운지 (얼마나 Real한지)에 대해 계산하는 metric

분포 간의 거리 계산이기 때문에 1:1 매칭되는 Target Image가 필요하지 않음

값이 작을수록 (0에 가까울수록) Fake Image가 Real Image와 비슷함