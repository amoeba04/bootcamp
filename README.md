# 이미지 생성모델 파헤치기

## 8/21(월) ~ 8/23(수) Theory: Autoencoder, CycleGAN, Stable Diffusion

## 8/24(목) Practice
### Autoencoder (Training, Inference)

### CycleGAN (Training, Inference)
학습
```
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 4 --epochs 2 --lr 2e-4 --sample_interval 100 --data_path ./horse2zebra --output_dir ./experiments
```

추론 (이미지 생성)
```
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 1 --eval
```
### Stable Diffusion (Inference)
