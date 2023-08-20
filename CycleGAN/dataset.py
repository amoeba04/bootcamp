import os
import glob
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, root, transforms=None, mode="train"):
        """
        ImageDataset 클래스
        Args:
        - root: 이미지 데이터셋의 경로.
        - transforms: 이미지에 적용할 전처리 함수들.
        - mode: 데이터 로딩 모드 ('train' 또는 'test').
        """
        self.transform = transforms

        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A") + "/*.jpg"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}B") + "/*.jpg"))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")
        img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB") # img_B는 랜덤하게 샘플링

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}


def get_dataloader(root, image_size=256, batch_size=1, shuffle=True, num_workers=4):
    """
    주어진 설정을 기반으로 train, test dataloader 를 생성하는 함수
    Args:
    - root: 이미지 데이터셋의 경로.
    - image_size: 이미지 크기.
    - batch_size: 배치 크기.
    - shuffle: 학습 데이터 셔플 여부.
    - num_workers: 데이터 로딩에 사용될 worker의 수.
    
    Returns:
    - train_dataLoader, test_dataloader: train, test dataloader.
    """
    train_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.12), Image.BICUBIC), # 이미지 크기를 조금 키우기
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(), # 각 데이터가 단일 이미지로 존재하므로 좌우 반전 가능
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transforms = transforms.Compose([
        # transforms.Resize(int(image_size * 1.12), Image.BICUBIC), # 이미지 크기를 조금 키우기
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ImageDataset(root, transforms=train_transforms, mode='train')
    test_dataset = ImageDataset(root, transforms=test_transforms, mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dataloader, test_dataloader
