import random
import torch
import matplotlib.pyplot as plt


def plot_and_save_losses(G_losses, D_losses, save_path="loss_curves.png"):
    """
    Generator, Discriminator loss를 그래프로 그려 저장하는 함수
    Args:
    - G_losses: 각 epoch Generator loss 리스트
    - D_losses: 각 epoch Discriminator loss 리스트
    - save_path: 그래프를 저장할 경로
    """
    
    # Plotting Generator Loss
    plt.figure(figsize=(10, 4))
    plt.plot(G_losses, label="Generator Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Generator Training Loss")
    plt.savefig(f"{save_path}_generator.png")
    plt.close()

    # Plotting Discriminator Losses
    plt.figure(figsize=(10, 4))
    plt.plot(D_losses, label="Discriminator X Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Discriminator Training Losses")
    plt.savefig(f"{save_path}_discriminator.png")
    plt.close()


class ReplayBuffer:
    def __init__(self, max_size=50):
        """
        ReplayBuffer에 사용되는 buffer 클래스. 이미지를 일정 개수만큼 저장.
        Args:
        - max_size: 버퍼의 최대 크기. 기본값은 50.
        """
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        """
        새로운 이미지 데이터를 버퍼에 추가하고, 이전에 저장되었던 이미지를 반환하는 함수
        Args:
        - data: 버퍼에 추가할 이미지 데이터.
        Returns:
        - torch.Tensor: 버퍼에서 선택된 이미지 데이터.
        """
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            # 아직 버퍼가 가득 차지 않았다면, 현재 삽입된 데이터를 반환
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            # 버퍼가 가득 찼다면, 이전에 삽입되었던 이미지를 랜덤하게 반환
            else:
                if random.uniform(0, 1) > 0.5: # 확률은 50%
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element # 버퍼에 들어 있는 이미지 교체
                else:
                    to_return.append(element)
        return torch.cat(to_return)


# 시간이 지남에 따라 학습률(learning rate)을 감소시키는 클래스
class LambdaLR:
    def __init__(self, n_epochs, decay_start_epoch):
        """
        학습률을 점진적으로 감소시키기 위한 스케줄러 클래스.
        Args:
        - n_epochs: 전체 학습 epoch의 수.
        - decay_start_epoch: learning rate decay를 시작할 epoch.
        """
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


# 가중치 초기화를 위한 함수 정의
def weights_init_normal(m):
    """
    네트워크의 가중치를 초기화하는 함수.
    
    Args:
    - m (torch.nn.Module): 초기화할 레이어.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)