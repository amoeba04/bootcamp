import random
import torch
import matplotlib.pyplot as plt


def plot_and_save_losses(G_losses, D_losses, save_path="loss_curves.png"):
    """
    Plot and save the training losses.

    Args:
    - G_losses (list): List of Generator losses across epochs.
    - D_X_losses (list): List of Discriminator X losses across epochs.
    - D_Y_losses (list): List of Discriminator Y losses across epochs.
    - save_path (str): Path to save the plot.

    Returns:
    None
    """
    
    # Plotting Generator Loss
    plt.figure(figsize=(10, 4))
    plt.plot(G_losses, label="Generator Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Generator Training Loss")
    plt.savefig(f"{save_path}_generator.png")
    plt.close()  # Ensure the figure is closed after saving

    # Plotting Discriminator Losses
    plt.figure(figsize=(10, 4))
    plt.plot(D_losses, label="Discriminator X Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Discriminator Training Losses")
    plt.savefig(f"{save_path}_discriminator.png")
    plt.close()  # Ensure the figure is closed after saving
            

def gradient_penalty(real, discriminator):
    real.requires_grad_(True)  # Ensure that requires_grad is set to True
    real_scores = discriminator(real)
    real_grads = torch.autograd.grad(
        outputs=real_scores.sum(), inputs=real,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    penalty = (real_grads.norm(2, dim=1) - 1).pow(2).mean()
    real.requires_grad_(False)  # Reset to original state
    return penalty


# 시간이 지남에 따라 학습률(learning rate)을 감소시키는 클래스
class LambdaLR:
    def __init__(self, n_epochs, decay_start_epoch):
        self.n_epochs = n_epochs # 전체 epoch
        self.decay_start_epoch = decay_start_epoch # 학습률 감소가 시작되는 epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


# 가중치 초기화를 위한 함수 정의
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)