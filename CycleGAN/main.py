import os
import time
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision.utils import save_image, make_grid

from utils import ReplayBuffer, LambdaLR, weights_init_normal, plot_and_save_losses
from dataset import get_dataloader
from cyclegan import Generator, Discriminator


def train(train_dataloader, test_dataloader, G_AB, G_BA, D_A, D_B, 
          criterion_GAN, criterion_cycle, criterion_identity,
          optimizer_G, optimizer_D_A, optimizer_D_B,
          fake_A_buffer, fake_B_buffer,
          device, epoch, args):
    """
    CycleGAN 학습 함수
    Args:
        train_dataloader: 학습 데이터를 로드하는 dataloader
        test_dataloader: 테스트 데이터를 로드하는 dataloader
        G_AB: A domain 이미지를 B domain 이미지로 변환하는 generator
        G_BA: B domain 이미지를 A domain 이미지로 변환하는 generator
        D_A: A domain 이미지의 진짜/가짜 여부를 판별하는 discriminator
        D_B: B domain 이미지의 진짜/가짜 여부를 판별하는 discriminator
        criterion_GAN: GAN loss 함수
        criterion_cycle: Cycle consistency loss 함수
        criterion_identity: Identity loss 함수
        optimizer_G, optimizer_D_A, optimizer_D_B: G, D 각각의 optimizer
        fake_A_buffer: Fake A에 대한 replay buffer
        fake_B_buffer: Fake B에 대한 replay buffer
        device: 학습에 사용할 디바이스 (CPU or GPU)
        epoch: 현재 epoch
        args: 기타 입력 인수들 (lambda 등)
    Returns:
        epoch_G_loss, epoch_D_loss: 매 epoch G, D loss
    """
    epoch_G_loss = 0.0
    epoch_D_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        start_time = time.time()
        
        # Real Image 로드
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        
        # Generator 학습 코드
        G_AB.train()
        G_BA.train()
        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_AB(real_A), real_A)
        loss_id_B = criterion_identity(G_BA(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2
        
        # GAN loss
        fake_B = G_AB(real_A)
        pred_fake = D_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        fake_A = G_BA(real_B)
        pred_fake = D_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_GAN = loss_GAN_A2B + loss_GAN_B2A

        # Cycle loss
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Generator 총 loss 계산 및 업데이트
        loss_G = loss_GAN + args.lambda_cycle * loss_cycle + args.lambda_identity * loss_identity
        loss_G.backward()
        optimizer_G.step()

        # Discriminator D_A 학습 코드
        optimizer_D_A.zero_grad()

        pred_real = D_A(real_A)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = D_A(fake_A_.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
        # Discriminator A 총 loss 계산 및 업데이트
        loss_D_A = (loss_D_real + loss_D_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        # Discriminator D_B 학습 코드
        optimizer_D_B.zero_grad()

        pred_real = D_B(real_B)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = D_B(fake_B_.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
        # Discriminator B 총 loss 계산 및 업데이트
        loss_D_B = (loss_D_real + loss_D_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()
        
        loss_D = loss_D_A + loss_D_B
        
        # G, D 누적 loss 계산
        epoch_G_loss += loss_G.item()
        epoch_D_loss += loss_D.item()
        
        # 학습 상태 로깅
        print(f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(train_dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}] [Elapsed time: {time.time() - start_time:.2f}]")
        
        # 일정 간격 iteration마다 이미지 저장
        if i % args.sample_interval == 0:
            G_AB.eval()
            G_BA.eval()
            imgs = next(iter(test_dataloader))
            real_A = imgs["A"].cuda()
            real_B = imgs["B"].cuda()
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            
            # 이미지 격자 생성
            real_A = make_grid(real_A, nrow=4, normalize=True)
            real_B = make_grid(real_B, nrow=4, normalize=True)
            fake_A = make_grid(fake_A, nrow=4, normalize=True)
            fake_B = make_grid(fake_B, nrow=4, normalize=True)
            
            # 이미지 1개로 통합하여 저장
            image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
            save_image(image_grid, os.path.join(args.output_dir, "outputs.png"), normalize=True)
    
    return epoch_G_loss, epoch_D_loss


def evaluate(test_dataloader, G_AB, G_BA, device, args):
    """
    CycleGAN 평가용 함수
    Args:
        test_dataloader: 테스트 데이터를 로드하는 dataloader
        G_AB: A domain 이미지를 B domain 이미지로 변환하는 generator
        G_BA: B domain 이미지를 A domain 이미지로 변환하는 generator
        device: 학습에 사용할 디바이스 (CPU or GPU)
        args: 기타 입력 인수들 (output_dir 등)
    """
    # Evaluation mode
    G_AB.eval()
    G_BA.eval()

    # Gradient 계산 생략
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # A domain 이미지로 B domain 이미지 생성 후 저장
            real_A = batch['A'].to(device)
            fake_B = G_AB(real_A)
            os.makedirs(os.path.join(args.output_dir, "fakeB"), exist_ok=True)
            save_image(fake_B, os.path.join(args.output_dir, "fakeB", f"fake_B_{i}.png"), normalize=True)
            
            # B domain 이미지로 A domain 이미지 생성 후 저장
            real_B = batch['B'].to(device)
            fake_A = G_BA(real_B)
            os.makedirs(os.path.join(args.output_dir, "fakeA"), exist_ok=True)
            save_image(fake_A, os.path.join(args.output_dir, "fakeA", f"fake_A_{i}.png"), normalize=True)

    
def save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, args):
    """
    현재 epoch에서의 CycleGAN model, optimizer의 체크포인트를 저장하는 함수
    Args:
        G_AB: A domain 이미지를 B domain 이미지로 변환하는 generator
        G_BA: B domain 이미지를 A domain 이미지로 변환하는 generator
        D_A: A domain 이미지의 진짜/가짜 여부를 판별하는 discriminator
        D_B: B domain 이미지의 진짜/가짜 여부를 판별하는 discriminator
        optimizer_G, optimizer_D_A, optimizer_D_B: G, D 각각의 optimizer
        args: 기타 입력 인수들 (output_dir 등)
    """
    state = {
        'epoch': epoch,
        'G_AB': G_AB.state_dict(),
        'G_BA': G_BA.state_dict(),
        'D_A': D_A.state_dict(),
        'D_B': D_B.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D_A': optimizer_D_A.state_dict(),
        'optimizer_D_B': optimizer_D_B.state_dict()
    }
    torch.save(state, os.path.join(args.output_dir, "checkpoint_latest.pth"))
    print(f"Saved checkpoint for epoch {epoch}")
    
    
def main(args):
    """
    CycleGAN 학습 및 평가를 위한 메인 함수
    Args:
        args: 입력 인수들
    """
    start_epoch = 0
    
    # Dataloader
    train_dataloader, test_dataloader = get_dataloader(args.data_path, image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers)

    # GPU 사용 여부 확인 및 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 초기화
    G_AB = Generator(input_nc=3).to(device).apply(weights_init_normal)
    G_BA = Generator(input_nc=3).to(device).apply(weights_init_normal)
    D_A = Discriminator(input_nc=3).to(device).apply(weights_init_normal)
    D_B = Discriminator(input_nc=3).to(device).apply(weights_init_normal)

    # Loss
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizer
    optimizer_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=args.lr, betas=args.opt_betas)
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=args.lr, betas=args.opt_betas)
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=args.lr, betas=args.opt_betas)
    
    # 학습률(learning rate) 스케줄러 초기화
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epochs, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.epochs, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.epochs, args.decay_epoch).step)
    
    # 체크포인트로부터 학습을 재개할 경우
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        G_AB.load_state_dict(checkpoint['G_AB'])
        G_BA.load_state_dict(checkpoint['G_BA'])
        D_A.load_state_dict(checkpoint['D_A'])
        D_B.load_state_dict(checkpoint['D_B'])
        
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
        optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
        
        start_epoch = checkpoint['epoch']
    
    # 평가 모드인 경우
    if args.eval:
        checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint_latest.pth"))
        G_AB.load_state_dict(checkpoint['G_AB'])
        G_BA.load_state_dict(checkpoint['G_BA'])
        del checkpoint
        
        evaluate(test_dataloader, G_AB, G_BA, device, args)
        print('Image results saved.')
        return
    
    # 생성된 이미지 데이터 일부를 포함할 버퍼(buffer)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    
    # 매 epoch loss 값 저장할 리스트
    G_losses = []
    D_losses = []

    # 학습
    for epoch in range(start_epoch, args.epochs):
        epoch_G_loss, epoch_D_loss = train(train_dataloader, test_dataloader, G_AB, G_BA, D_A, D_B,
                                           criterion_GAN, criterion_cycle, criterion_identity,
                                           optimizer_G, optimizer_D_A, optimizer_D_B,
                                           fake_A_buffer, fake_B_buffer,
                                           device, epoch, args)
        
        # 학습률(learning rate) 갱신
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        
        # 평균 loss 저장 및 그래프 생성
        G_losses.append(epoch_G_loss / len(train_dataloader))
        D_losses.append(epoch_D_loss / len(train_dataloader))
        plot_and_save_losses(G_losses, D_losses, save_path=os.path.join(args.output_dir, "loss_curves.png"))
        
        # 체크포인트 저장
        save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, args)

    # 모델 평가
    evaluate(test_dataloader, G_AB, G_BA, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CycleGAN training and evaluation script')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--image_size', default=256, type=int, help='images input size')
    parser.add_argument('--lambda_cycle', default=10, type=int, help='lambda of cycle consistency loss')
    parser.add_argument('--lambda_identity', default=5, type=int, help='lambda of identity loss')
    parser.add_argument('--sample_interval', default=50, type=int)

    parser.add_argument('--opt_betas', default=(0.5, 0.999), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--decay_epoch', default=100, type=int, help='epoch to start lr decay')

    parser.add_argument('--data_path', default='./horse2zebra', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./experiments', help='path where to save, empty for no saving')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)