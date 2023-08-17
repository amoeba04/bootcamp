import os
import time
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision.utils import save_image, make_grid

from utils import LambdaLR, weights_init_normal, plot_and_save_losses
from dataset import get_dataloader
from cyclegan import Generator, Discriminator


def train(train_dataloader, test_dataloader, G_AB, G_BA, D_A, D_B, 
          criterion_GAN, criterion_cycle, criterion_identity,
          optimizer_G, optimizer_D_A, optimizer_D_B, 
          lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B,
          device, epoch, args):
    '''Training for 1 epoch
    '''
    epoch_G_loss = 0.0
    epoch_D_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        start_time = time.time()
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        
        # Generators G_AB and G_BA
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

        loss_G = loss_GAN + args.lambda_cycle * loss_cycle + args.lambda_identity * loss_identity
        
        loss_G.backward()
        optimizer_G.step()

        # Discriminator D_A
        optimizer_D_A.zero_grad()

        pred_real = D_A(real_A)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        pred_fake = D_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D_A = (loss_D_real + loss_D_fake) / 2
        
        loss_D_A.backward()
        optimizer_D_A.step()

        # Discriminator D_B
        optimizer_D_B.zero_grad()

        pred_real = D_B(real_B)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        pred_fake = D_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D_B = (loss_D_real + loss_D_fake) / 2
        
        loss_D_B.backward()
        optimizer_D_B.step()
        
        loss_D = loss_D_A + loss_D_B
        
        # Accumulate losses for average computation
        epoch_G_loss += loss_G.item()
        epoch_D_loss += loss_D.item()
        
        # Logging
        print(f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(train_dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}] [Elapsed time: {time.time() - start_time:.2f}]")
        
        done = epoch * len(train_dataloader) + i
        if done % args.sample_interval == 0:
            G_AB.eval()
            G_BA.eval()
            imgs = next(iter(test_dataloader)) # 5개의 이미지를 추출해 생성
            real_A = imgs["A"].cuda()
            real_B = imgs["B"].cuda()
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            # X축을 따라 각각의 그리디 이미지 생성
            real_A = make_grid(real_A, nrow=4, normalize=True)
            real_B = make_grid(real_B, nrow=4, normalize=True)
            fake_A = make_grid(fake_A, nrow=4, normalize=True)
            fake_B = make_grid(fake_B, nrow=4, normalize=True)
            # 각각의 격자 이미지를 높이(height)를 기준으로 연결하기 
            image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
            save_image(image_grid, "outputs.png", normalize=True)
    
    return epoch_G_loss, epoch_D_loss


def evaluate(test_dataloader, G_AB, G_BA, device, args):
    """
    Evaluate the model on test data.
    Args:
    - test_dataloader: dataloader for test data.
    - G_AB, G_BA: Generators for transformations A->B and B->A respectively.
    - device: computation device (cpu or cuda)
    - args: input arguments
    """
    # Set the models to evaluation mode
    G_AB.eval()
    G_BA.eval()

    # Process and save the images from testA and testB
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            real_A = batch['A'].to(device)
            fake_B = G_AB(real_A)
            os.makedirs(os.path.join(args.output_dir, "genB"), exist_ok=True)
            save_image(fake_B, os.path.join(args.output_dir, "genB", f"fake_B_{i}.png"), normalize=True)
            
            real_B = batch['B'].to(device)
            fake_A = G_BA(real_B)
            os.makedirs(os.path.join(args.output_dir, "genA"), exist_ok=True)
            save_image(fake_A, os.path.join(args.output_dir, "genA", f"fake_A_{i}.png"), normalize=True)


def save_models(epoch, G_AB, G_BA, D_A, D_B, args):
    torch.save(G_AB.state_dict(), os.path.join(args.output_dir, "G_AB_latest.pth"))
    torch.save(G_BA.state_dict(), os.path.join(args.output_dir, "G_BA_latest.pth"))
    torch.save(D_A.state_dict(), os.path.join(args.output_dir, "D_A_latest.pth"))
    torch.save(D_B.state_dict(), os.path.join(args.output_dir, "D_B_latest.pth"))
    print(f"Saved models for epoch {epoch}")
    
def save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, args):
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
    start_epoch = 0
    # Example usage:
    train_dataloader, test_dataloader = get_dataloader(args.data_path, image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers)

    # Set device for computation (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the models
    G_AB = Generator(input_nc=3).to(device).apply(weights_init_normal)
    G_BA = Generator(input_nc=3).to(device).apply(weights_init_normal)
    D_A = Discriminator(input_nc=3).to(device).apply(weights_init_normal)
    D_B = Discriminator(input_nc=3).to(device).apply(weights_init_normal)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Create optimizers
    optimizer_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=args.lr, betas=args.opt_betas)
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=args.lr, betas=args.opt_betas)
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=args.lr, betas=args.opt_betas)
    
    # 학습률(learning rate) 업데이트 스케줄러 초기화
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epochs, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.epochs, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.epochs, args.decay_epoch).step)
    
    # Check if we should resume training from a checkpoint
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
    
    # Load saved models for evaluation
    if args.eval:
        checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint_latest.pth"))
        G_AB.load_state_dict(checkpoint['G_AB'])
        G_BA.load_state_dict(checkpoint['G_BA'])
        del checkpoint
        
        evaluate(test_dataloader, G_AB, G_BA, device, args)
        print('Image results saved.')
        return
    
    # Initialize lists to store average losses for each epoch
    G_losses = []
    D_losses = []

    # Training
    for epoch in range(start_epoch, args.epochs):
        epoch_G_loss, epoch_D_loss = train(train_dataloader, test_dataloader, G_AB, G_BA, D_A, D_B, criterion_GAN, criterion_cycle, criterion_identity,
            optimizer_G, optimizer_D_A, optimizer_D_B, lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B,
            device, epoch, args)
        
        # 학습률(learning rate)
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        
        # Append average losses for the epoch
        G_losses.append(epoch_G_loss / len(train_dataloader))
        D_losses.append(epoch_D_loss / len(train_dataloader))
        
        plot_and_save_losses(G_losses, D_losses, save_path=os.path.join(args.output_dir, "loss_curves.png"))
        
        save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, args)

    # Evaluate
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

    parser.add_argument('--data_path', default='/raid/jaesin/horse2zebra', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./experiments', help='path where to save, empty for no saving')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)