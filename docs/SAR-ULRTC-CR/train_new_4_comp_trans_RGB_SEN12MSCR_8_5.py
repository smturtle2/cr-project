import os
import torch
import argparse
from torch.utils.data import Subset, DataLoader
from torch import mean
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from torchinfo import summary
from main_net import RPCA_Net_4_stages_SAR_Trans_RGB_8 as net
from dataLoader import SIM_MSCR, SEN12MSCR_0
import numpy as np
from util.pytorch_ssim import ssim
import visdom
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='ckpts_SEN12MSCR/SEN12MSCR_CRTC_ckpts_4_trans_RGB_10iter_8', help='Path for checkpointing.') # 1
    parser.add_argument('--resume', help='Resume training from saved checkpoint(s).')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Checkpoint model every x epochs.')
    parser.add_argument('--loss_freq', type=int, default=10, help='Report (average) loss every x iterations.')
    parser.add_argument('--N_iter', type=int, default=10, help='Number of unrolled iterations.')
    parser.add_argument('--set_lr', type=float, default=-1, help='Set new learning rate.')
    parser.add_argument('--decay', type=int, default=2, help='Set seed value.')
    parser.add_argument('--data_mode', type=str, default='partial', help='dataset mode: full or partial')
    parser.add_argument('--train_path', type=str, default='/media/ssd1/chuong/datasets/SEN12MS-CR', help='Path to training data.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for Visdom server.')
    parser.add_argument('--visdom_env', type=str, default='cloud_7', help='Environment for Visdom server.')
    parser.add_argument('--vis_list', type=str, default='none', help='List of images to visualize.')

    return parser.parse_args()

def plot_learning_curve(vis, losses, val_losses, epoch):
    vis.line(X=np.arange(1, len(losses) + 1), Y=np.array(losses), win='train_loss', opts=dict(title='Training Loss', xlabel='Epoch', ylabel='Loss'))
    vis.line(X=np.arange(1, len(val_losses) + 1), Y=np.array(val_losses), win='val_loss', opts=dict(title='Validation Loss', xlabel='Epoch', ylabel='Loss'))

def plot_learning_curve_2(opt, losses, val_losses, epoch):
    figure, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, len(losses) + 1), np.array(losses), label='Training Loss')
    ax.plot(np.arange(1, len(val_losses) + 1), np.array(val_losses), label='Validation Loss', color='orange')
    ax.set_title('Learning Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(opt.save_path, f'learning_curve.png'))
    plt.close(figure)

def train(opt):
    # visdom_env = 'cloud_0'
    # vis = visdom.Visdom(env=opt.visdom_env, port=opt.visdom_port)
    vis = None
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(net(N_iter=opt.N_iter).cuda())
    else:
        model = net(N_iter=opt.N_iter).cuda()
    
    torch.backends.cudnn.benchmark = True
    simmscr_train = SEN12MSCR_0(opt.train_path, split='train', data_mode=opt.data_mode)
    
    seed = 42
    torch.manual_seed(seed)
    
    if opt.data_mode == 'full':
        data_train_loader = DataLoader(simmscr_train, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        simmscr_val = SEN12MSCR_0(opt.train_path, split='val')
        data_val_loader = DataLoader(simmscr_val, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    elif opt.data_mode == 'partial':
        subset_indices = torch.randperm(len(simmscr_train))[:5000]
        torch.save(subset_indices, './seed/train_5k_subset_indices.pth')
        simmscr_train_subset = Subset(simmscr_train, subset_indices)
        data_train_loader = DataLoader(simmscr_train_subset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        simmscr_val = SEN12MSCR_0(opt.train_path, split='val')
        data_val_loader = DataLoader(simmscr_val, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    elif opt.data_mode == 'partial_fixed':
        # subset_indices = torch.load('./seed/train_5k_subset_indices.pth')
        # simmscr_train_subset = Subset(simmscr_train, subset_indices)
        data_train_loader = DataLoader(simmscr_train, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        simmscr_val = SEN12MSCR_0(opt.train_path, split='val')
        data_val_loader = DataLoader(simmscr_val, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        raise ValueError('data_mode should be either full or partial')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # 1e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
    loss_fn = torch.nn.L1Loss()
    def loss_fn2(x, y):
        # SSIM returns a value between 0 and 1, higher is better, so we use (1 - ssim) as loss
        return 1 - ssim(x, y)
    # summary(model, input_size=[(1, 13, 256, 256), (1, 2, 256, 256), (1, 3, 256, 256)])

    parameters = [round(2 * k / (opt.N_iter * (opt.N_iter + 1)), 4) for k in range(1, opt.N_iter + 1)]
    print('Loss component parameters:', parameters)
    
    losses, val_losses = [], []
    
    if opt.resume:
        print('Resume training from', opt.resume)
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_0 = checkpoint['epoch'] + 1
        model.train()
        log_path = os.path.join(opt.save_path, 'training_log.txt')
        if os.path.exists(log_path):
            with open(log_path, 'r') as log_file:
                for line in log_file:
                    if 'Epoch' in line:
                        epoch_logged = int(line.split('Epoch: ')[1].split(',')[0])
                        if epoch_logged < epoch_0:
                            losses.append(float(line.split('Training Loss: ')[1].split(',')[0]))
                            val_losses.append(float(line.split('Validation Loss: ')[1]))
    else:
        print('Start training from scratch.')
        epoch_0 = 1

    if opt.set_lr != -1:
        for group in optimizer.param_groups:
            group['lr'] = opt.set_lr
        print('New learning rate:', opt.set_lr)

    os.makedirs(opt.save_path, exist_ok=True)

    for epoch in range(epoch_0, opt.num_epoch + 1):
        model.train()
        train_loss = 0
        progress_bar = tqdm(enumerate(data_train_loader), total=len(data_train_loader))
        for i, samples in progress_bar:
            data, target = samples['input'].cuda(), samples['target'].cuda()
            sar = samples['sar'].cuda()
            trans = samples['trans'].cuda()
            layers = model(data, sar, trans)
            total_loss = sum(parameters[j] * (loss_fn(layers[j][2],  target) + 1.0 * loss_fn2(layers[j][2], target)) for j in range(len(layers)))
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            train_loss += total_loss.item()

            if (i + 1) % opt.loss_freq == 0:
                progress_bar.set_postfix({'Epoch': epoch, 'Loss': f'{train_loss / (i + 1):.2e}'})
            if i % 100 == 0 and opt.vis_list == 'all' and vis != None:
                vis.images(np.clip(data[0,[3,2,1],:,:].cpu().numpy() *255*2, 0, 255), nrow=4, win='input', opts=dict(title='Input'))
                vis.images(np.clip(sar[0][0].cpu().numpy() *255, 0, 255), nrow=4, win='sar', opts=dict(title='SAR'))
                vis.images(np.clip(trans[0].cpu().numpy() *255, 0, 255), nrow=4, win='trans', opts=dict(title='Trans'))
                vis.images(np.clip(target[0,[3,2,1],:,:].cpu().numpy() *255*2, 0, 255), nrow=4, win='target', opts=dict(title='Target'))
                vis.images(np.clip(layers[-1][3][0][1].detach().cpu().numpy() *255, 0, 255), nrow=4, win='cloud_mask', opts=dict(title='Cloud_mask'))
                vis.images(np.clip(layers[-1][6][0][0].detach().cpu().numpy() *255, 0, 255), nrow=4, win='H', opts=dict(title='H_K'))
                vis.images(np.clip(layers[-1][5][0][1].detach().cpu().numpy() *255, 0, 255), nrow=4, win='cloud', opts=dict(title='Cloud'))
                vis.images(np.clip(layers[-1][2][0,[3,2,1],:,:].detach().cpu().numpy() *255*2, 0, 255), nrow=4, win='output', opts=dict(title='Output'))
        losses.append(train_loss / len(data_train_loader))

        if epoch % opt.checkpoint_freq == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(opt.save_path, f'epoch_{epoch}.pth'))
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_samples in tqdm(data_val_loader, desc=f'Validation Epoch {epoch}'):
                val_data, val_target = val_samples['input'].cuda(), val_samples['target'].cuda()
                val_sar = val_samples['sar'].cuda()
                val_trans = val_samples['trans'].cuda()
                val_layers = model(val_data, val_sar, val_trans)
                val_total_loss = sum(parameters[k] * (loss_fn(val_layers[k][2], val_target) + 1.0 * loss_fn2(val_layers[k][2], val_target)) for k in range(len(val_layers)))
                val_loss += val_total_loss.item()

        val_losses.append(val_loss / len(data_val_loader))
        # plot_learning_curve(vis, losses, val_losses, epoch)
        plot_learning_curve_2(opt, losses, val_losses, epoch)
        # Save training log
        with open(os.path.join(opt.save_path, 'training_log.txt'), 'a') as log_file:
            log_file.write(f'Epoch: {epoch}, Training Loss: {train_loss / len(data_train_loader):.2e}, Validation Loss: {val_loss / len(data_val_loader):.2e}\n')
        
        scheduler.step()

if __name__ == '__main__':
    opt_args = parse_args()
    train(opt_args)
