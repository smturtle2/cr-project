import os
# Removed unused import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from util import pytorch_ssim
from main_net import RPCA_Net_4_stages_SAR_Trans_RGB_8 as net
from dataLoader import SEN12MSCR_0  # Removed unused import SIM_MSCR
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from thop import profile
import time
from skimage.transform import resize
# Removed unused import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='CRTC Network Training/Inference')
    parser.add_argument(
        '--ckpt_dir',
        type=Path,
        default=Path('ckpts_SEN12MSCR/SEN12MSCR_CRTC_ckpts_4_trans_RGB_10iter_8'),
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '--N_iter',
        type=int,
        default=10,
        help='Number of unrolled iterations'
        # choices=range(1, 101)  # Reasonable range for iterations
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='Set GPU ID'
    )
    parser.add_argument(
        '--save_folder',
        type=Path,
        default=Path('SEN12MSCR_CRTC_ckpts_4_trans_RGB_10iter_8'),
        help='Path for saving test output images'
    )
    parser.add_argument(
        '--fig_mode',
        type=str,
        default='all',
        help='Mode for showing or saving figures: I, I+M, I+M+C, X_hat, all, none',
        choices=['I', 'I+M', 'I+M+C', 'X_hat', 'all', 'none']
    )
    parser.add_argument(
        '--test_mode',
        type=str,
        default='single',
        help='Testing mode: all or single',
        choices=['all', 'single']
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=200,
        help='Epoch number for single checkpoint testing'
    )
    parser.add_argument(
        '--data_mode',
        type=str,
        default='partial',
        help='Path to training data',
        choices=['partial', 'full']
    )
    parser.add_argument('--data_path', type=str, default='SEN12MS-CR_1', help='Path to data.')
    parser.add_argument(
        '--skip_epochs',
        type=int,
        default=150,
        help='Number of epochs to skip for testing'
    )
    parser.add_argument('--params', action='store_true', help='If true, get the model parameters.')
    args = parser.parse_args()
    
    # Validate checkpoint directory
    if not args.ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.ckpt_dir}")
        
    return args

def load_pretrained(path: Path, N_iter: int) -> torch.nn.Module:
    """
    Load a pretrained model from a checkpoint file.
    
    Args:
        path (Path): Path to the checkpoint file
        N_iter (int): Number of unrolled iterations
    
    Returns:
        torch.nn.Module: Loaded model
    """
    model = net(N_iter=N_iter)
    model = model.cuda()
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {path}")
    return model

def compute_metric(real_B, fake_B):
    rmse = torch.sqrt(torch.mean(torch.square(real_B - fake_B)))
    psnr = 20 * torch.log10(1 / rmse)
    mae = torch.mean(torch.abs(real_B - fake_B))
    
    # spectral angle mapper
    mat = real_B * fake_B
    mat = torch.sum(mat, 1)
    mat = torch.div(mat, torch.sqrt(torch.sum(real_B * real_B, 1)))
    mat = torch.div(mat, torch.sqrt(torch.sum(fake_B * fake_B, 1)))
    sam = torch.mean(torch.acos(torch.clamp(mat, -1, 1))*torch.tensor(180)/torch.pi)
    
    ssim = pytorch_ssim.ssim(real_B, fake_B)

    return {'RMSE': rmse.cpu().detach().numpy().item(),
            'MAE': mae.cpu().detach().numpy().item(),
            'PSNR': psnr.cpu().detach().numpy().item(),
            'SAM': sam.cpu().detach().numpy().item(),
            'SSIM': ssim.cpu().detach().numpy().item()}

def test(opt_args, ckpt_path, epoch_num):

    torch.backends.cudnn.benchmark = True

    data_path = opt_args.data_path
    sen12mscr_test       = SEN12MSCR_0(data_path, split='test', data_mode=opt_args.data_mode)
    data_test_loader      = torch.utils.data.DataLoader(sen12mscr_test, batch_size=1, shuffle=False, num_workers=4)
    
    img_dir = f'results_SEN12MSCR/{opt_args.save_folder}_{opt_args.fig_mode}'
    os.makedirs(img_dir, exist_ok=True)
    if opt_args.fig_mode != 'none':
        # Delete all files in img_dir
        file_list = glob.glob(img_dir + '/*')
        for file in file_list:
            os.remove(file)

    print('fig_mode:', opt_args.fig_mode)
    # metrics = {'RMSE': [], 'MAE': [], 'PSNR': [], 'SAM': [], 'SSIM': []}
    # metrics2 = {'RMSE': [], 'MAE': [], 'PSNR': [], 'SAM': [], 'SSIM': []}
    num_samples = len(data_test_loader)
    metrics_names = ['RMSE', 'MAE', 'PSNR', 'SAM', 'SSIM']
    metrics = np.zeros((num_samples, len(metrics_names)))
    metrics2 = np.zeros((num_samples, len(metrics_names)))
    with torch.cuda.device(opt_args.gpu):
        with torch.no_grad():
            model = load_pretrained(path=ckpt_path, N_iter=opt_args.N_iter)
            # Log model parameters
            # if opt_args.params:
            #     for name, param in model.named_parameters():
            #         logger.info(f"Layer: {name}, Params: {param.numel()}")
            #         if param.requires_grad:
            #             logger.info(f"  Trainable: Yes")
            #         else:
            #             logger.info(f"  Trainable: No")
            #     logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
            #Calculate FLOPs and params
            if opt_args.params:
                try:
                    
                    dummy_input = torch.randn(1, 13, 64, 64).cuda()
                    dummy_sar = torch.randn(1, 2, 64, 64).cuda()
                    dummy_trans = torch.randn(1, 3, 64, 64).cuda()
                    flops, params = profile(model, inputs=(dummy_input, dummy_sar, dummy_trans), verbose=False)
                    logger.info(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
                    logger.info(f"Params: {params / 1e3:.2f} K")
                except ImportError:
                    logger.warning("THOP not installed, skipping FLOPs and Params calculation.")
            model.eval()
            total_time = 0.0
            for i, (samples) in enumerate(tqdm(data_test_loader)):
                # if i == 100:
                #     break
                data = samples['input']
                sar = samples['sar']
                # omega = samples['input']['masks']
                target = samples['target']
                data = data.to(torch.float32)
                # omega = omega.to(torch.float32)
                target = target.to(torch.float32)
                data, target, sar = data.cuda(), target.cuda(), sar.cuda()
                trans = samples['trans'].cuda()
                img_path = os.path.join(img_dir, samples['filename'][0].replace('.npy', '.png'))

                tic = time.time()
                layers = model(data, sar, trans)
                toc = time.time()
                elapsed_time = (toc - tic) * 1000  # convert to milliseconds
                total_time += elapsed_time

                if i == 0:
                    # Record parameters for each layer to params.txt
                    with open(os.path.join(img_dir, f'params_{epoch_num}.txt'), 'a') as f:
                        f.write(f"Sample {samples['filename'][0]}:\n")
                        f.write(f" Epoch: {epoch_num}\n")
                        for layer_idx, layer in enumerate(layers):
                            lamb, gamma, mu, alpha, beta = layer[13], layer[14], layer[15], layer[16], layer[17]
                            f.write(f"Layer {layer_idx}:\n")
                            f.write(f"  lambda: {lamb.item():.4f}\n")
                            f.write(f"  gamma: {gamma.item():.4f}\n")
                            f.write(f"  mu: {mu.item():.4f}\n")
                            f.write(f"  alpha: {alpha.item():.4f}\n")
                            f.write(f"  beta: {beta.item():.4f}\n")
                        f.write("\n")
                    with open(os.path.join(img_dir, f'L012_{epoch_num}.txt'), 'a') as f:
                        f.write(f"Sample {samples['filename'][0]}:\n")
                        f.write(f" Epoch: {epoch_num}\n")
                        for layer_idx, layer in enumerate(layers):
                            Lamb, Gamma, Phi, S, T = layer[7], layer[8], layer[9], layer[10], layer[11]
                            f.write(f"Layer {layer_idx}:\n")
                            f.write(f"  Lambda: {Lamb.mean().item():.4f}\n")
                            f.write(f"  Gamma: {Gamma.mean().item():.4f}\n")
                            f.write(f"  Phi: {Phi.mean().item():.4f}\n")
                            f.write(f"  S: {S.mean().item():.4f}\n")
                            f.write(f"  T: {T.mean().item():.4f}\n")
                            # Save whole values
                            f.write(f"  Whole S: {S[0].cpu().detach().numpy()}\n")
                            f.write(f"  Whole T: {T[0].cpu().detach().numpy()}\n")
                            f.write(f"  Whole Lambda: {Lamb[0].cpu().detach().numpy()}\n")
                            f.write(f"  Whole Gamma: {Gamma[0].cpu().detach().numpy()}\n")
                            f.write(f"  Whole Phi: {Phi[0].cpu().detach().numpy()}\n")
                        f.write("\n")
                    # break
                if opt_args.fig_mode == 'I+M+C':
                    fig, ax = plt.subplots(4, len(layers)+1, figsize=(len(layers)*2, 6))

                    rgb_input = (data[0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    rgb_input = np.clip(rgb_input, 0, 255)
                    ax[0, 0].imshow(rgb_input)
                    ax[0, 0].set_title('Input Image')
                    ax[0, 0].axis('off')
                    ax[1, 0].imshow(np.zeros_like(rgb_input))
                    ax[1, 0].set_title('Initial Mask')
                    ax[1, 0].axis('off')
                    ax[2, 0].imshow(np.zeros_like(rgb_input))
                    ax[2, 0].set_title('Initial Cloud')
                    ax[2, 0].axis('off')
                    # ax[3, 0].imshow(np.zeros_like(rgb_input))
                    # ax[3, 0].set_title('Initial M*C')
                    # ax[3, 0].axis('off')
                    ax[3, 0].imshow(np.zeros_like(rgb_input))
                    ax[3, 0].set_title('Initial (1-M)*C')
                    ax[3, 0].axis('off')

                    for j in range(len(layers)):
                        # _, _, X, M, C, _, _, _, _, _ = layers[j]
                        X,M,C = layers[j][2], layers[j][3], layers[j][4]
                        img_k = torch.squeeze(X).cpu().detach().numpy()
                        mask_k = torch.squeeze(M).cpu().detach().numpy()
                        cloud_k = torch.squeeze(C).cpu().detach().numpy()
                        rgb_img = (img_k[[3,2,1], :, :].transpose(1, 2, 0) * 255 * 3).astype(np.uint8)
                        # rgb_img = (img_k[[3+2, 2+2, 1+2], :, :].transpose(1, 2, 0) * 255 *2).astype(np.uint8)
                        rgb_img = np.clip(rgb_img, 0, 255)
                        
                        C_component = (torch.ones(M[0,1].shape).cuda()-M[0, 1]) * C[0, 1]
                        C_component = (C_component - C_component.min()) / (C_component.max() - C_component.min())
                        C_component = (C_component.cpu().detach().numpy() * 255).astype(np.uint8)
                        C_component = np.clip(C_component, 0, 255)
                        
                        C_component2 = M[0, 1] * C[0, 1]
                        C_component2 = (C_component2 - C_component2.min()) / (C_component2.max() - C_component2.min())
                        C_component2 = (C_component2.cpu().detach().numpy() * 255).astype(np.uint8)
                        C_component2 = np.clip(C_component2, 0, 255)
                        # mask = (mask_k * 255).astype(np.uint8)
                        # mask = np.clip(mask, 0, 255)
                        ax[0, j+1].imshow(rgb_img)
                        ax[0, j+1].set_title(f'Stage {j} Output')
                        ax[0, j+1].axis('off')
                        ax[1, j+1].imshow(mask_k[1,:,:], cmap='gray')
                        ax[1, j+1].set_title(f'Stage {j} Mask')
                        ax[1, j+1].axis('off')
                        ax[2, j+1].imshow(cloud_k[1,:,:], cmap='gray')
                        ax[2, j+1].set_title(f'Stage {j} Cloud')
                        ax[2, j+1].axis('off')
                        # ax[3, j+1].imshow((mask_k * cloud_k)[1,:,:], cmap='gray')
                        # ax[3, j+1].set_title(f'Stage {j} M * C')
                        # ax[3, j+1].axis('off')
                        # ax[4, j+1].imshow(C_component, cmap='gray')
                        # ax[4, j+1].set_title(f'Stage {j} (1-M)*C')
                        ax[3, j+1].imshow(((torch.ones(mask_k.shape) - mask_k) * cloud_k)[1,:,:], cmap='gray')
                        ax[3, j+1].set_title(f'Stage {j} (1 - M) * C')
                        ax[3, j+1].axis('off')
                elif opt_args.fig_mode == 'I+M':
                    fig, ax = plt.subplots(3, len(layers)+1, figsize=(len(layers)*2, 4))
                    rgb_input = (data[0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0) * 255 * 2).astype(np.uint8)
                    rgb_input = np.clip(rgb_input, 0, 255)
                    ax[0, 0].imshow(rgb_input)
                    ax[0, 0].set_title('Input Image')
                    ax[0, 0].axis('off')
                    ax[1, 0].imshow(np.zeros_like(rgb_input))
                    ax[1, 0].set_title('Initial Mask')
                    ax[1, 0].axis('off')
                    ax[2, 0].imshow(np.zeros_like(rgb_input))
                    ax[2, 0].set_title('Initial H_k')
                    ax[2, 0].axis('off')

                    for j in range(len(layers)):
                        # _, _, X, M, C, _, _, _, _, _ = layers[j]
                        X,M,C,H = layers[j][2], layers[j][3], layers[j][4], layers[j][6]
                        img_k = torch.squeeze(X).cpu().detach().numpy()
                        mask_k = torch.squeeze(M).cpu().detach().numpy()
                        H_k = torch.squeeze(H).cpu().detach().numpy()
                        # rgb_img = (img_k[[3+2, 2+2, 1+2], :, :].transpose(1, 2, 0) * 255 *2).astype(np.uint8)
                        rgb_img = (img_k[[3,2,1], :, :].transpose(1, 2, 0) * 255 *2).astype(np.uint8)
                        rgb_img = np.clip(rgb_img, 0, 255)
                        mask = (mask_k * 255).astype(np.uint8)
                        mask = np.clip(mask, 0, 255)
                        H_k = (H_k * 255).astype(np.uint8)
                        H_k = np.clip(H_k, 0, 255)
                        ax[0, j+1].imshow(rgb_img)
                        ax[0, j+1].set_title(f'Stage {j} Image')
                        ax[0, j+1].axis('off')
                        ax[1, j+1].imshow(mask[0,:,:], cmap='gray')
                        ax[1, j+1].set_title(f'Stage {j} Mask')
                        ax[1, j+1].axis('off')
                        ax[2, j+1].imshow(H_k[0,:,:], cmap='gray')
                        ax[2, j+1].set_title(f'Stage {j} H_k')
                        ax[2, j+1].axis('off')
                elif opt_args.fig_mode == 'I':
                    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
                    rgb_input = (data[0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0) * 255 * 2).astype(np.uint8)
                    rgb_input = np.clip(rgb_input, 0, 255)
                    sar_img = (samples['sar'][0][0].cpu().detach().numpy() * 255).astype(np.uint8)
                    sar_img = np.clip(sar_img, 0, 255)
                    rgb_target = target[0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0)
                    # rgb_target = np.clip(rgb_target, 0, 255)
                    rgb_target = (rgb_target - rgb_target.min()) / (rgb_target.max() - rgb_target.min())
                    rgb_target = (rgb_target * 255).astype(np.uint8)

                    # rgb_img = (layers[-1][2][0, [3+2,2+2,1+2], :, :].cpu().detach().numpy().transpose(1, 2, 0) * 255 * 2).astype(np.uint8)
                    rgb_img = layers[-1][2][0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0)
                    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                    rgb_img = (rgb_img * 255).astype(np.uint8)
                    # rgb_img = np.clip(rgb_img, 0, 255)
                    
                    # Save rgb_img for all layers
                    rgb_imgs = []
                    for j in range(len(layers)):
                        img_k = torch.squeeze(layers[j][2]).cpu().detach().numpy()
                        # Calculate L1 error map between rgb_img_layer and rgb_target
                        # Resize rgb_target if needed to match rgb_img_layer
                        # Check shapes at each step
                        # print(f"target[0].shape: {target[0].shape}")
                        # print(f"layers[{j}][2][0].shape: {layers[j][2][0].shape}")

                        if target[0].shape != layers[j][2][0].shape:
                            target_resized = resize(
                                target[0].cpu().detach().numpy(),
                                layers[j][2][0].shape,
                                preserve_range=True,
                                anti_aliasing=True
                            ).astype(np.uint8)
                            # print(f"Resized target shape: {target_resized.shape}")
                        else:
                            target_resized = target[0].cpu().detach().numpy()
                            # print(f"Resized target shape: {target_resized.shape}")
                        # print(f"layers[{j}][2][0,[3,2,1],:,:].shape: {layers[j][2][0,[3,2,1],:,:].shape}")
                        # print(f"target_resized[[3,2,1],:,:].shape: {target_resized[[3,2,1],:,:].shape}")
                        l1_error = np.abs(layers[j][2][0,[3,2,1],:,:].cpu().detach().numpy().astype(np.float32) - target_resized[[3,2,1],:,:].astype(np.float32)).transpose(1, 2, 0)
                        # print(f"l1_error shape: {l1_error.shape}")
                        # average over channels to get a single channel error map
                        l1_error = l1_error.mean(axis=-1)
                        l1_error_norm = (l1_error - l1_error.min()) / (l1_error.ptp() + 1e-8) # Normalize to [0, 1]
                        l1_error_img = (plt.cm.jet(l1_error_norm)[..., :3] * 255).astype(np.uint8)
                        # print(f"l1_error_img shape: {l1_error_img.shape}")

                        # Paste 1/4th right-below corner of l1_error_img into rgb_img_layer
                        h, w, _ = layers[j][2][0].cpu().detach().numpy().transpose(1, 2, 0).shape
                        h4, w4 = h // 2, w // 2
                        rgb_img_with_patch = (layers[j][2][0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0).copy() * 255 * 3).astype(np.uint8)
                        # print(f"rgb_img_with_patch shape: {rgb_img_with_patch.shape}")
                        # Ensure l1_error_img has only 3 channels for RGB
                        rgb_img_with_patch[-h4:, -w4:, :] = l1_error_img[-h4:, -w4:, :3]
                        # print(f"Patch pasted: rgb_img_with_patch[-{h4}:, -{w4}:, :].shape = {rgb_img_with_patch[-h4:, -w4:, :].shape}")

                        # Save the new image
                        plt.imsave(img_path.replace('.png', f'_stage{j}_rgb_l1patch.png'), rgb_img_with_patch, dpi=300, vmin=0, vmax=255)
                        rgb_img_layer = (img_k[[3,2,1], :, :].transpose(1, 2, 0) * 255 * 3).astype(np.uint8)
                        rgb_img_layer = np.clip(rgb_img_layer, 0, 255)
                        rgb_imgs.append(rgb_img_layer)
                        plt.imsave(img_path.replace('.png', f'_stage{j}_rgb.png'), rgb_img_layer, dpi=300, vmin=0, vmax=255)
                    ax[0].imshow(rgb_input)
                    ax[0].set_title('Input Image')
                    ax[0].axis('off')
                    ax[1].imshow(sar_img, cmap='gray')
                    ax[1].set_title('SAR Image')
                    ax[1].axis('off')
                    ax[2].imshow(rgb_img)
                    ax[2].set_title('Output Image')
                    ax[2].axis('off')
                    ax[3].imshow(rgb_target)
                    ax[3].set_title('Groundtruth')
                    ax[3].axis('off')
                elif opt_args.fig_mode == 'all':
                    fig, ax = plt.subplots(1, 6, figsize=(21, 5))
                    rgb_input = (data[0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    rgb_input = np.clip(rgb_input, 0, 255)
                    sar_img = (samples['sar'][0][0].cpu().detach().numpy() * 255).astype(np.uint8)
                    sar_img = np.clip(sar_img, 0, 255)
                    rgb_target = (target[0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0) * 255 * 3).astype(np.uint8)
                    # rgb_target = (rgb_target - rgb_target.min()) / (rgb_target.max() - rgb_target.min())
                    # rgb_target = (rgb_target * 255).astype(np.uint8)
                    rgb_target = np.clip(rgb_target, 0, 255)
                    rgb_img = (layers[-1][2][0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0) * 255 * 3).astype(np.uint8)
                    # rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                    # rgb_img = (rgb_img * 255).astype(np.uint8)
                    rgb_img = np.clip(rgb_img, 0, 255)
                    # trans_min = trans.min().cpu().detach().numpy()
                    # trans_max = trans.max().cpu().detach().numpy()
                    # print(f'Trans Min: {trans_min}, Trans Max: {trans_max}')
                    trans_img = (trans[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    trans_img = np.clip(trans_img, 0, 255)
                    X, M, C, X_hat = layers[-1][2], layers[-1][3], layers[-1][4], layers[-1][12]
                    # mask_k = torch.squeeze(M).cpu().detach().numpy()
                    # cloud_k = torch.squeeze(C).cpu().detach().numpy()
                    rgb_X_hat = (X_hat[0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0)).astype(np.uint8)
                    # rgb_X_hat = np.stack([rgb_X_hat], axis=-1)
                    rgb_X_hat = np.clip(rgb_X_hat, 0, 1) * 255
                    # new_img = torch.where(M[0] < 0.5, data, X)
                    # rgb_new_img = (new_img[0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0) * 255 * 3).astype(np.uint8)
                    # rgb_new_img = np.clip(rgb_new_img, 0, 255)

                    # Visualize the (1-M) * C component in all the stages
                    fig0, ax0 = plt.subplots(1, len(layers), figsize=(len(layers) * 5.5, 5))
                    for j in range(len(layers)):
                        M, C = layers[j][3], layers[j][4]
                        # print(f'Stage {j} C min: {C.min().item()}, C max: {C.max().item()}')
                        C_component = (torch.ones(M[0,1].shape).cuda()-M[0, 1]) * C[0, 1]
                        C_component = (C_component - C_component.min()) / (C_component.max() - C_component.min())
                        C_component = (C_component.cpu().detach().numpy() * 255).astype(np.uint8)
                        C_component = np.clip(C_component, 0, 255)
                        
                        ax0[j].imshow(C_component, cmap='gray')
                        ax0[j].set_title(f'Stage {j} C_component')
                        ax0[j].axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(img_dir, img_path.replace('.png', '_C_stages.png')))
                    plt.close(fig0)

                    M, C = layers[-2][3], layers[-2][4]
                    M = M[0,1].cpu().detach().numpy()
                    # M = (M * 255).astype(np.uint8)
                    # M = np.clip(M, 0, 255)
                    C = C[0,1].cpu().detach().numpy()
                    # C = (C * 255).astype(np.uint8)
                    # C = np.clip(C, 0, 255)
                    plt.imsave(img_path.replace('.png', '_pred.png'), rgb_img, dpi=300, vmin=0, vmax=255)
                    plt.imsave(img_path.replace('.png', '_gt.png'), rgb_target, dpi=300, vmin=0, vmax=255)
                    plt.imsave(img_path.replace('.png', '_input.png'), rgb_input, dpi=300, vmin=0, vmax=255)
                    plt.imsave(img_path.replace('.png', '_trans.png'), trans_img, dpi=300, vmin=0, vmax=255)
                    # plt.imsave(img_path.replace('.png', '_pred0.png'), rgb_new_img, dpi=300, vmin=0, vmax=255)
                    plt.imsave(img_path.replace('.png', '_pred_X_hat.png'), rgb_X_hat, dpi=300, vmin=0, vmax=255)
                    plt.imsave(img_path.replace('.png', '_mask.png'), M, dpi=300, cmap='gray')
                    # plt.imsave(img_path.replace('.png', '_cloud.png'), C_component, dpi=300, cmap='gray')

                    ax[0].imshow(rgb_input)
                    ax[0].set_title('Input Image')
                    ax[0].axis('off')
                    ax[1].imshow(sar_img, cmap='gray')
                    ax[1].set_title('SAR Image')
                    ax[1].axis('off')
                    # ax[2].imshow(trans_img)
                    # ax[2].set_title('Trans Image')
                    # ax[2].axis('off')
                    # ax[3].imshow(rgb_X_hat)
                    # ax[3].set_title('X_hat Image')
                    # ax[3].axis('off')
                    ax[2].imshow(C, cmap='gray')
                    ax[2].set_title('Cloud Component')
                    ax[2].axis('off')
                    ax[3].imshow(M, cmap='gray')
                    ax[3].set_title('Cloud Mask')
                    ax[3].axis('off')
                    ax[4].imshow(rgb_img)
                    ax[4].set_title('Output Image')
                    ax[4].axis('off')
                    ax[5].imshow(rgb_target)
                    ax[5].set_title('Groundtruth')
                    ax[5].axis('off')
                elif opt_args.fig_mode == 'X_hat':
                    fig, ax = plt.subplots(3, 13, figsize=(64, 16))
                    X_hat = layers[-1][12]
                    rgb_X_hat = (X_hat[0].cpu().detach().numpy() * 255).astype(np.uint8)
                    rgb_X_hat = np.clip(rgb_X_hat, 0, 255)
                    rgb_target = (target[0].cpu().detach().numpy() * 255).astype(np.uint8)
                    rgb_target = np.clip(rgb_target, 0, 255)
                    rgb_output = (layers[-1][2][0].cpu().detach().numpy() * 255).astype(np.uint8)
                    rgb_output = np.clip(rgb_output, 0, 255)

                    for i in range(13):
                        ax[0, i].imshow(rgb_X_hat[i], cmap='gray')
                        ax[0, i].set_title(f'X_hat Channel {i+1}')
                        ax[0, i].axis('off')
                        ax[1, i].imshow(rgb_output[i], cmap='gray')
                        ax[1, i].set_title(f'Output Channel {i+1}')
                        ax[1, i].axis('off')
                        ax[2, i].imshow(rgb_target[i], cmap='gray')
                        ax[2, i].set_title(f'Target Channel {i+1}')
                        ax[2, i].axis('off')
                elif opt_args.fig_mode == 'none':
                    pass
                else:
                    raise ValueError('Invalid figure mode')

                if opt_args.fig_mode != 'none':
                    plt.tight_layout()
                    plt.savefig(img_path)
                    plt.close(fig)
                
                # Compute metrics

                if opt_args.fig_mode == 'X_hat':
                    X_hat = layers[-1][12]
                    output_X_hat = compute_metric(target, X_hat)
                    metrics2[i] = [output_X_hat[name] for name in metrics_names]
                # _, _, X, M, _, _, _, _, _, _, _, _ = layers[-1]
                X, M = layers[-1][2], layers[-1][3]
                output = compute_metric(target, X)
                # _, _, _, M, _, _, _, _, _, _, _, _ = layers[-3]
                # new_img = torch.where(M[0] < 0.5, data, X)
                # output2 = compute_metric(target, new_img)

                # for key, value in output.items():
                #     metrics[key].append(value)
                metrics[i] = [output[name] for name in metrics_names]
                # for key, value in output2.items():
                #     metrics2[key].append(value)
            plt.figure(figsize=(10, 6))
            plt.boxplot(metrics[2], vert=True, flierprops=dict(marker='o', color='r', alpha=0.5))
            plt.title('PSNR Box Plot')
            plt.xlabel('PSNR')
            plt.savefig(os.path.join(img_dir, 'psnr_boxplot.png'))
            plt.close()
            # Get the file list of PSNR over 30 and print them out
            # psnr_over_30_files = [samples['filename'][0] for i, samples in enumerate(data_test_loader) if metrics['PSNR'][i] > 30]
            # print(f'Files with PSNR over 30: {psnr_over_30_files}')
            # average_metrics = {key: round(np.mean(values), 4) for key, values in metrics.items()}
            average_metrics = np.round(np.mean(metrics, axis=0), 4)
            
            # average_metrics2 = {key: round(np.mean(values), 4) for key, values in metrics2.items()}
            average_metrics_dict = {name: average_metrics[idx] for idx, name in enumerate(metrics_names)}
            if opt_args.fig_mode == 'X_hat':
                average_metrics2 = np.round(np.mean(metrics2, axis=0), 4)
                average_metrics2_dict = {name: average_metrics2[idx] for idx, name in enumerate(metrics_names)}
                print(f'Average Metrics X_hat for {ckpt_path}:', average_metrics2_dict)
            print(f'Average Metrics for {ckpt_path}:', average_metrics_dict)
            # print(f'Average Metrics for {ckpt_path}:', average_metrics)
            # print(f'Average Metrics2 for {ckpt_path}:', average_metrics2)
            print(f'Average time per image: {total_time / num_samples:.2f} ms')
            return average_metrics_dict

if __name__ == '__main__':
    opt_args = parse_args()
    if opt_args.test_mode == 'all':
        best_metrics = None
        best_ckpt = None
        best_SAM_metrics = None
        best_SAM_ckpt = None
        for ckpt_path in opt_args.ckpt_dir.glob('*.pth'):
            # Extract epoch number from filename
            epoch_num = int(ckpt_path.stem.split('_')[1])
            if epoch_num < opt_args.skip_epochs:  # Skip first 100 epochs
                continue
            metrics = test(opt_args, ckpt_path, epoch_num)
            if best_metrics is None or metrics['PSNR'] > best_metrics['PSNR']:
                best_metrics = metrics 
                best_ckpt = ckpt_path
            if best_SAM_metrics is None or metrics['SAM'] < best_SAM_metrics['SAM']:
                best_SAM_metrics = metrics
                best_SAM_ckpt = ckpt_path
        print(f'Best checkpoint: {best_ckpt}')
        print(f'Best metrics: {best_metrics}')
        print(f'Best SAM checkpoint: {best_SAM_ckpt}')
        print(f'Best SAM metrics: {best_SAM_metrics}')
    elif opt_args.test_mode == 'single':
        if opt_args.epoch is None:
            raise ValueError('Epoch number must be specified for single checkpoint testing mode')
        ckpt_path = opt_args.ckpt_dir / f'epoch_{opt_args.epoch}.pth'
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint file not found: {ckpt_path}')
        metrics = test(opt_args, ckpt_path, opt_args.epoch)
        # print(f'Metrics for checkpoint {ckpt_path}:', metrics)
        # print(f'Metrics2 for checkpoint {ckpt_path}:', metrics2)
    else:
        raise ValueError('Invalid test mode')