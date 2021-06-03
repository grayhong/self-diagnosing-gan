import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata


def to_numpy_image(x):
    # convert torch tensor [-1,1] to numpy image [0,255]
    x = x.cpu().numpy().transpose(0, 2, 3, 1)  # C x H x W -> H x W x C
    x = ((x + 1) / 2).clip(0, 1)  # [-1,1] -> [0,1]
    x = (x * 255).astype(np.uint8)  # uint8 numpy image
    return x

def plot_data(x, num_per_side=10, save_path=None, file_name='plot_data', is_tensor=True, vis=None):
    if is_tensor:
        x = to_numpy_image(x)
    _, h, w, c = x.shape
    img = np.empty((h * num_per_side, w * num_per_side, c))

    for i in range(num_per_side):
        for j in range(num_per_side):
            img[i*w : (i+1)*w, j*h : (j+1)*h, :] = x[i*num_per_side + j]
    img = img.astype(np.uint8)
    fig = plt.figure()

    plt.rcParams['figure.figsize'] = 20,20
    plt.rcParams['axes.linewidth'] = 1
    # fig.suptitle(file_name, fontsize=10)
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    #fig.suptitle(file_name, fontsize=10)


    plt.imshow(img)
    if vis:
        vis.matplot(plt)
    plt.savefig(save_path / f'{file_name}.jpg', bbox_inches='tight', pad_inches = 0)
    plt.close()

def plot_imbalance_dataset_samples(dataset, save_path, filename):
    categories = {'major': 0, 'minor': 1}
    num_imgs = 50
    for name, cls in categories.items():
        imgs = torch.stack([dataset.__getitem__(i)[0] for i in np.where(dataset.labels == cls)[0][:num_imgs]], 0)
        show_images_grid(imgs[:num_imgs], num_imgs, save_path, f'{filename}_{name}')

def gaussian_plot(x, path):
    plt.rcParams['figure.figsize'] = 8,8
    plt.rcParams['font.size'] = 20
    x = x.cpu().numpy()
    for i in range(len(x)):
        plt.plot(x[i][0], x[i][1], 'go', alpha=0.2, label='generated')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axis('off')
    plt.box()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def show_images_grid(x, num_images=25, save_path=None, file_name='show_images_grid', is_tensor=True, vis=None):
    from torchvision import utils as vutils
    assert x.shape[0] >= num_images
    x = x[:num_images]
    images_viz = vutils.make_grid(
        x,
        nrow=10,
        padding=2,
        normalize=True)

    vutils.save_image(images_viz,
                    save_path / f'{file_name}.jpg',
                    normalize=True)

def plot_from_generator(netG, netD, nz=128, save_path=None, file_name='plot_generator', vis=None):
    grid_size = 49
    with torch.no_grad():
        num_data = 10000
        test_z = torch.randn(num_data, nz).to('cuda')
        test_imgs = netG(test_z).detach().cpu()
        plot_data(test_imgs, num_per_side=20, save_path=save_path, file_name=f'{file_name}_all', vis=vis)
        show_images_grid(test_imgs, num_images=grid_size, save_path=save_path, file_name=f'{file_name}')


def show_sorted_score_samples(dataset, score, save_path=None, score_name='ae_score', plot_name='ae_score'):
    plt.rcParams['figure.figsize'] = 16,8
    plt.rcParams['font.size'] = 20
    plt.xlabel('index', fontsize=20)
    plt.ylabel(score_name, fontsize=20)

    sorted_idx = np.argsort(score)
    for range_name, rng in [('low100', sorted_idx[:20]), ('high100', sorted_idx[-20:])]:
        imgs = torch.stack([dataset.__getitem__(i)[0] for i in rng], 0)
        # plot_data(imgs, num_per_side=10, is_tensor=True, save_path=save_path, file_name=f'{plot_name}_{range_name}')
        show_images_grid(imgs, num_images=20, is_tensor=True, save_path=save_path, file_name=f'{plot_name}_{range_name}')

    
def print_num_params(netG, netD):
    gen_trainable_parameters = sum([p.data.nelement() for p in netG.parameters()])
    disc_trainable_parameters = sum([p.data.nelement() for p in netD.parameters()])
    print(f'gen_trainable_parameters: {gen_trainable_parameters}, disc_trainable_parameters: {disc_trainable_parameters}')


def LDR_plot(ldr_list, start, end, save_path):
    plt.rcParams['figure.figsize'] = 16,8
    plt.rcParams['font.size'] = 20
    plt.plot(range(start*100, (end+1)*100, 100), ldr_list)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def LDR_cont_plot(LDR_dict, start, end, output_dir, mode):
    sort_pivot = -1

    epochs = end - start + 1

    LDR_arr = np.array([LDR_dict[idx][start-1:end] for idx in LDR_dict])
    ind = np.argsort(LDR_arr[:,sort_pivot])
    sorted_LDR_arr = LDR_arr[ind]

    rank_LDR_arr = sorted_LDR_arr.copy()
    for i in range(epochs):
        rank_LDR_arr[:,i] = rankdata(rank_LDR_arr[:,i])
        
    plt.pcolor (sorted_LDR_arr, cmap='coolwarm')
    plt.xlabel('epoch')
    plt.ylabel('data point')
    plt.title('LDR plot')
    plt.colorbar()
    fig_path = os.path.join(output_dir, 'images', f'ldr_{mode}_cont.png')
    plt.savefig(fig_path)
    plt.clf()

    plt.pcolor (rank_LDR_arr, cmap='coolwarm')
    plt.xlabel('epoch')
    plt.ylabel('data point')
    plt.title('LDR rank plot')
    plt.colorbar()
    fig_path = os.path.join(output_dir, 'images', f'ldr_{mode}_rank.png')
    plt.savefig(fig_path)
    plt.clf()


def plot_score_sort(dataset, score_dict, save_path, phase, plot_metric_name=None):
    n_data = len(dataset)
    n_plt = min(5000, n_data)
    plot_idx = np.sort(np.random.choice(n_data, n_plt, replace=False))

    for metric_name, metric in score_dict.items():
        if plot_metric_name and plot_metric_name not in metric_name:
            continue
        print(f'plot_score_sort: {metric_name}')
        plt.rcParams['figure.figsize'] = 16,8
        plt.rcParams['font.size'] = 20
        plt.xlabel('index', fontsize=20)
        plt.ylabel(metric_name, fontsize=20)

        sorted_idx = np.argsort(metric)[plot_idx]
        sorted_ldrd = metric[sorted_idx]
        sorted_type = np.array(dataset.dataset.labels)[sorted_idx]

        for i, color in enumerate(['blue', 'red']):
            plt.bar(np.arange(n_plt)[sorted_type == i], sorted_ldrd[sorted_type == i], color=color)
        plt.savefig(save_path / f'{phase}_{metric_name}_sort.jpg')
        plt.close()

def plot_score_box(dataset, score_dict, save_path, phase, plot_metric_name=None, class_name=None):
    n_data = len(dataset)
    n_plt = min(500000, n_data)
    plot_idx = np.sort(np.random.choice(n_data, n_plt, replace=False))

    plot_name_dict = {
        'ldrv' : 'LDRV',
        'ldrm' : 'LDRM'
    }

    for metric_name, metric in score_dict.items():
        if plot_metric_name and plot_metric_name not in metric_name:
            continue
        if metric_name not in ['ldrv', 'ldrm']:
            continue
        print(f'plot_score_box: {metric_name}')

        boxprops = dict(linewidth=3)
        whiskerprops = dict(linewidth=3)
        medianprops = dict(linewidth=3)

        plt.rcParams['figure.figsize'] = 7, 12
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.linewidth'] = 2

        if metric_name in plot_name_dict:
            plt.ylabel(plot_name_dict[metric_name], fontsize=50)
        else:
            plt.ylabel(metric_name, fontsize=50)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=25)

        sorted_idx = np.argsort(metric)[plot_idx]
        sorted_ldrd = metric[sorted_idx]
        sorted_type = np.array(dataset.dataset.labels)[sorted_idx]
        
        plt.boxplot([sorted_ldrd[sorted_type == i] for i in range(2)], 
                    boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops)
        if class_name:
            plt.xticks([1,2],[class_name[0], class_name[1]])
            
        plt.savefig(save_path / f'{phase}_{metric_name}_box.jpg', bbox_inches='tight')
        plt.close()

def calculate_scores(logits, start_epoch=50, end_epoch=75, clip_val=1.5, conf=1):
    def clip_value(score, clip_val):
        upper_bd =  np.mean(score) + clip_val * np.std(score)
        lower_bd =  np.mean(score) - clip_val * np.std(score)
        return np.clip(score, lower_bd, upper_bd)

    def clip_max_ratio(score, ratio=20):
        upper_bd = score.min() * ratio
        return np.clip(score, None, upper_bd)

    def clip_min(score, lower_bd=1e-2):
        return np.clip(score, a_min=lower_bd, a_max=None)

    def l2_norm(score1, score2):
        score1 = score1 / score1.mean()
        score2 = score2 / score2.mean()
        score = np.sqrt(score1 ** 2 + score2 ** 2)
        return score / score.mean()

    logits_arr = np.array([v for k, v in logits.items() if k >= start_epoch and k < end_epoch])
    print(f'calculate_scores -- start_epoch: {start_epoch} end_epoch: {end_epoch} logits_arr: {logits_arr.shape}')
    num_data = logits_arr.shape[0] 
    score_dict = dict()
    score_dict['ldr'] = logits_arr[-1]
    score_dict['ldrd'] = np.abs(logits_arr[1:] - logits_arr[:-1]).mean(0)
    score_dict['ldrv'] = np.var(logits_arr, axis=0, ddof=1)
    score_dict['ldrm'] = logits_arr.mean(0)
    for t in np.arange(0.1, 10.0, 0.1):
        score_dict[f'ldr_conf_{t:.1f}_ratio_50'] = clip_max_ratio(clip_min((logits_arr.mean(0) + t * np.std(logits_arr, 0, ddof=1))), ratio=50)
    return score_dict

def plot_intensity_histogram(sample_weights, dataset, generated_images, save_path, prefix):
    plt.figure(figsize=(10, 7.25))

    sorted_idx = np.argsort(sample_weights)
    low_histogram = np.array([Image.fromarray(to_numpy_image(dataset.__getitem__(i)[0].unsqueeze(0)).squeeze(0)).histogram() for i in sorted_idx[:100]]).sum(0)
    low_plt, = plt.plot(low_histogram, color='blue', label='Low Scored Samples')

    high_histogram = np.array([Image.fromarray(to_numpy_image(dataset.__getitem__(i)[0].unsqueeze(0)).squeeze(0)).histogram() for i in sorted_idx[-100:]]).sum(0)
    high_plt, = plt.plot(high_histogram, color='red', label='High Scored Samples')

    generated_histogram = np.array([Image.fromarray(to_numpy_image(i.unsqueeze(0)).squeeze(0)).histogram() for i in generated_images]).sum(0)
    generated_plt, = plt.plot(generated_histogram, color='green', label='Generated Samples')
    plt.legend(fontsize='10', loc='upper left', prop={'size': 12})
    plt.xlabel('Intensity', fontsize=20)
    plt.ylabel('Pixel count', fontsize=20)

    plt.savefig(save_path / f'{prefix}_histogram.jpg', bbox_inches = 'tight', pad_inches = 0.1)

def plot_color_mnist_generator(netG, save_path=None, file_name='plot_generator'):
    print(f'plot_color_mnist_generator file_name: {file_name}')

    grid_size = 6
    netG.eval()
    with torch.no_grad():
        num_data = 10000
        test_imgs = netG.generate_images(num_data, 'cuda').detach().cpu()

        show_images_grid(test_imgs, num_images=100, save_path=save_path, file_name=f'{file_name}_all')
        green_channel_maxs = (test_imgs[:,1,:,:].view(num_data, -1) > 0).sum(-1)
        red_channel_maxs = (test_imgs[:,0,:,:].view(num_data, -1) > 0).sum(-1)
        green_quantile = np.quantile(green_channel_maxs[green_channel_maxs > 0], 0.75)
        red_quantile = np.quantile(red_channel_maxs[red_channel_maxs > 0], 0.75)
        
        print(green_quantile)
        print(red_quantile)
        sorted_green_test_imgs = test_imgs[torch.argsort(green_channel_maxs, -1, True)]
        sorted_red_test_imgs = test_imgs[torch.argsort(red_channel_maxs, -1, True)]
        plot_dict = dict()
        num_greens = sum(green_channel_maxs > 0).item()
        green_bdry = int(num_greens / 4)
        print(num_greens)
        num_reds = sum(red_channel_maxs > 0).item()
        red_bdry = int(num_reds / 4)
        print(num_reds)
        """
        green_over_quantile = test_imgs[green_channel_maxs >= green_quantile]
        print(len(green_over_quantile))
        red_over_quantile = test_imgs[red_channel_maxs >= red_quantile]
        print(len(red_over_quantile))
        """
        if green_bdry >= grid_size:
            plot_dict['green'] = sorted_green_test_imgs[:green_bdry][torch.from_numpy(np.random.choice(green_bdry, grid_size, replace=False)).cuda()]
        # if len(green_over_quantile) >= grid_size:
        #     plot_dict['green'] = green_over_quantile[np.random.choice(len(green_over_quantile), grid_size, replace=False)]
        
        # if num_data - num_greens >= grid_size:
            # plot_dict['red'] = sorted_test_imgs[num_greens:][torch.from_numpy(np.random.choice(num_data - num_greens, grid_size)).cuda()]
        if red_bdry >= grid_size:
            plot_dict['red'] = sorted_red_test_imgs[:red_bdry][torch.from_numpy(np.random.choice(red_bdry, grid_size, replace=False)).cuda()]
        # if len(red_over_quantile) >= grid_size:
        #     plot_dict['red'] = red_over_quantile[np.random.choice(len(red_over_quantile), grid_size, replace=False)]


    for name, data in plot_dict.items():
        # plot_data(data, num_per_side=7, save_path=save_path, file_name=f'{file_name}_{name}')
        show_images_grid(data, num_images=grid_size, save_path=save_path, file_name=f'{file_name}_{name}')
        # show_images_grid(data, num_images=20, save_path=save_path, file_name=f'{file_name}_{name}')
    netG.train()


def plot_mnist_fmnist_generator(netG, save_path=None, file_name='plot_generator'):
    print(f'plot_color_mnist_generator file_name: {file_name}')
    # grid_size = 49
    netG.eval()
    with torch.no_grad():
        num_data = 10000
        batch_size = 100
        test_imgs = None
        for i in range(num_data // batch_size):
            tmp_imgs = netG.generate_images(batch_size, 'cuda').detach().cpu()
            if test_imgs is None:
                test_imgs = tmp_imgs
            else:
                test_imgs = torch.cat((test_imgs, tmp_imgs), 0)
        # test_imgs = netG.generate_images(num_data, 'cuda').detach().cpu()
        # plot_data(test_imgs, num_per_side=10, save_path=save_path, file_name=f'{file_name}_all_100')
        show_images_grid(test_imgs, num_images=100, save_path=save_path, file_name=f'{file_name}_all')
    netG.train()


def _get_fixed_noise(log_dir, nz, num_samples, device, output_dir=None):
    """
    Produce the fixed gaussian noise vectors used across all models
    for consistency.
    """
    if output_dir is None:
        output_dir = os.path.join(log_dir, 'viz')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir,
                                'fixed_noise_nz_{}.pth'.format(nz))

    if os.path.exists(output_file):
        noise = torch.load(output_file)

    else:
        noise = torch.randn((num_samples, nz))
        torch.save(noise, output_file)

    return noise.to(device)


def plot_gaussian_samples(netG, global_step, log_dir, device, num_samples=10000):
    """
    Produce visualisations of the G(z), one fixed and one random.

    Args:
        netG (Module): Generator model object for producing images.
        global_step (int): Global step variable for syncing logs.
        num_samples (int): The number of samples to visualise.

    Returns:
        None
    """
    img_dir = os.path.join(log_dir, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    with torch.no_grad():
        # Generate random images
        noise = torch.randn((num_samples, netG.nz), device=device)
        fake_images = netG(noise).detach().cpu()

        # Generate fixed random images
        fixed_noise = _get_fixed_noise(log_dir=log_dir, nz=netG.nz,
                                        num_samples=num_samples, device=device)

        if hasattr(netG, 'num_classes') and netG.num_classes > 0:
            fixed_labels = _get_fixed_labels(num_samples,
                                                    netG.num_classes)
            fixed_fake_images = netG(fixed_noise,
                                        fixed_labels).detach().cpu()
        else:
            fixed_fake_images = netG(fixed_noise).detach().cpu()

        # Map name to results
        images_dict = {
            'fixed_fake': fixed_fake_images,
            'fake': fake_images
        }

        # Visualise all results
        for name, images in images_dict.items():
            gaussian_plot(images, '{}/{}_samples_step_{}.png'.format(
                                    img_dir, name, global_step))
