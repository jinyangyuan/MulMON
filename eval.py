# -*- coding: utf-8 -*-
"""
Training MulMON on a single GPU.
@author: Nanbo Li
"""
import sys
import os
import argparse
import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.utils.data as utils_data
import yaml
from models.mulmon import MulMON
from torch.utils.tensorboard import SummaryWriter

# set project search path
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

from config import CONFIG
from scheduler import AnnealingStepLR
from trainer.model_trainer import ModelTrainer
from utils import set_random_seed, load_trained_mp, ensure_dir


# ------------------------- important specs ------------------------
def running_cfg(cfg):

    cfg.view_dim = cfg.v_in_dim

    # log directory
    ckpt_base = cfg.ckpt_base
    ensure_dir(ckpt_base)

    # model savedir
    check_dir = os.path.join(ckpt_base, '{}_log/'.format(cfg.arch))
    ensure_dir(check_dir)

    # generated sample dir
    save_dir = os.path.join(check_dir, 'saved_models/')
    ensure_dir(save_dir)

    # visualise training epochs
    vis_train_dir = os.path.join(check_dir, 'vis_training/')
    ensure_dir(vis_train_dir)

    # generated sample dir  (for testing generation)
    generated_dir = os.path.join(check_dir, 'generated/')
    ensure_dir(generated_dir)

    if cfg.resume_path is not None:
        assert os.path.isfile(cfg.resume_path)
    elif cfg.resume_epoch is not None:
        resume_path = os.path.join(save_dir,
                                   'checkpoint-epoch{}.pth'.format(cfg.resume_epoch))
        assert os.path.isfile(resume_path)
        cfg.resume_path = resume_path

    cfg.check_dir = check_dir
    cfg.save_dir = save_dir
    cfg.vis_train_dir = vis_train_dir
    cfg.generated_dir = generated_dir

    cfg.image_size = [64, 64]

    return cfg


# ---------------------------- main function -----------------------------
def get_trainable_params(model):
    params_to_update = []
    print('trainable parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name)
            params_to_update.append(param)
    return params_to_update


def train(gpu_id, CFG):
    if 'GQN' in CFG.arch:
        from models.baseline_gqn import GQN as ScnModel
        print(" --- Arch: GQN ---")
    elif 'IODINE' in CFG.arch:
        from models.baseline_iodine import IODINE as ScnModel
        print(" --- Arch: IODINE ---")
    elif 'MulMON' in CFG.arch:
        from models.mulmon import MulMON as ScnModel
        print(" --- Arch: MulMON ---")
    else:
        raise NotImplementedError

    # Create the model
    scn_model = ScnModel(CFG)
    if CFG.resume_epoch is not None:
        state_dict = load_trained_mp(CFG.resume_path)
        scn_model.load_state_dict(state_dict, strict=True)
    params_to_update = get_trainable_params(scn_model)

    if CFG.optimiser == 'RMSprop':
        optimiser = torch.optim.RMSprop(params_to_update,
                                        lr=CFG.lr_rate,
                                        weight_decay=CFG.weight_decay)
        lr_scheduler = None
    else:
        optimiser = torch.optim.Adam(params_to_update,
                                     lr=CFG.lr_rate,
                                     weight_decay=CFG.weight_decay)
        lr_scheduler = AnnealingStepLR(optimiser, mu_i=CFG.lr_rate, mu_f=0.1*CFG.lr_rate, n=1e6)

    if 'gqn' in CFG.DATA_TYPE:
        from data_loader.getGqnH5 import DataLoader
    elif 'clevr' in CFG.DATA_TYPE:
        from data_loader.getClevrMV import DataLoader
    else:
        raise NotImplementedError

    # get data Loader
    train_dl = DataLoader(CFG.DATA_ROOT,
                          CFG.train_data_filename,
                          batch_size=CFG.batch_size,
                          shuffle=True,
                          num_slots=CFG.num_slots,
                          use_bg=True)
    val_dl = DataLoader(CFG.DATA_ROOT,
                        CFG.test_data_filename,
                        batch_size=CFG.batch_size,
                        shuffle=True,
                        num_slots=CFG.num_slots,
                        use_bg=True)

    if CFG.seed is None:
        CFG.seed = random.randint(0, 1000000)
    set_random_seed(CFG.seed)

    trainer = ModelTrainer(
        model=scn_model,
        loss=None,
        metrics=None,
        optimizer=optimiser,
        step_per_epoch=CFG.step_per_epoch,
        config=CFG,
        train_data_loader=train_dl,
        valid_data_loader=val_dl,
        device=gpu_id,
        lr_scheduler=lr_scheduler
    )
    # Start training session
    trainer.train()


class Dataset(utils_data.Dataset):

    def __init__(self, data, data_view):
        super(Dataset, self).__init__()
        self.images = torch.tensor(data['image'])
        self.segments = torch.tensor(data['segment'])
        self.overlaps = torch.tensor(data['overlap'])
        self.viewpoints = torch.tensor(data_view)
        if self.images.ndim == 4 and self.segments.ndim == 3 and self.overlaps.ndim == 3:
            self.images = self.images[:, None]
            self.segments = self.segments[:, None]
            self.overlaps = self.overlaps[:, None]
            self.viewpoints = self.viewpoints[:, None]
        assert self.images.ndim == 5
        assert self.segments.ndim == 4
        assert self.overlaps.ndim == 4
        assert self.viewpoints.ndim == 3

    def __getitem__(self, idx):
        image = self.images[idx]
        segment = self.segments[idx]
        overlap = self.overlaps[idx]
        viewpoint = self.viewpoints[idx]
        data = {'image': image, 'segment': segment, 'overlap': overlap, 'viewpoint': viewpoint}
        return data

    def __len__(self):
        return self.images.shape[0]


def get_data_loaders(config):
    image_shape = None
    datasets = {}
    with h5py.File(config['path_data'], 'r', libver='latest', swmr=True) as f, \
            h5py.File(config['path_viewpoint'], 'r', libver='latest', swmr=True) as f_view:
        phase_list = [*f.keys()]
        if not config['train']:
            phase_list = [val for val in phase_list if val not in ['train', 'valid']]
        index_sel = slice(config['batch_size']) if config['debug'] else ()
        for phase in phase_list:
            data = {key: f[phase][key][index_sel] for key in f[phase] if key not in ['layers', 'masks']}
            data['image'] = np.moveaxis(data['image'], -1, -3)
            if image_shape is None:
                image_shape = data['image'].shape[-3:]
            else:
                assert image_shape == data['image'].shape[-3:]
            data_view = f_view[phase]['viewpoint'][index_sel]
            datasets[phase] = Dataset(data, data_view)
    if 'train' in datasets and 'valid' not in datasets:
        datasets['valid'] = datasets['train']
    data_loaders = {}
    for key, val in datasets.items():
        data_loaders[key] = utils_data.DataLoader(
            val,
            batch_size=config['batch_size'],
            num_workers=1,
            shuffle=(key == 'train'),
            drop_last=(key == 'train'),
            pin_memory=True,
        )
    return data_loaders, image_shape


def add_scalars(writer, metrics, losses, step, phase):
    for key, val in metrics.items():
        writer.add_scalar('{}/metric_{}'.format(phase, key), val, global_step=step)
    for key, val in losses.items():
        writer.add_scalar('{}/loss_{}'.format(phase, key), val, global_step=step)
    return


def compute_overview(config, results, dpi=100):
    def convert_image(image):
        image = np.moveaxis(image, 0, 2)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image
    def plot_image(ax, image, xlabel=None, ylabel=None, color=None):
        plot = ax.imshow(image, interpolation='bilinear')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel, color='k' if color is None else color, fontfamily='monospace') if xlabel else None
        ax.set_ylabel(ylabel, color='k' if color is None else color, fontfamily='monospace') if ylabel else None
        ax.xaxis.set_label_position('top')
        return plot
    def get_overview(fig_idx):
        image = results_sel['image'][fig_idx]
        recon = results_sel['recon'][fig_idx]
        apc = results_sel['apc'][fig_idx]
        mask = results_sel['mask'][fig_idx]
        pres = results_sel['pres'][fig_idx]
        num_views, num_slots = apc.shape[:2]
        rows, cols = 2 * num_views, num_slots + 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows + 0.2), dpi=dpi)
        for idx_v in range(num_views):
            plot_image(axes[idx_v * 2, 0], convert_image(image[idx_v]), xlabel='scene' if idx_v == 0 else None)
            plot_image(axes[idx_v * 2 + 1, 0], convert_image(recon[idx_v]))
            for idx_s in range(num_slots):
                xlabel = 'obj_{}'.format(idx_s) if idx_s < num_slots - 1 else 'back'
                xlabel = xlabel if idx_v == 0 else None
                color = [1.0, 0.5, 0.0] if pres[idx_s] >= 128 else [0.0, 0.5, 1.0]
                plot_image(axes[idx_v * 2, idx_s + 1], convert_image(apc[idx_v, idx_s]), xlabel=xlabel, color=color)
                plot_image(axes[idx_v * 2 + 1, idx_s + 1], convert_image(mask[idx_v, idx_s]))
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        out = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, -1)
        plt.close(fig)
        return out
    summ_image_count = min(config['summ_image_count'], config['batch_size'])
    results_sel = {key: val[:summ_image_count].data.cpu().numpy() for key, val in results.items()}
    overview_list = [get_overview(idx) for idx in range(summ_image_count)]
    overview = np.concatenate(overview_list, axis=0)
    overview = np.moveaxis(overview, 2, 0)
    return overview


def add_overviews(config, writer, results, step, phase):
    overview = compute_overview(config, results)
    writer.add_image(phase, overview, global_step=step)
    return


def main(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config')
    parser.add_argument('--path_data')
    parser.add_argument('--path_viewpoint')
    parser.add_argument('--folder_log')
    parser.add_argument('--folder_out')
    parser.add_argument('--timestamp')
    parser.add_argument('--num_tests', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_timestamp', action='store_true')
    parser.add_argument('--file_ckpt', default='ckpt.pth')
    parser.add_argument('--file_model', default='model.pth')
    parser.add_argument('--arch', type=str, default='ScnModel',
                        help="model name")
    parser.add_argument('--datatype', type=str, default='clevr',
                        help="one of [gqn_jaco, clevr_mv, clevr_aug]")
    parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--step_per_epoch', default=0, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int, metavar='N', help='number of data samples of a minibatch')
    parser.add_argument('--work_mode', type=str, default='training', help="model's working mode")
    parser.add_argument('--optimiser', type=str, default='Adam', help="help= one of [Adam, RMSprop]")
    parser.add_argument('--resume_epoch', default=None, type=int, metavar='N',
                        help='resume weights from [N]th epochs')

    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--nrank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--gpu_start', default=0, type=int, help='first gpu indicator, default using 0 as the start')
    parser.add_argument('--master_port', default='8888', type=str, help='used for rank0 communication with others')

    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--lr_rate', default=1e-4, type=float, help='learning rate')

    parser.add_argument('--num_slots', default=7, type=int, help='(maximum) number of component slots')
    parser.add_argument('--temperature', default=0.0, type=float,
                        help='spatial scheduler increase rate, the hotter the faster coeff grows')
    parser.add_argument('--latent_dim', default=16, type=int, help='size of the latent dimensions')
    parser.add_argument('--view_dim', default=5, type=int, help='size of the viewpoint latent dimensions')
    parser.add_argument('--min_sample_views', default=1, type=int, help='mininum allowed #views for scene learning')
    parser.add_argument('--max_sample_views', default=5, type=int, help='maximum allowed #views for scene learning')
    parser.add_argument('--num_vq_show', default=5, type=int, help='#views selected for visualisation')
    parser.add_argument('--pixel_sigma', default=0.1, type=float, help='loss strength item')
    parser.add_argument('--kl_latent', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--kl_spatial', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--exp_attention', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--query_nll', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--exp_nll', default=1.0, type=float, help='loss strength item')

    parser.add_argument("--use_mask", help="use gt mask to by pass the segmentation phase",
                        action="store_true", default=False)
    parser.add_argument("--use_bg", help="treat background as an object",
                        action="store_true", default=False)

    parser.add_argument("-i", '--input_dir', required=True,  help="path to the input data for the model to read")
    parser.add_argument("-o", '--output_dir', required=True,  help="destination dir for the model to write out results")
    args = parser.parse_args()

    ###########################################
    # General training reconfig
    ###########################################
    cfg.arch = args.arch
    cfg.DATA_TYPE = args.datatype
    cfg.num_epochs = args.epochs
    cfg.step_per_epoch = args.step_per_epoch if args.step_per_epoch > 0 else None
    cfg.batch_size = args.batch_size
    cfg.WORK_MODE = args.work_mode
    cfg.optimiser = args.optimiser
    cfg.resume_epoch = args.resume_epoch
    cfg.seed = args.seed
    cfg.lr_rate = args.lr_rate
    cfg.num_slots = args.num_slots
    cfg.temperature = args.temperature
    cfg.latent_dim = args.latent_dim
    cfg.v_in_dim = args.view_dim
    cfg.view_dim = args.view_dim
    cfg.min_sample_views = args.min_sample_views
    cfg.max_sample_views = args.max_sample_views
    cfg.num_vq_show = args.num_vq_show
    cfg.pixel_sigma = args.pixel_sigma
    cfg.use_mask = args.use_mask
    cfg.use_bg = args.use_bg
    cfg.elbo_weights = {
        'kl_latent': args.kl_latent,
        'kl_spatial': args.kl_spatial,
        'exp_attention': args.exp_attention,
        'exp_nll': args.exp_nll,
        'query_nll': args.query_nll
    }
    # I/O path configurations
    cfg.DATA_ROOT = args.input_dir
    cfg.ckpt_base = args.output_dir

    ###########################################
    # Config gpu usage
    ###########################################
    cfg.nodes = args.nodes
    cfg.gpus = args.gpus
    cfg.nrank = args.nrank
    cfg.gpu_start = args.gpu_start
    cfg.world_size = args.gpus * args.nodes  #

    cfg = running_cfg(cfg)

    with open(args.path_config) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if key not in config or val is not None:
            config[key] = val
    if config['debug']:
        config['ckpt_intvl'] = 1
    if config['resume']:
        config['train'] = True
    if config['timestamp'] is None:
        config['timestamp'] = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if config['use_timestamp']:
        for key in ['folder_log', 'folder_out']:
            config[key] = os.path.join(config[key], config['timestamp'])
    data_loaders, image_shape = get_data_loaders(config)
    config['image_shape'] = image_shape
    model = MulMON(cfg).cuda()
    path_model = os.path.join(config['folder_out'], config['file_model'])
    model.load_state_dict(torch.load(path_model))
    model.train(False)
    def get_path_detail():
        return os.path.join(config['folder_out'], '{}_{}.h5'.format(phase, num_views))
    phase_list = [n for n in config['phase_param'] if n not in ['train', 'valid']]
    for phase in phase_list:
        phase_param = config['phase_param'][phase]
        data_key = phase_param['key'] if 'key' in phase_param else phase
        for num_views in [1, 2, 4, 8]:
            phase_param['num_views'] = num_views
            model.K = phase_param['num_slots']
            path_detail = get_path_detail()
            results_all = {}
            for data in data_loaders[data_key]:
                results = {}
                for idx_run in range(config['num_tests']):
                    with torch.set_grad_enabled(True):
                        _, sub_results, _ = model(data, phase_param, require_results=True, deterministic_data=True)
                    for key, val in sub_results.items():
                        if key in ['image']:
                            continue
                        val = val.data.cpu().numpy()
                        if key in ['recon', 'logits_mask', 'mask', 'apc']:
                            val = np.moveaxis(val, -3, -1)
                        if key in results:
                            results[key].append(val)
                        else:
                            results[key] = [val]
                for key, val in results.items():
                    val = np.stack(val)
                    if key in results_all:
                        results_all[key].append(val)
                    else:
                        results_all[key] = [val]
            with h5py.File(path_detail, 'w') as f:
                for key, val in results_all.items():
                    f.create_dataset(key, data=np.concatenate(val, axis=1), compression='gzip')
    return


##############################################################################
if __name__ == "__main__":
    cfg = CONFIG()
    main(cfg)
