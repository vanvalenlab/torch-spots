import torch
import os
import random
import datetime
import zarr
import json
import click
import copy
import numpy as np

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from torch_spots.detection.dotnet import SpotNet
from torch_spots.detection.loss import DotNetLosses, LossTracker, kernel_l2_penalty
from torch_spots.detection.loader import spot_dataloader
from torch_spots.dataset import DEFAULT_DATA_DIR, verify_store

def reference_lr(epoch, global_step, lr, epoch_decay, step_decay):
    """Learning rate at ``global_step`` during ``epoch`` (both counted from 0).

    ``lr * epoch_decay ** epoch / (1 + step_decay * global_step)``: the original's per-epoch
    exponential decay combined with Keras 2.8's per-step ``decay`` (``deepcell_spots/training.py:29-37``;
    Keras ``optimizer_v2.py:1003-1013``).
    """
    return lr * epoch_decay ** epoch / (1.0 + step_decay * global_step)


def _to_device(image, labels, device):
    return image.to(device), {k: v.to(device) for k, v in labels.items()}


def train(
        trainloader,
        valloader,
        model=None,
        optimizer_config=None,
        epochs=10,
        save_path_prefix="data/saved_model",
        writer=None,
        write=True,
        device='cuda:2',
        loss_config=None,
    ):
    """Train ``model`` and return ``(model, summary)``.

    ``summary`` holds ``best_epoch`` (0-based), ``best_val_objective`` and ``steps``.
    The objective is ``regression_weight * reg + classification_weight * cls`` plus the
    classification-head L2 penalty, on both training and validation batches. The best checkpoint
    (``saved_model_best_dict.pth``) is written on the first epoch and on each strict improvement
    of the validation objective; ``last_model_dict.pth`` holds the final epoch.
    """
    assert model is not None, "Please specify a model"

    opt = dict(REFERENCE_TRAINING['optimizer'] if optimizer_config is None else optimizer_config)
    loss_config = loss_config or REFERENCE_TRAINING['loss']
    loss = DotNetLosses(sigma=3.0, gamma=0.5, **loss_config)

    # No weight decay: the L2 penalty is an explicit loss term. The learning rate is set
    # before every step by reference_lr, so there is no scheduler object.
    optimizer = torch.optim.SGD(
        model.parameters(), lr=opt['lr'], momentum=opt['momentum'], weight_decay=0, nesterov=True)

    def objective(outputs, labels):
        return loss(outputs, labels) + kernel_l2_penalty(model, opt['kernel_l2'])

    global_step = 0
    best_vloss, best_epoch = None, None

    for epoch in range(epochs):

        model.train()
        train_loss = LossTracker()
        val_loss = LossTracker()

        pbar_train = tqdm(trainloader, desc=f'Epoch {epoch} [Train]', dynamic_ncols=True)

        for batch in pbar_train:
            image, labels = _to_device(*batch, device)
            batch_size = image.shape[0]

            for group in optimizer.param_groups:
                group['lr'] = reference_lr(epoch, global_step, opt['lr'], opt['epoch_decay'],
                                           opt['step_decay'])

            optimizer.zero_grad()
            curr_loss = objective(model(image), labels)

            if not torch.isfinite(curr_loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}, step {global_step}; "
                    "the run has diverged.")

            train_loss.update(curr_loss, batch_size=batch_size)
            pbar_train.set_postfix({'loss': f"{train_loss.get_loss():.4f}"})

            curr_loss.backward()

            if opt['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=opt['grad_clip'], error_if_nonfinite=True)

            optimizer.step()
            global_step += 1

        if writer is not None:
            writer.add_scalar('avg_loss/train', train_loss.get_loss(), epoch)

        model.eval()
        pbar_val = tqdm(valloader, desc=f'Epoch {epoch} [Val]', dynamic_ncols=True)

        with torch.no_grad():
            for batch in pbar_val:
                image, labels = _to_device(*batch, device)
                curr_vloss = objective(model(image), labels)
                val_loss.update(curr_vloss, batch_size=image.shape[0])
                pbar_val.set_postfix({'loss': f"{val_loss.get_loss():.4f}"})

        avg_vloss = val_loss.get_loss()
        if writer is not None:
            writer.add_scalar('avg_loss/val', avg_vloss, epoch)

        if best_vloss is None or avg_vloss < best_vloss:
            best_vloss, best_epoch = avg_vloss, epoch
            if write:
                torch.save(model.state_dict(), save_path_prefix + "/saved_model_best_dict.pth")
            print()
            print("New best model.")
            print()

        print(f'Training loss: {train_loss.get_loss():.3f}')
        print(f'Validation loss: {val_loss.get_loss():.3f}')
        print()

    if write:
        torch.save(model.state_dict(), save_path_prefix + "/last_model_dict.pth")

    return model, {'best_epoch': best_epoch, 'best_val_objective': best_vloss, 'steps': global_step}

def seed_everything(seed):
    """Seed Python, NumPy and torch."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# The reference training settings (those of the published SpotDetection-8 model). Runs are
# compared against it, so a run that changes any value is recorded as
# deviating from the reference in ``training_config.json`` (``non_reference``).
#
# Sources, in order of authority: the published SpotDetection-8 model's saved files,
# the shipped application's ``model_metadata``, the paper's Methods, then the deepcell_spots 0.4.2
# code defaults.
#   augmentation   flip: ``deepcell_spots/training.py:38-42,144-145``; rotation 180 and zoom
#                  (0.8, 1.2): paper Methods (not recorded in the model files).
#   loss           1/6 and 5/6: model's saved ``loss_weights`` (offset_regression 0.1667,
#                  classification 0.8333). ``focal`` False: ``dotnet.py:38,62,66``.
#   optimizer      lr 0.01, momentum 0.9, step decay 1e-6: model's saved SGD config;
#                  epoch_decay 0.99: application ``model_metadata['lr_decay']`` and the saved
#                  learning rate 0.01 * 0.99**4. kernel_l2 1e-5: ``dotnet.py``.
#   model          norm_method None: all five ImageNormalization2D layers saved with null.
#   epochs         10: the saved optimizer state (0.01 * 0.99**4) shows the shipped weights came from
#                  epoch 4 of a longer planned run (``model_metadata['n_epochs']`` is 50, never reached);
#                  test F1 was 0.8867 at 10 epochs and 0.7607 at 50.
#                  batch_size 1: application ``model_metadata``.
REFERENCE_TRAINING = {
    'augmentation': {
        'flip': True,
        'rotation_range': 180.0,
        'zoom_range': (0.8, 1.2),
    },
    'loss': {
        'regression_weight': 1.0 / 6.0,
        'classification_weight': 5.0 / 6.0,
        'focal': False,
    },
    'optimizer': {
        'lr': 1e-2,
        'momentum': 0.9,
        'step_decay': 1e-6,
        'epoch_decay': 0.99,
        'kernel_l2': 1e-5,
        'grad_clip': None,
    },
    # In-model input normalisation is off (the published model's layers are pass-through).
    # deepcell's local 13 x 13 'std' stays available as an option.
    'model': {
        'norm_method': None,
    },
    'epochs': 10,
    'batch_size': 1,
}

#: Values the model accepts for ``model.norm_method``; ``None`` means no normalisation.
VALID_NORM_METHODS = ('std', 'whole_image', None)


def validate_config(config):
    """Raise ``ValueError`` for a setting value the code cannot honour (currently the
    normalisation method), so a typo fails at load time rather than after data is read."""
    method = config['model']['norm_method']
    if method not in VALID_NORM_METHODS:
        raise ValueError(f"Unknown model.norm_method {method!r}; "
                         f"expected one of 'std', 'whole_image' or null")

# Run settings (where things live, which device, how many workers, which seed). These are not
# part of the reference, so they never show up as non-reference settings.
RUN_SETTINGS = {
    'model_path': "data/model/",
    'data_path': str(DEFAULT_DATA_DIR),
    'run_info': 'data/logs/',
    'num_workers': 4,
    'write': True,
    'device': 'cuda:2',
    'seed': 0,
}

DEFAULT_CONFIG = {**copy.deepcopy(REFERENCE_TRAINING), **RUN_SETTINGS}


def merge_config(base, override):
    """Return ``base`` with ``override`` merged on top (nested dicts merge key by key).

    Unknown keys raise ``ValueError`` so a typo in a JSON file cannot silently do nothing.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key not in merged:
            raise ValueError(f"Unknown training setting: {key!r}")
        if isinstance(merged[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Setting {key!r} must be an object of sub-settings")
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path=None):
    """Defaults, with the JSON file at ``path`` (if any) merged on top."""
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path) as file:
        config = merge_config(DEFAULT_CONFIG, json.load(file))
    validate_config(config)
    return config


def non_reference_settings(config):
    """Settings in ``config`` whose value differs from :data:`REFERENCE_TRAINING`.

    Returns a dict from dotted setting name (e.g. ``'optimizer.lr'``) to the value used.
    """
    def flatten(d, prefix=''):
        for key, value in d.items():
            if isinstance(value, dict):
                yield from flatten(value, f"{prefix}{key}.")
            else:
                yield f"{prefix}{key}", value

    # Round-trip through JSON so tuples and lists compare equal.
    norm = lambda v: json.loads(json.dumps(v))
    reference = dict(flatten(REFERENCE_TRAINING))
    used = dict(flatten({k: config[k] for k in REFERENCE_TRAINING}))
    return {name: norm(value) for name, value in used.items() if norm(value) != norm(reference[name])}


def run_training(config):
    """Train one model from ``config`` (see :data:`DEFAULT_CONFIG`) and return the model directory.

    The train and val stores are verified against the pinned SpotNet v1.1 splits before the
    model is built; the seed and dataset provenance are written to ``training_config.json``.
    """
    config = dict(config)
    validate_config(config)
    config['data_path'] = str(config['data_path'])
    seed = int(config['seed'])

    z_train = zarr.open(f"{config['data_path']}/train.zarr", mode='r')
    z_val = zarr.open(f"{config['data_path']}/val.zarr", mode='r')
    # Refuse mismatched data before anything is built or written.
    provenance = {
        'train': verify_store(z_train, 'train'),
        'val': verify_store(z_val, 'val'),
    }

    seed_everything(seed)

    curr_time = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    run_info = config['run_info'] + '/' + curr_time
    model_path = config['model_path'] + '/' + curr_time

    if not os.path.isdir(run_info):
        os.makedirs(run_info, exist_ok=True)
    if not os.path.isdir(model_path) and config['write']:
        os.makedirs(model_path, exist_ok=True)

    if config['write']:
        with open(model_path + '/' + 'training_config.json', 'w') as file:
            json.dump({**config, 'seed': seed, 'dataset': provenance,
                       'non_reference': non_reference_settings(config)}, file, indent=2)

    writer = SummaryWriter(run_info)

    print("Initializing model:")
    print()

    model = SpotNet(
                 input_channels=1,
                 receptive_field=13,
                 n_skips=3,
                 n_conv_filters=32,
                 n_dense_filters=128,
                 num_classes=2,
                 regression_feature_size=256,
                 **config['model'])

    model = model.to(config['device'])

    print("SpotNet Model:")
    print(f"    Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Set up data generators with updated data
    train_data = spot_dataloader(
        X=z_train['X'],
        y=z_train['y'],
        y_inds=z_train['y_inds'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        augment=True,
        augmentation=config['augmentation'],
        drop_last=True,
        seed=seed,
    )

    val_data = spot_dataloader(
        X=z_val['X'],
        y=z_val['y'],
        y_inds=z_val['y_inds'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        augment=False,
        shuffle=False,
        drop_last=True,
        seed=seed,
    )

    # train the model
    model, summary = train(
        train_data,
        val_data,
        model=model,
        optimizer_config=config['optimizer'],
        epochs=config['epochs'],
        save_path_prefix=model_path,
        writer=writer,
        write=config['write'],
        device=config['device'],
        loss_config=config['loss'],
    )

    writer.close()

    if config['write']:
        # Record where the best checkpoint came from, next to the settings used.
        path = model_path + '/training_config.json'
        with open(path) as file:
            record = json.load(file)
        record['best_epoch'] = summary['best_epoch']
        record['best_val_objective'] = summary['best_val_objective']
        with open(path, 'w') as file:
            json.dump(record, file, indent=2)
    return model_path


@click.command()
@click.option('--config', 'config_path', type=click.Path(exists=True, dir_okay=False), default=None,
              help="JSON file of training settings merged over the defaults (e.g. an ablation). "
                   "--seed, --data-path and --device still apply on top.")
@click.option('--seed', type=int, default=None,
              help=f"Random seed for initialization, data order and augmentation "
                   f"[default: {DEFAULT_CONFIG['seed']}].")
@click.option('--data-path', default=None,
              help="Directory with the converted SpotNet v1.1 train.zarr and val.zarr "
                   f"[default: {DEFAULT_CONFIG['data_path']}].")
@click.option('--device', default=None,
              help=f"Device to train on, e.g. 'cpu' or 'cuda:0' [default: {DEFAULT_CONFIG['device']}].")
def main(config_path, seed, data_path, device):
    config = load_config(config_path)
    for key, value in (('seed', seed), ('data_path', data_path), ('device', device)):
        if value is not None:
            config[key] = value
    run_training(config)


if __name__ == "__main__":
    main()
