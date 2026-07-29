import torch
import os
import datetime
import zarr
import json

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from torch_spots.detection.dotnet import SpotNet
from torch_spots.detection.loss import DotNetLosses, LossTracker
from torch_spots.detection.loader import spot_dataloader

def train(
        trainloader,
        valloader,
        model=None,
        lr=1e-2,
        epochs=8,
        save_path_prefix = "data/saved_model",
        writer=None,
        write=True,
        device='cuda:2',
    ):

    assert model is not None, "Please specify a model"

    loss = DotNetLosses(sigma=3.0, gamma=0.5, focal=False)

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, momentum=0.9, 
        weight_decay=1e-6, 
        nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(epochs):

        model.train()
        train_loss = LossTracker()
        val_loss = LossTracker()

        pbar_train = tqdm(trainloader, desc=f'Epoch {epoch} [Train]', dynamic_ncols=True)

        for batch in pbar_train:
            
            image, labels = batch
            batch_size = image.shape[0]
            
            image = image.to(device)

            for k, v in labels.items():
                labels[k] = v.to(device)

            optimizer.zero_grad()

            outputs = model(image)
            curr_loss = loss(outputs, labels) 

            train_loss.update(curr_loss, batch_size=batch_size)  

            pbar_train.set_postfix({
                'loss': f"{train_loss.get_loss():.4f}"
            })

            curr_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        
            optimizer.step()

        writer.add_scalar('avg_loss/train', train_loss.get_loss(), epoch)

        model.eval()
        pbar_val = tqdm(valloader, desc=f'Epoch {epoch} [Val]', dynamic_ncols=True)

        with torch.no_grad():
            for batch in pbar_val:
                image, labels = batch

                batch_size = image.shape[0]
                
                image = image.to(device)

                for k, v in labels.items():
                    labels[k] = v.to(device)

                voutputs = model(image)
                curr_vloss = loss(voutputs, labels)

                val_loss.update(curr_vloss, batch_size=batch_size)  

                pbar_val.set_postfix({
                    'loss': f"{val_loss.get_loss():.4f}"
                })

        # Shape 8, H, W

        avg_vloss = val_loss.get_loss()
        writer.add_scalar('avg_loss/val', avg_vloss, epoch)

        scheduler.step()
        dict_save_path = save_path_prefix + "/saved_model_best_dict.pth"
        torch.save(model.state_dict(), dict_save_path)
        if epoch == 0:

            best_vloss = avg_vloss
            
            if write:
                dict_save_path = save_path_prefix + "/saved_model_best_dict.pth"
                torch.save(model.state_dict(), dict_save_path)

            print()
            print("New best model.")
            print()  

        elif avg_vloss < best_vloss:
            best_vloss = avg_vloss
            
            if write:
                dict_save_path = save_path_prefix + "/saved_model_best_dict.pth"
                torch.save(model.state_dict(), dict_save_path)

            print()
            print("New best model.")
            print()
            
        print(f'Training loss: {train_loss.get_loss():.3f}')
        print(f'Validation loss: {val_loss.get_loss():.3f}')
        print()

    if write:
        dict_save_path = save_path_prefix + "/last_model_dict.pth"
        torch.save(model.state_dict(), dict_save_path)

    return model

def main():

    config = {
        'model_path': "data/model/",
        'data_path': Path.home() / '.deepcell/spotnet/',
        'run_info': 'data/logs/',
        'epochs': 20,
        'batch_size': 10,
        'lr': 1e-3,
        'num_workers': 4,
        'write': True,
        'device': 'cuda:2',
    }
    
    config['data_path'] = str(config['data_path'])

    curr_time = f"{datetime.datetime.now():%Y%m%d%H%M%S}"

    z_train = zarr.open(f"{config['data_path']}/train.zarr")
    z_val = zarr.open(f"{config['data_path']}/val.zarr")

    run_info = config['run_info'] + '/' + curr_time
    model_path = config['model_path'] + '/' + curr_time
    
    if not os.path.isdir(run_info):
        os.makedirs(run_info, exist_ok=True)
    if not os.path.isdir(model_path) and config['write']:
        os.makedirs(model_path, exist_ok=True)

    with open(model_path + '/' + 'training_config.json', 'w') as file:
        json.dump(config, file)

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
                 norm_method=None)

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
        augment=True
    )

    val_data = spot_dataloader(
        X=z_val['X'],
        y=z_val['y'],
        y_inds=z_val['y_inds'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        augment=False,
        shuffle=False
    )

    # train the model
    model = train(
        train_data,
        val_data,
        model=model,
        lr=config['lr'],
        epochs=config['epochs'],
        save_path_prefix=model_path,
        writer=writer,
        write=config['write'],
        device=config['device'],
    )

    writer.close()

if __name__ == "__main__":
    main()