from pathlib import Path
from datetime import datetime
import zarr
import numpy as np

from torch_spots.detection.inference import SpotDetection
from torch_spots.detection.metrics import PointMetrics

import click
import itertools
import pandas as pd

from tqdm import tqdm

def make_sweep(sweep, default_kwargs):

    sweep_list = []

    sweep_names = [*sweep.keys()]
    sweep_vals = [*sweep.values()]

    for combos in itertools.product(*sweep_vals):
        curr_kwargs = default_kwargs.copy()

        for i, sweep_name in enumerate(sweep_names):
            curr_kwargs[sweep_name] = combos[i]

        sweep_list.append(curr_kwargs)

    return sweep_list


@click.command()
@click.option(
    '--device', 
    default='cuda:0', 
    help="""The device that you want to run the inference on. 
            Can be `'cpu'` or `'cuda'`. If `'cuda'`, can also specify the specific GPU if multiple are available.""")

@click.option(
    '--model-path', 
    default= "data/model/20260513091742/saved_model_best_dict.pth", 
    help="""Path to model. 
            If unset, will use default DeepCell location (`~/.deepcell/models/mesmer/saved_model_best_dict.pth`)"""
            )

@click.option(
    '--data-path', 
    default=Path.home() / ".deepcell/spotnet/test.zarr" , 
    help="""Path to the Zarr file containing the test data for evaluation.
            If unset, defaults to DeepCell location (`~/.deepcell/spotnet/test.zarr)"""
        )

def main(device: str,
         model_path: str,
         data_path: str):
    
    default_kwargs={
        'threshold': 0.99,
        'min_distance': 1
    }

    sweep_vals = {
        'threshold': np.linspace(0.99, 0.999, 10),
        'min_distance': np.arange(1, 6, dtype=int),
    }

    sweep = make_sweep(sweep_vals, default_kwargs)
        
    z_test = zarr.open(data_path)

    X = z_test['X']
    y = z_test['y']
    y_inds = z_test['y_inds']

    # Load model and application
    detector = SpotDetection(
        model_path=model_path,
        device=device,
    )


    results = {
        'image_id': [],
        'n_pred': [],
        'n_true': [],
        'tp': [],
        'fp': [],
        'fn': [],
        'threshold': [],
        'min_dist': [],
        'tau': [],
    }

    for curr_sweep in tqdm(sweep):

        spots = []
        true_spots = []

        curr_threshold = curr_sweep['threshold']
        curr_min_dist = curr_sweep['min_distance']

        for i in range(X.shape[0]):
            spots.append(detector._eval_predict(X[i:i+1], threshold=curr_threshold, min_distance=curr_min_dist))
            true_spots.append(y[i, 0:y_inds[i]])

        for tau in np.linspace(0, 5, 10):
            for i in range(len(spots)):
                res = PointMetrics(true_spots[i], spots[i]).evaluate(threshold=tau)
                results['image_id'].append(i)
                results['n_pred'].append(len(spots[i]))
                results['n_true'].append(len(true_spots[i]))
                results['tp'].append(res.TP)
                results['fp'].append(res.FP)
                results['fn'].append(res.FN)
                results['threshold'].append(curr_threshold)
                results['min_dist'].append(curr_min_dist)
                results['tau'].append(tau)
    
    df = pd.DataFrame(results)

    df.to_csv(f'eval_sweep.csv')
    return df

if __name__ == "__main__":
    main()