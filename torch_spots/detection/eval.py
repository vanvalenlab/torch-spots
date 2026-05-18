from pathlib import Path
from datetime import datetime
import zarr
import numpy as np

from .inference import SpotDetection
from .metrics import PointMetrics

import click

import pandas as pd


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
    
    sweep = {
        
    }
        
    z_test = zarr.open(data_path)

    X = z_test['X']
    y = z_test['y']
    y_inds = z_test['y_inds']

    # Load model and application
    detector = SpotDetection(
        model_path=model_path,
        device=device,
    )

    spots = []
    true_spots = []

    for i in range(X.shape[0]):
        spots.append(detector._eval_predict(X[i:i+1]))
        true_spots.append(y[i, 0:y_inds[i]])

    results = []
    for i in range(len(spots)):
        res = PointMetrics(true_spots[i], spots[i]).evaluate(threshold=2).return_numpy()
        results.append(res)

    results = np.array(results)

    # make column names
    fields = PointMetrics._empty_result(threshold=1).return_fields()
    
    df = pd.DataFrame(results, columns=fields)

    df.to_csv(f'eval_results_{datetime.now().isoformat(timespec="seconds")}.csv')
    return df

if __name__ == "__main__":
    main()