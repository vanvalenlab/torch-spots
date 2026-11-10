import numpy as np
import torch

from huggingface_hub import hf_hub_download

from torch_spots.detection.dotnet import (MODEL_ARCHITECTURE_VERSION, IncompatibleCheckpointError,
                                          load_spotnet)
from skimage.feature import peak_local_max

import numpy as np
import glob

from pathlib import Path

from torch_spots.detection.loader import preprocess_image
from torch_spots.detection.utils import tile_input, untile_output, \
      max_cp_array_to_point_list_max, extract_spots_prob_from_coords_maxpool


#: Filename of the default pretrained model on Hugging Face. ``None`` until a model trained on
#: the current architecture is published (the 2026-07-29 file predates it).
DEFAULT_PRETRAINED = None

#: Reference operating point. The threshold was chosen once, on the validation split, from the
#: grid 0.90-0.999 using the three round 6 seeds (``data/eval/round6_threshold_selection.json``,
#: published-model-reference). The minimum distance and the matching distance follow
#: ``deepcell_spots/applications/spot_detection.py:266-269``. The application's own default is 0.95.
EVAL_THRESHOLD = 0.993
EVAL_MIN_DISTANCE = 1


class SpotDetection():

    def __init__(
            self,
            model_path=None,
            device=None
    ):
        
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        if model_path is None:
            if DEFAULT_PRETRAINED is None:
                raise IncompatibleCheckpointError(
                    "The default pretrained model (torch-spots_2026-07-29.pth) predates the "
                    "current architecture (version "
                    f"{MODEL_ARCHITECTURE_VERSION}) and cannot be loaded. Pass a compatible "
                    "checkpoint with model_path=...; it needs a training_config.json next to it "
                    "that records the architecture version.")
            local = Path.home() / '.deepcell/models'
            for name in (DEFAULT_PRETRAINED, 'training_config.json'):
                hf_hub_download(repo_id='vanvalenlab/torch-spots', filename=name, local_dir=local)
            model_path = local / DEFAULT_PRETRAINED

        self.model_path = model_path
        self.model = load_spotnet(self.model_path, device=self.device)
        self.input_shape = 128
    
    def get_spot_probabilities(
            self,
            transforms,
            threshold=0.9,
            min_distance=1,
            pool_size=0
    ):
        """Hand the detection output for one field of view to the decoding module.

        Candidate locations are the local maxima of the pixel-wise maximum, over all N images,
        of the spot-class probability. The probability of each image is then read at every
        candidate.

        Args:
            transforms (dict): Output of ``predict_transforms`` for the N images of the stack, in
                the codebook's column order. Only the spot-class probability (channel 1 of
                ``transforms['detections']``) is used.
            threshold (float): Candidate threshold: peaks of the projection above it are kept.
                Independent of the evaluation threshold (``EVAL_THRESHOLD``). Defaults to 0.9.
            min_distance (int): Minimum separation between candidates, in pixels. Defaults to 1.
            pool_size (int): Take the maximum over the ``(2 * pool_size + 1) ** 2`` window around
                each candidate instead of the pixel itself. Defaults to 0.

        Returns:
            dict: ``'locations'``, integer ``[y, x]`` (row, column) of shape ``[n, 2]``;
            ``'probabilities'``, spot probabilities in [0, 1] of shape ``[n, N]``, column *i*
            from image *i*; ``'settings'``, the threshold, min_distance and pool_size used.
        """

        spot_probs = transforms['detections'][:, 1:2] # Shape N, 1, H, W
        prob_mips = np.max(spot_probs, axis=0) # Shape 1, H, W

        locations = max_cp_array_to_point_list_max(prob_mips, threshold=threshold, min_distance=min_distance)

        probabilities = extract_spots_prob_from_coords_maxpool(spot_probs, locations, pool_size=pool_size)
        probabilities = np.concatenate(probabilities, axis=1)

        return {
            'locations': locations,
            'probabilities': probabilities,
            'settings': {
                'threshold': threshold,
                'min_distance': min_distance,
                'pool_size': pool_size,
            },
        }


    def _y_annotations_to_point_list_max(self, y_pred, threshold=0.95, min_distance=2):
        """Convert raw prediction to a predicted point list using
        ``skimage.feature.peak_local_max`` to determine local maxima in classification
        prediction image, and their corresponding regression values will be used to
        create a final spot position prediction which will be added to the output spot
        center coordinates list.

        Args:
            y_pred (array): a dictionary of predictions with keys `'classification'` and
                `'offset_regression'` corresponding to the named outputs of the
                ``dot_net_2D model``.
            threshold (float): a number in ``[0, 1]``. Pixels with classification
                score > `threshold` are considered as containing a spot center.
            min_distance (float): the minimum distance between detected spots in pixels.

        Returns:
            array: spot center coordinates of the format [[y0, x0], [y1, x1],...]
        """
        if not isinstance(y_pred, dict):
            raise TypeError('Input predictions must be a dictionary.')

        dot_centers = []

        for ind in range(np.shape(y_pred['detections'])[0]):
            # iterate through individual images in this batch

            dot_pixel_inds = peak_local_max(y_pred['detections'][ind, 1],
                                            min_distance=min_distance,
                                            threshold_abs=threshold)

            delta_y = y_pred['offsets'][ind,0]
            delta_x = y_pred['offsets'][ind,1]

            dot_temp = np.zeros_like(dot_pixel_inds, dtype=np.float32)

            for i, (y_ind, x_ind) in enumerate(dot_pixel_inds):
                dot_temp[i, 0] = y_ind + delta_y[y_ind, x_ind]
                dot_temp[i, 1] = x_ind + delta_x[y_ind, x_ind]
            dot_centers.append(dot_temp)

        return np.vstack(dot_centers)
    
    @staticmethod
    def _preprocess(X, clip=False):
        """Preprocess each full (C, H, W) image of a (B, C, H, W) batch, before tiling."""
        X = np.asarray(X)
        if X.ndim != 4:
            raise ValueError(f"Image data must be 4D (B, C, H, W), got shape {X.shape}")
        return np.stack([preprocess_image(img, clip=clip) for img in X]).astype(np.float32)

    def predict_transforms(
            self,
            X,
            batch_size=20,
            clip=False,
        ):
        """Input X of shape (B, C, H, W), raw images.

        Each image is preprocessed with ``preprocess_image`` (min-max scaling, the same
        transform training uses) before it is tiled. ``clip=True`` first clips each image
        and channel to its 0.01st and 99.9th percentiles, as the original application does.
        """

        # Preprocess
        X = self._preprocess(X, clip=clip)
        tiles, tile_info = tile_input(X, (self.input_shape,self.input_shape))

        # Send data to device
        X = torch.tensor(tiles, device=self.device)

        batches = torch.split(X, batch_size)

        transforms = {
            'offsets': [],
            'detections': []
        }
        
        for batch in batches:

            # Infer
            with torch.inference_mode():
                pred = self.model(batch)
            

            for k,v in pred.items():
                transforms[k].append(v.cpu().numpy())

        for k,v in transforms.items():
            transforms[k] = np.concatenate(v)

        for k,v in transforms.items():
            transforms[k] = untile_output(v, tile_info)    

        return transforms

    
    def _eval_predict(
            self,
            X,
            threshold=EVAL_THRESHOLD,
            min_distance=EVAL_MIN_DISTANCE
    ):
        
        """Evaluation-path prediction (benchmark-evaluation).

        Input of X in shape (B, C, H, W), raw images.
        threshold: spot-probability threshold for local maxima (default: the reference threshold, ``EVAL_THRESHOLD``)
        min_distance: minimum distance between predicted peaks (default: reference 1 px)

        Images are preprocessed per image with ``preprocess_image``, the same function the
        training dataset uses, so evaluation sees exactly what training saw.
        """

        # Preprocess exactly like the training data path
        X = self._preprocess(X)

        # Send data to device
        X = torch.tensor(X, device=self.device, dtype=torch.float32)

        # Infer
        with torch.inference_mode():
            transforms = self.model(X)

        for k,v in transforms.items():
            transforms[k] = v.cpu().numpy()

        # Postprocess
        points = self._y_annotations_to_point_list_max(transforms, threshold=threshold, min_distance=min_distance)        

        return points

    def _local_points_to_absolute_batch(self, points_per_tile, tile_info, tile_indices):
        """
        Convert local (y, x) points from multiple tiles into absolute
        coordinates in the original (unpadded) image.

        points_per_tile: list of (N_i, 2) arrays, one per tile, in local
                        [0, tile_size] frame -- e.g. output of peak_local_max
                        + offset regression for each tile in a batch
        tile_info: dict returned by tile_input
        tile_indices: list/array of indices into tile_info's lists,
                    one per entry in points_per_tile (i.e. which tile
                    each set of points came from)

        Returns:
            (N_total, 2) array of absolute [y, x] points, and a parallel
            (N_total,) array indicating which tile_idx each point came from
        """

        ## X and Y are swapped in tile info
        x_offset_all = np.array(tile_info['y_starts']) - tile_info['pad_y'][0]
        y_offset_all = np.array(tile_info['x_starts']) - tile_info['pad_x'][0]

        abs_points = []
        tile_ids = []

        for pts, t_idx in zip(points_per_tile, tile_indices):
            pts = np.asarray(pts, dtype=np.float32)
            if pts.shape[0] == 0:
                continue
            shifted = pts.copy()
            shifted[:, 0] += y_offset_all[t_idx]
            shifted[:, 1] += x_offset_all[t_idx]
            abs_points.append(shifted)
            tile_ids.append(np.full(pts.shape[0], t_idx))

        if not abs_points:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=int)

        return np.vstack(abs_points)
    
    def predict_points(
            self,
            X,
            threshold=0.99,
            min_distance=1,
            batch_size=20,
            clip=False,
    ):
        
        """Input of X in shape (B, 1, H, W)
        threshold of finding local maxima
        min distance between the points predicted
        clip: clip each image to its 0.01/99.9 percentiles before scaling (the original
            application's behaviour). Off by default, so inference matches training.
        """

        # Preprocess
        X = self._preprocess(X, clip=clip)
        tiles, tile_info = tile_input(X, (self.input_shape, self.input_shape))

        X = torch.tensor(tiles, device=self.device)
        batches = torch.split(X, batch_size)

        points_per_tile = []
        tile_indices = []
        counter = 0

        for batch in batches:
            with torch.inference_mode():
                transforms = self.model(batch)

            for k, v in transforms.items():
                transforms[k] = v.cpu().numpy()

            # Get LOCAL points per tile in this batch (no offset applied here)
            for i in range(batch.shape[0]):
                single = {
                    'detections': transforms['detections'][i:i+1],
                    'offsets': transforms['offsets'][i:i+1],
                }
                # threshold/min_distance peak-finding, still in local frame
                local_pts = self._y_annotations_to_point_list_max(
                    single, threshold=threshold, min_distance=min_distance
                )
                points_per_tile.append(local_pts)
                tile_indices.append(counter)
                counter += 1

        # Convert all tiles' local points to absolute coordinates at once
        abs_points = self._local_points_to_absolute_batch(
            points_per_tile, tile_info, tile_indices
        )

        return abs_points
        