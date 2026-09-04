import numpy as np
import torch

from torch_spots.detection.dotnet import SpotNet
from skimage.feature import peak_local_max

import numpy as np
import glob

from pathlib import Path

from torch_spots.detection.utils import spotnet_preprocess, tile_input, untile_output, \
      max_cp_array_to_point_list_max, extract_spots_prob_from_coords_maxpool


class SpotDetection():

    def __init__(
            self,
            model_path=None,
            device=None
    ):
        
        self.model = SpotNet()

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        if model_path is None:
            
            from deepcell_auth import download_torch_spots_model
            download_torch_spots_model()

            canonical_path = Path.home() / ".deepcell/models"
            # Use latest version
            model_path = sorted(
                glob.glob(str(canonical_path / "torch-spots*.pth"))
            )[-1]
            
        self.model_path = model_path
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.eval().to(self.device)
        self.input_shape = 128
    
    def get_spot_intensities(
            self,
            transforms,
            threshold=0.9,
            min_distance=1,
            extra_pixel_num=0
    ):
        
        spot_probs = transforms['detections'][:, 1:2] # Shape R, C, H, W
        prob_mips = np.max(spot_probs, axis=0) # Shape C, H, W
        
        point_coord = max_cp_array_to_point_list_max(prob_mips, threshold=threshold, min_distance=min_distance)

        intensities = extract_spots_prob_from_coords_maxpool(spot_probs, point_coord, extra_pixel_num=extra_pixel_num)        
        intensities = np.concatenate(intensities, axis=1)

        return intensities, point_coord


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
    
    def predict_transforms(
            self,
            X,
            batch_size=20,
        ):
        
        # Preprocess
        X = spotnet_preprocess(X)
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
            threshold=0.99,
            min_distance=1
    ):
        
        """Input of X in shape (B, 1, H, W)
        threshold of finding local maxima
        min distance between the points predicted
        """

        # Preprocess
        X = spotnet_preprocess(X)

        # Send data to device
        X = torch.tensor(X, device=self.device)

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
    ):
        
        """Input of X in shape (B, 1, H, W)
        threshold of finding local maxima
        min distance between the points predicted
        """

        # Preprocess
        X = spotnet_preprocess(X)
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
        