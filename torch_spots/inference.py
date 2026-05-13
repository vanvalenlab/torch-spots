import numpy as np
import torch

from dotnet import SpotNet
from skimage.feature import peak_local_max

class SpotDetection():

    def __init__(
            self,
            model_path=None,
            device=None
    ):
        
        self.model_path = model_path
        self.model = SpotNet()

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.eval().to(self.device)


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
            dot_pixel_inds = peak_local_max(y_pred['detections'][ind, 1],
                                            min_distance=min_distance,
                                            threshold_abs=threshold)

            delta_y = y_pred['offsets'][ind,0]
            delta_x = y_pred['offsets'][ind,1]

            dot_centers.append(np.array(
                [[y_ind + delta_y[y_ind, x_ind],
                x_ind + delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds]))

        return np.array(dot_centers)
    
    def _minmax(self,x):

        xmin = np.min(x)
        xmax = np.max(x)
        x = (x - xmin) / (xmax - xmin)

        return x
    
    def _perc_clip(self, x):

        p1, p99 = np.percentile(x, (0.1, 99.9))
        return np.clip(x, p1, p99, dtype=np.float32)
    
    def _standardize(self, x):
        """Apply percentile clipping, min-max normalization, then subtraction at 0.5."""
        x = self._perc_clip(x)
        x = self._minmax(x)
        x = x - 0.5

        return x
    
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
        X = self._standardize(X)

        # Send data to device
        X = torch.tensor(X, device=self.device)

        # Infer
        with torch.inference_mode():
            transforms = self.model(X)

        for k,v in transforms.items():
            transforms[k] = v.cpu().numpy()

        # Postprocess
        points = self._y_annotations_to_point_list_max(transforms, threshold=threshold, min_distance=min_distance)
        
        if points.ndim > 2:
            points = points.squeeze()

        if points.ndim < 2:
            points = np.expand_dims(points, 0)
        
        return points

    def predict(self):



        return None