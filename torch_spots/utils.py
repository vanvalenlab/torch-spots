"""Variational Inference functions for spot decoding. Code adapted from PoSTcode
https://github.com/gerstung-lab/postcode (https://doi.org/10.1101/2021.10.12.464086)."""

import scipy
import torch
import numpy as np
from tqdm import tqdm
import pyro
from pyro.distributions import (RelaxedBernoulli, Categorical, constraints,
                                MultivariateNormal, Bernoulli)
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from torch.autograd import Function

assert pyro.__version__.startswith('1')


import numpy as np
from skimage import measure
from skimage.feature import peak_local_max

########################################################################
#                          Decoding functions                          #
########################################################################

def reshape_torch_array(torch_array):
    """Reshape a ``[k, r, c]`` array into a ``[k, r * c]``.

    Args:
        torch_array (torch.tensor): Array to be reshaped.

    Returns:
        torch.tensor: Reshaped array.
    """
    return torch_array.transpose(1, 2).reshape(torch_array.shape[0], -1)


def normalize_spot_values(data):
    """Normalizes spot intensity data array such that log of data has a mean of zero and standard
    deviation of one.
    
    Args:
        data (torch.tensor): Input data formatted as torch array with shape ``[num_spots, r * c]``.
    
    Returns:
        torch.tensor: Normalized data array.
    """
    # TODO: add clipping functionality

    s = torch.tensor(np.percentile(data.cpu().numpy(), 60, axis=0))
    max_s = torch.tensor(np.percentile(data.cpu().numpy(), 99.9, axis=0))
    min_s = torch.min(data, dim=0).values
    eps = 1e-6*max_s
    log_add = (s ** 2 - max_s * min_s) / (max_s + min_s - 2 * s)
    log_add = torch.max(-torch.min(data, dim=0).values + eps,
                        other=log_add.float() + eps)
    data_log = torch.log10(data + log_add)
    data_log_mean = data_log.mean(dim=0, keepdim=True)
    data_log_std = data_log.std(dim=0, keepdim=True)
    data_norm = (data_log - data_log_mean) / data_log_std  # column-wise normalization

    return data_norm


def chol_sigma_from_vec(sigma_vec, dim):
    L = torch.zeros(dim, dim)
    L[np.tril_indices(dim)] = sigma_vec

    return torch.mm(L, torch.t(L))


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float64)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float64)
            gm = grad_output.data.cpu().numpy().astype(np.float64)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input

mat_sqrt_ = MatrixSquareRoot.apply
def mat_sqrt(A):
    return mat_sqrt_(A)


def instantiate_rb_params(r, c, codes, params_mode):
    """Instantiates parameters for model of mixture of Relaxed Bernoulli distributions.

    Args:
        r (int): Number of rounds.
        c (int): Number of channels.
        codes (torch.tensor): Codebook formatted as torch array with shape
            ``[num_barcodes + 1, r * c]``.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds. valid options: ``['2', '2*R', '2*C', '2*R*C']``.

    Returns:
        scaled_sigma (torch.tensor): Sigma parameter of Relaxed Bernoulli.
        aug_temperature (torch.tensor): Temperature parameter of Relaxed Bernoulli.
    """

    if params_mode == '2':
        # two param - one for 0-channel, one for 1-channel
        sigma = pyro.param("sigma", torch.ones(torch.Size(
            [2])) * 0.5, constraint=constraints.unit_interval)
        aug_sigma = torch.gather(
            sigma, 0, codes.reshape(-1).long()).reshape(codes.shape)
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma  # is 0.3 an arbitrary choice?
        temperature = pyro.param('temperature', torch.ones(
            torch.Size([2])) * 0.5, constraint=constraints.unit_interval)
        aug_temperature = torch.gather(
            temperature, 0, codes.reshape(-1).long()).reshape(codes.shape)
    elif params_mode == '2*R':
        # 2*R params
        sigma = pyro.param("sigma", torch.ones(torch.Size(
            [2, r])) * 0.5, constraint=constraints.unit_interval)
        sigma_temp = sigma.unsqueeze(-1).repeat(1, 1, c)
        sigma_temp1 = reshape_torch_array(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature = pyro.param("temperature", torch.ones(
            torch.Size([2, r])) * 0.5, constraint=constraints.unit_interval)
        temperature_temp = temperature.unsqueeze(-1).repeat(1, 1, c)
        temperature_temp1 = reshape_torch_array(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*C':
        # 2*C params
        sigma = pyro.param("sigma", torch.ones(torch.Size(
            [2, c])) * 0.5, constraint=constraints.unit_interval)
        sigma_temp = sigma.unsqueeze(-1).repeat(1, 1, r)
        sigma_temp1 = reshape_torch_array(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature = pyro.param("temperature", torch.ones(
            torch.Size([2, c])) * 0.5, constraint=constraints.unit_interval)
        temperature_temp = temperature.unsqueeze(-1).repeat(1, 1, r)
        temperature_temp1 = reshape_torch_array(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*R*C':
        # 2*R*C params
        sigma = pyro.param("sigma", torch.ones(torch.Size(
            [2, r * c])) * 0.5, constraint=constraints.unit_interval)
        aug_sigma = torch.gather(sigma, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature = pyro.param("temperature", torch.ones(
            torch.Size([2, r * c])) * 0.5, constraint=constraints.unit_interval)
        aug_temperature = torch.gather(temperature, 0, codes.long())
    else:
        assert False, "%s not supported" % params_mode

    return scaled_sigma, aug_temperature


@config_enumerate
def model_constrained_tensor(
        data,
        codes,
        c,
        r,
        batch_size=None,
        distribution='Relaxed Bernoulli',
        params_mode='2*R*C'):
    """Model definition: relaxed bernoulli, paramters are shared across all genes, but might
    differ across channels or rounds.

    Args:
        data (torch.tensor): Input data formatted as torch array with shape `[num_spots, r * c]`.
        codes (torch.tensor): Codebook formatted as torch array with shape
            `[num_barcodes + 1, r * c]`.
        c (int): Number of channels.
        r (int): Number of rounds.
        batch_size (int): Size of batch for training.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds for model of Relaxed Bernoulli distributions, or model of Gaussians.
            Valid options: `['2', '2*R', '2*C', '2*R*C']`. Defaults to `'2*R*C'`. 

    Returns:
        None
    """
    k = codes.shape[0]
    w = pyro.param('weights', torch.ones(k) / k, constraint=constraints.simplex)

    scaled_sigma, aug_temperature = instantiate_rb_params(r, c, codes, params_mode)

    with pyro.plate('data', data.shape[0], batch_size) as batch:
        z = pyro.sample('z', Categorical(w))
        pyro.sample(
            'obs',
            RelaxedBernoulli(
                temperature=aug_temperature[z],
                probs=scaled_sigma[z]).to_event(1),
            obs=data[batch])


def train(svi, num_iter, data, codes, c, r, batch_size, distribution, params_mode):
    """Do the training for SVI model.

    Args:
        svi (pyro.infer.SVI): stochastic variational inference model.
        num_iter (int): Number of iterations for training.
        data (torch.tensor): Input data formatted as torch array with shape `[num_spots, r * c]`.
        codes (torch.tensor): Codebook formatted as torch array with shape
            `[num_barcodes + 1, r * c]`.
        c (int): Number of channels.
        r (int): Number of rounds.
        batch_size (int): Size of batch for training.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds for model of Relaxed Bernoulli distributions, or model of Gaussians.
            Valid options: `['2', '2*R', '2*C', '2*R*C']`. Defaults to `'2*R*C'`. 

    Returns:
        list: losses.

    """
    pyro.clear_param_store()
    losses = []
    for _ in tqdm(range(num_iter)):
        loss = svi.step(data, codes, c, r, batch_size, distribution, params_mode)
        losses.append(loss)
    return losses


def rb_e_step(data, codes, w, temperature, sigma, c, r, params_mode='2*R*C'):
    """Estimate the posterior probability for spot assignment from a mixture of Relaxed
    Bernoulli distributions.

    Args:
        data (torch.tensor): Input data formatted as torch array with shape `[num_spots, r * c]`.
        codes (torch.tensor): Codebook formatted as torch array with shape
            `[num_barcodes + 1, r * c]`.
        w (torch.array): Weight parameter with length `num_barcodes + 1`.
        temperature (torch.array): Temperature parameter for Relaxed Bernoulli, shape depends on
             `params_mode`.
        sigma (torch.array): Sigma parameter for Relaxed Bernoulli, shape depends on `params_mode`.
        c (int): Number of channels.
        r (int): Number of rounds.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds. Valid options: `['2', '2*R', '2*C', '2*R*C']`.

    Returns:
        normalized class probability with shape `[num_spots, num_barcodes + 1]`.
    """
    K = codes.shape[0]  # num_barcodes + 1
    class_logprobs = np.ones((data.shape[0], K))

    if params_mode == '2':  # two params
        aug_sigma = torch.gather(
            sigma, 0, codes.reshape(-1).long()).reshape(codes.shape)
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        aug_temperature = torch.gather(
            temperature, 0, codes.reshape(-1).long()).reshape(codes.shape)
    elif params_mode == '2*R':  # 2*R params
        sigma_temp = sigma.unsqueeze(-1).repeat(1, 1, c)
        sigma_temp1 = reshape_torch_array(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature_temp = temperature.unsqueeze(-1).repeat(1, 1, c)
        temperature_temp1 = reshape_torch_array(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*C':  # 2*C params
        sigma_temp = sigma.unsqueeze(-1).repeat(1, 1, r)
        sigma_temp1 = reshape_torch_array(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature_temp = temperature.unsqueeze(-1).repeat(1, 1, r)
        temperature_temp1 = reshape_torch_array(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*R*C':  # 2*R*C params
        aug_sigma = torch.gather(sigma, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        aug_temperature = torch.gather(temperature, 0, codes.long())
    else:
        assert False, "%s not supported" % params_mode

    batch_sz = 50000
    for idx in range(len(data) // batch_sz + 1):
        ind_start = idx * batch_sz
        ind_end = (idx+1) * batch_sz
        if len(data[ind_start:ind_end]) == 0:
            break
        for k in range(K):
            dist = RelaxedBernoulli(
                temperature=aug_temperature[k],
                probs=scaled_sigma[k]).to_event(1)
            class_logprobs[ind_start:ind_end, k] = (
                w[k].log() + dist.log_prob(data[ind_start:ind_end])).cpu().numpy()

    # basically doing a stable_softmax here
    numerator = np.exp(class_logprobs - np.max(class_logprobs, axis=1)[:, None])
    class_prob_norm = np.divide(numerator, np.sum(numerator, axis=1)[:, None])

    return class_prob_norm


def decoding_function(spots,
                      barcodes,
                      num_iter=500,
                      batch_size=15000,
                      set_seed=1,
                      distribution='Relaxed Bernoulli',
                      params_mode='2*R*C'):
    """Main function for the spot decoding.

    Args:
        spots (numpy.array): Input spot intensity array with shape `[num_spots, c, r]`.
        barcodes (numpy.array): Input codebook array with shape `[num_barcodes, c, r]`.
        num_iter (int): Number of iterations for training. Defaults to 500.
        batch_size (int): Size of batch for training. Defaults to 15000.
        set_seed (int): Seed for randomness. Defaults to 1.
        distribution (str): Distribution for spot intensities in spot decoding model. Valid options:
            `['Gaussian', 'Bernoulli', 'Relaxed Bernoulli']`. Defaults to `'Relaxed Bernoulli'`.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds for model of Relaxed Bernoulli distributions, or model of Gaussians.
            Valid options: `['2', '2*R', '2*C', '2*R*C']`. Defaults to `'2*R*C'`. 

    Raises:
        ValueError: `distribution` must be one of `['2', '2*R', '2*C', '2*R*C']`.
        ValueError: `params_mode` must be one of `['Relaxed Bernoulli', 'Bernoulli', 'Gaussian']`.
    
    Returns:
        results (dict): The decoding results as a dictionary: `'class_probs'`: posterior
            probabilities for each spot and each gene category; `'params'`: estimated model
            parameters.
    """
    # if cuda available, runs on gpu
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    valid_distributions = ['Relaxed Bernoulli', 'Bernoulli', 'Gaussian']
    if distribution not in valid_distributions:
            raise ValueError('Invalid params_mode supplied: {}. '
                             'Must be one of {}'.format(distribution,
                                                        valid_distributions))

    valid_params_modes = ['2', '2*R', '2*C', '2*R*C']
    if params_mode not in valid_params_modes:
            raise ValueError('Invalid params_mode supplied: {}. '
                             'Must be one of {}'.format(params_mode,
                                                        valid_params_modes))

    num_spots, c, r = spots.shape

    data = reshape_torch_array(torch.tensor(spots).float())
    codes = reshape_torch_array(torch.tensor(barcodes).float())

    auto_guide_constrained_tensor = AutoDelta(poutine.block(model_constrained_tensor,
                                                expose=['weights',
                                                        'temperature',
                                                        'sigma']))

    optim = Adam({'lr': 0.085, 'betas': [0.85, 0.99]})
    svi = SVI(model_constrained_tensor, auto_guide_constrained_tensor,
              optim, loss=TraceEnum_ELBO(max_plate_nesting=1))
    pyro.set_rng_seed(set_seed)

    print('Training...')
    losses = train(svi, num_iter, data, codes, c, r,
                   min(num_spots, batch_size), distribution, params_mode)

    print('Estimating barcode probabilities...')
    
    if distribution=='Relaxed Bernoulli':

        w_star = pyro.param('weights').detach()
        temperature_star = pyro.param('temperature').detach()
        sigma_star = pyro.param('sigma').detach()

        class_probs_star = rb_e_step(
            data, codes, w_star, temperature_star, sigma_star, c, r, params_mode)

        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.FloatTensor")

        torch_params = {
            'w_star': w_star.cpu(),
            'temperature_star': temperature_star.cpu(),
            'sigma_star': sigma_star.cpu(),
            'losses': losses
        }

    results = {'class_probs': class_probs_star,
               'params': torch_params}

    return results

########################################################################
#                      Postprocess functions                           #
########################################################################

def y_annotations_to_point_list(y_pred, threshold=0.95):
    """Convert raw prediction to a predicted point list: classification of
    pixel as containing dot > `threshold`, , and their corresponding regression
    values will be used to create a final spot position prediction which will
    be added to the output spot center coordinates list.

    Args:
        y_pred (array): a dictionary of predictions with keys `'classification'` and
            `'offset_regression'` corresponding to the named outputs of the
            ``dot_net_2D model``.
        ind (int): the index of the image in the batch for which to convert the
            annotations.
        threshold (float): a number in ``[0, 1]``. Pixels with classification
            score > `threshold` are considered containing a spot center.

    Returns:
        array: spot center coordinates of the format ``[[y0, x0], [y1, x1],...]``
    """
    if not isinstance(y_pred, dict):
        raise TypeError('Input predictions must be a dictionary.')

    dot_centers = []
    for ind in range(np.shape(y_pred['detections'])[0]):
        contains_dot = y_pred['detections'][ind, ..., 1] > threshold
        delta_y = y_pred['offsets'][ind, ..., 0]
        delta_x = y_pred['offsets'][ind, ..., 1]

        dot_pixel_inds = np.argwhere(contains_dot)
        dot_centers.append([[y_ind + delta_y[y_ind, x_ind], x_ind +
                             delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds])

    return np.array(dot_centers)


def y_annotations_to_point_list_restrictive(y_pred, threshold=0.95):
    """Convert raw prediction to a predicted point list: classification of
    pixel as containing dot > `threshold` AND center regression is contained
    in the pixel. The corresponding regression values will be used to create
    a final spot position prediction which will be added to the output spot
    center coordinates list.

    Args:
        y_pred (array): a dictionary of predictions with keys `'classification'` and
            `'offset_regression'` corresponding to the named outputs of the
            ``dot_net_2D model``.
        ind (int): the index of the image in the batch for which to convert the
            annotations.
        threshold (float): a number in ``[0, 1]``. Pixels with classification
            score > `threshold` are considered containing a spot center.

    Returns:
        array: spot center coordinates of the format ``[[y0, x0], [y1, x1],...]``.
    """
    if not isinstance(y_pred, dict):
        raise TypeError('Input predictions must be a dictionary.')

    dot_centers = []
    for ind in range(np.shape(y_pred['classification'])[0]):
        contains_dot = y_pred['classification'][ind, 1] > threshold
        delta_y = y_pred['offset_regression'][ind, 0]
        delta_x = y_pred['offset_regression'][ind, 1]
        contains_its_regression = (abs(delta_x) <= 0.5) & (abs(delta_y) <= 0.5)

        final_dot_detection = contains_dot & contains_its_regression

        dot_pixel_inds = np.argwhere(final_dot_detection)
        dot_centers.append(np.array(
            [[y_ind + delta_y[y_ind, x_ind],
              x_ind + delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds]))

    return np.array(dot_centers)


def y_annotations_to_point_list_max(y_pred, threshold=0.95, min_distance=2):
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


def max_cp_array_to_point_list_max(max_cp_array, threshold=0.95, min_distance=2):
    """Convert raw prediction to a predicted point list using
    ``skimage.feature.peak_local_max`` to determine local maxima in classification
    prediction image, and their corresponding regression values will be used to
    create a final spot position prediction which will be added to the output spot
    center coordinates list. This is performed on the max projected cp_array.

    Args:
        max_cp_array (array): (batch, x, y)
        threshold (float): A number in ``[0, 1]``. Pixels with classification
            score > `threshold` are considered as containing a spot center.
        min_distance (float): The minimum distance between detected spots in pixels.

    Returns:
        array: Spot center coordinates, list (num_images,) with each entry shape
        (num_spots, 2).
    """
    dot_centers = []
    for ind in range(np.shape(max_cp_array)[0]):
        dot_pixel_inds = peak_local_max(max_cp_array[ind, ...],
                                        min_distance=min_distance,
                                        threshold_abs=threshold)
        dot_centers.append(dot_pixel_inds)

    return dot_centers


def y_annotations_to_point_list_cc(y_pred, threshold=0.95):
    """Convert raw prediction to a predicted point list: classification of
    connected component as containing dot > `threshold`, , and their corresponding
    regression values will be used to create a final spot position prediction which
    will be added to the output spot center coordinates list.

    Args:
        y_pred (array): a dictionary of predictions with keys `'classification'` and
            `'offset_regression'` corresponding to the named outputs of the
            ``dot_net_2D model``.
        threshold (float): a number in ``[0, 1]``. Pixels with classification
            score > `threshold` are considered containing a spot center.

    Returns:
        array: spot center coordinates of the format ``[[y0, x0], [y1, x1],...]``
    """
    if not isinstance(y_pred, dict):
        raise TypeError('Input predictions must be a dictionary.')
    if 'classification' not in y_pred.keys() or 'offset_regression' not in y_pred.keys():
        raise NameError('Input must have keys \'classification\' and \'offset_regression\'')

    dot_centers = []
    for ind in range(np.shape(y_pred['classification'])[0]):

        delta_y = y_pred['offset_regression'][ind, 0]
        delta_x = y_pred['offset_regression'][ind, 1]

        blobs = y_pred['classification'][ind, 1] > threshold
        label_image = measure.label(blobs, background=0)
        rp = measure.regionprops(label_image)

        dot_centers_temp = []
        for region in rp:
            region_pixel_inds = region.coords
            reg_pred = [[y_ind + delta_y[y_ind, x_ind], x_ind + delta_x[y_ind, x_ind]]
                        for y_ind, x_ind in region_pixel_inds]
            dot_centers_temp.append(np.mean(reg_pred, axis=0))

        dot_centers.append(dot_centers_temp)
    return np.array(dot_centers)
