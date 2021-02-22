# taken and modified from https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX#scrollTo=JovhcSy1NIhr
# will need to be refactored from 3d input to 5d (with ray direction)

import torch
import torch.nn.functional as F
from einops import repeat, rearrange

def meshgrid_xy(tensor1, tensor2):
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)

def cumprod_exclusive(tensor):
    cumprod = torch.cumprod(tensor, dim = -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.
    return cumprod

def get_ray_bundle(height, width, focal_length, tform_cam2world):
    ii, jj = meshgrid_xy(
      torch.arange(width).to(tform_cam2world),
      torch.arange(height).to(tform_cam2world)
    )

    directions = torch.stack([(ii - width * .5) / focal_length,
                            -(jj - height * .5) / focal_length,
                            -torch.ones_like(ii)
                           ], dim=-1)
    ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions

def compute_query_points_from_rays(
    ray_origins,
    ray_directions,
    near_thresh,
    far_thresh,
    num_samples,
    randomize = True
):
    depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
    if randomize is True:
        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        depth_values = depth_values \
            + torch.rand(noise_shape).to(ray_origins) * (far_thresh
                - near_thresh) / num_samples
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    return query_points, depth_values

def render_volume_density(
    radiance_field,
    ray_origins,
    depth_values
):
    sigma_a = F.relu(radiance_field[..., 3])
    rgb = torch.sigmoid(radiance_field[..., :3])
    one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],
                  one_e_10.expand(depth_values[..., :1].shape)), dim=-1)
    alpha = 1. - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(-1)

    return rgb_map, depth_map, acc_map

def get_image_from_nerf_model(
    model,
    latents,
    height,
    width,
    focal_length = 140,
    tform_cam2world = torch.eye(4),
    near_thresh = 2.,
    far_thresh = 6.,
    depth_samples_per_ray = 32
):
    tform_cam2world = tform_cam2world.to(latents)

    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length,
                                               tform_cam2world)

    query_points, depth_values = compute_query_points_from_rays(
      ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    flattened_query_points = query_points.reshape((-1, 3))

    images = []
    for latent in latents.unbind(0):
        predictions = []
        predictions.append(model(latent, flattened_query_points))

        radiance_field_flattened = torch.cat(predictions, dim=0)

        unflattened_shape = list(query_points.shape[:-1]) + [4]
        radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

        rgb_predicted, _, _ = render_volume_density(radiance_field, ray_origins, depth_values)
        image = rearrange(rgb_predicted, 'h w c -> c h w')
        images.append(image)

    return torch.stack(images)
