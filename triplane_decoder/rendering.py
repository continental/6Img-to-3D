import torch
import triplane_decoder.ray_samplers as ray_sampler
from triplane_decoder.losses import distortion_loss
from triplane_decoder.pif import PIF
from triplane_decoder.decoder import TriplaneDecoder

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1) #rays, #samples, 1
    return torch.cat((torch.ones((accumulated_transmittance.size(0), 1, 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=1)


def ray_aabb_intersection(ray_origins, ray_directions, aabb_min, aabb_max):
    tmin = (aabb_min - ray_origins) / ray_directions
    tmax = (aabb_max - ray_origins) / ray_directions
    
    t1 = torch.min(tmin, tmax)
    t2 = torch.max(tmin, tmax)
    
    t_enter = torch.max(t1,dim=1).values
    t_exit = torch.min(t2,dim=1).values
    
    return t_enter, t_exit

def render_rays(nerf_model:TriplaneDecoder, ray_origins, ray_directions, config, triplane=None, pif: PIF = None,  training=True, only_coarse=False, **kwargs):

    device = ray_origins.device
    
    uniform_sampler = ray_sampler.UniformSampler(num_samples=config.decoder["nb_bins"], train_stratified=config.decoder["train_stratified"])
    pdf_sampler = ray_sampler.PDFSampler(num_samples=config.decoder["nb_bins"], train_stratified=config.decoder["train_stratified"], include_original=False)

    uniform_sampler.training = training
    pdf_sampler.training = training

    ray_bundle = ray_sampler.RayBundle(
        origins=ray_origins,
        directions=ray_directions,
        nears=torch.ones((ray_origins.size(0),1), device=device) * config.decoder["hn"],
        fars=torch.ones((ray_origins.size(0),1), device=device)  * config.decoder["hf"],
    )

    # Coarse sampling
    samples_coarse = uniform_sampler.generate_ray_samples(ray_bundle)

    midpoints = (samples_coarse.starts + samples_coarse.ends) / 2                 #rays, #samples, 1
    x = samples_coarse.origins + samples_coarse.directions.squeeze(2) * midpoints #rays, #samples, 3
    viewing_directions = ray_directions.expand(x.size(1), -1, 3).permute(1,0,2)   #rays, #samples, 3

    colors, densities = nerf_model(x.reshape(-1,3), viewing_directions.reshape(-1,3), pif=pif)
    colors_coarse = colors.reshape_as(x)               #rays, #samples, 3 
    densities_coarse = densities.reshape_as(midpoints) #rays, #samples, 1
    
    weights = samples_coarse.get_weights(densities_coarse) #rays, #samples, 1


    if only_coarse:
        colors = volume_rendering(samples_coarse.deltas,
                        colors_coarse, 
                        densities_coarse, 
                        config.decoder.white_background)

        dist_loss = distortion_loss(weights, samples_coarse)

        depth = get_depth(weights, midpoints)

        return colors, dist_loss, depth


    # Fine sampling
    samples_fine = pdf_sampler.generate_ray_samples(ray_bundle, samples_coarse, weights,  config.decoder["nb_bins"])

    midpoints = (samples_fine.starts + samples_fine.ends) / 2 #rays, #samples, 1
    x = samples_coarse.origins + samples_coarse.directions.squeeze(2) * midpoints #rays, #samples, 3

    colors, densities = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3), pif=pif)

    colors_fine = colors.reshape_as(x)               #rays, #samples_per_ray, 3 
    densities_fine = densities.reshape_as(midpoints) #rays, #samples_per_ray

    colors = volume_rendering(samples_fine.deltas,
                            colors_fine, 
                            densities_fine, 
                            config.decoder.white_background)

    weights = samples_fine.get_weights(densities_fine) #rays, #samples, 1
    dist_loss = distortion_loss(weights, samples_fine)

    depth = get_depth(weights, midpoints)

    return colors, dist_loss, depth

def volume_rendering(deltas: torch.Tensor, colors: torch.Tensor, sigma: torch.Tensor, white_background: bool) -> torch.Tensor:
    alpha = 1 - torch.exp(-sigma * deltas)  #rays, #samples, 1
    weights = compute_accumulated_transmittance(1 - alpha) * alpha #rays, #samples, 1
    colors = (weights * colors).sum(dim=1)  #rays, 3
    if white_background:
        weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
        colors = colors + 1 - weight_sum.unsqueeze(-1)  #samples, 3

    return colors #rays, 3


def get_depth(weights, steps):
    """
    https://docs.nerf.studio/_modules/nerfstudio/model_components/renderers.html#DepthRenderer
    """
    cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # [..., num_samples]
    split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5  # [..., 1]
    median_index = torch.searchsorted(cumulative_weights, split, side="left")  # [..., 1]
    median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1)  # [..., 1]
    median_depth = torch.gather(steps[..., 0], dim=-1, index=median_index)  # [..., 1]
    return median_depth
