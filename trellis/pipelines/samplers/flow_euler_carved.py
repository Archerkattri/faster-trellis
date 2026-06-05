"""Carved SLaT sampler (ported verbatim from fast-trellis v1, renamed _carved).

Token carving + delta-cache temporal skip. Used as the SLaT sampler in v1
faster-trellis carved config (HiCache SS + carved SLaT), mirroring v2."""
from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .flow_euler import FlowEulerSampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from ...modules import sparse as sp

# Faster sampler.
from token_slat.token_leader import TokenLeader
from token_slat.token_argparser import parse_token_args
from token_slat.selection import AdvancedStabilityTracker
from faster_utils_slat import faster_cal_type, faster_init
class FlowEulerSampler_carved(FlowEulerSampler):
    def __init__(
        self,
        sigma_min: float,
    ):
        super().__init__(sigma_min)

        self.LEADER = TokenLeader()
        self.stability_tracker = AdvancedStabilityTracker()
        self.args = parse_token_args()
        self.coords_scores = None

        # Set parameters. Env-overridable; carve 25% of the smoothest tokens
        # (vs upstream Fast-TRELLIS's 10%) — the carved hybrid has quality headroom
        # from the cheaper HiCache SS stage, matching the v2 config.
        import os
        self.thresh = float(os.environ.get("GF_CARVE_THRESH", 5.0))
        self.ret_steps = int(os.environ.get("GF_CARVE_RET_STEPS", 2))
        self.carving_ratio = float(os.environ.get("GF_CARVE_RATIO", 0.25))
        self.dir_weight = 0.5
        self.cache_dic, self.current = faster_init(25)
        self.cache_dic['thresh'] = self.thresh 
        self.cache_dic['dir_weight'] =  self.dir_weight
        self.cache_dic['first_enhance'] = self.ret_steps
    
    # Inject coords_scores.
    def set_coords_scores(self,coords_scores):
        self.coords_scores = coords_scores

    # Initialize cache.
    def _init_token_state(self, x_t_shape, device, args, model):
        self.LEADER.set_parameters(args)
        if hasattr(model, 'dtype'):
             self.model_dtype = model.dtype
        elif hasattr(model, 'parameters'):
            try:
                self.model_dtype = next(model.parameters()).dtype
            except StopIteration:
                pass
  
        self.stability_tracker.reset(device = device, num_tokens = x_t_shape[0],latent_channels = x_t_shape[1])
        self.stability_tracker.coords_scores = self.coords_scores


    # Override the base implementation.
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        sample = noise
        # Generate t_seq from 1 to 0.
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        self.cache_dic, self.current = faster_init(steps)
        self.cache_dic['thresh'] = self.thresh
        self.cache_dic['dir_weight'] = self.dir_weight
        self.cache_dic['first_enhance'] = self.ret_steps
        
  
        N,C = sample.feats.shape
        self.args.effective_steps = steps
        self.args.full_sampling_end_steps = int(np.ceil(steps * self.args.full_sampling_end_ratio))
        self.args.anchor_step = max(1, int(np.floor(steps * self.args.anchor_ratio)))
        self._init_token_state((N, C), sample.device, self.args, model)

        self.LEADER.total_tokens = N
        self.LEADER.schedule_is_set = True

        self.coords_raw = sample.coords
       
        # Iterate over each timestep.
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            cache = self.cache_dic['cache']
            self.current['is_token_active'] = False
            current_step = self.LEADER.current_step

            if self.current['use_token'] and cache['prev_v'] is not None and current_step >= self.LEADER.full_sampling_steps:
                self.current['num_to_skip'] = int(self.carving_ratio * N)
                if self.current['num_to_skip'] > 0 and self.current['num_to_skip'] < N:
                    self.current['is_token_active'] = True

            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)

            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):  
        output = None
        should_calc = faster_cal_type(self.cache_dic, self.current, x_t.feats)
        velocity = None

        if should_calc:
            if self.current['is_token_active'] and self.current['use_token']:
                coords_scores = self.stability_tracker.coords_scores

                self.current['cached_indices'],  self.current['fast_update_indices'] = self.stability_tracker.update_and_select_combined(self.cache_dic['cache']['prev_v'], self.current['num_to_skip'],t=0, coords_scores = coords_scores,spatial_weight=0.3)
                
                # Select tokens.
                x_input_feats = x_t.feats[self.current['fast_update_indices'], :] if self.current['is_token_active'] else x_t.feats
                x_input_coords = x_t.coords[self.current['fast_update_indices'], :] if self.current['is_token_active'] else x_t.coords
                x_input = sp.SparseTensor(feats=x_input_feats, coords=x_input_coords)
                
                pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_input, t, cond, **kwargs)
                velocity_feats  = pred_v.feats
            else:
                pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
                velocity_feats  = pred_v.feats
            

            # Restore the original shape.
            if self.current['is_token_active'] and self.current['use_token']:
                final_v_tokens = self.cache_dic['cache']['prev_v'].clone() 
                final_v_tokens[self.current['fast_update_indices'], :] = velocity_feats.to(final_v_tokens.dtype)
                velocity_feats = final_v_tokens

            prev_x = self.cache_dic['cache']['prev_x']
            prev_prev_x = self.cache_dic['cache']['prev_prev_x']
            prev_v = self.cache_dic['cache']['prev_v']
            k = self.cache_dic['cache']['k']

            if prev_x is not None and prev_prev_x is not None:
                output_change = (velocity_feats - prev_v).abs().mean()
                prev_input_change = (prev_x - prev_prev_x).abs().mean() + 1e-8
                current_k = output_change / prev_input_change
                
                if k is None:
                    self.cache_dic['cache']['k'] = current_k
                else:
                    self.cache_dic['cache']['k'] = 0.7 * k + 0.3 * current_k 
     
            if prev_x is not None:
                self.cache_dic['cache']['prev_prev_x'] = prev_x
            self.cache_dic['cache']['prev_x'] = x_t.feats.detach().clone()
            self.cache_dic['cache']['prev_v'] = velocity_feats.detach().clone()
            self.cache_dic['cache']['easy'] = velocity_feats - x_t.feats
        
        else:
            # Reuse cache.
            velocity_feats = x_t.feats + self.cache_dic['cache']['easy']
            self.cache_dic['cache']['prev_x'] = x_t.feats.detach().clone()
            self.cache_dic['cache']['prev_v'] = velocity_feats.detach().clone()

        velocity = sp.SparseTensor(
            feats=velocity_feats,
            coords=self.coords_raw,
        )
        pred_v = velocity
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        pred_x_prev = x_t - (t - t_prev) * pred_v

        output = edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})
        
        self.current['step'] += 1
        self.LEADER.increase_step()
        return output
    

    # Model forward pass.
    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        
        output = model(x_t, t, cond, **kwargs)
        return output


    # Single model forward pass plus data conversion.
    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

class FlowEulerGuidanceIntervalSampler_carved(GuidanceIntervalSamplerMixin, FlowEulerSampler_carved):
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        
        # Apply CFG.
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
