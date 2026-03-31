"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.aux_heads import AccentAdversarialHead, CTCAuxHead
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        use_distill: bool = False,
        lambda_distill_out: float = 0.5,
        lambda_distill_hidden: float = 0.25,
        distill_hidden_layers: list[int] | None = None,
        teacher_transformer: nn.Module | None = None,
        distill_temperature: float = 4.0,
        use_ctc: bool = False,
        lambda_ctc: float = 0.05,
        ctc_on_generated_only: bool = True,
        ctc_layer_index: int = -2,
        use_accent_adv: bool = False,
        lambda_adv: float = 0.01,
        adv_num_classes: int = 0,
        adv_pooling: str = "masked_mean",
        adv_feature_layer: int = -1,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

        # optional experiment flags
        self.use_distill = use_distill
        self.lambda_distill_out = lambda_distill_out
        self.lambda_distill_hidden = lambda_distill_hidden
        self.distill_hidden_layers = distill_hidden_layers or []
        self.distill_temperature = distill_temperature
        self.use_ctc = use_ctc
        self.lambda_ctc = lambda_ctc
        self.ctc_on_generated_only = ctc_on_generated_only
        self.ctc_layer_index = ctc_layer_index
        self.use_accent_adv = use_accent_adv
        self.lambda_adv = lambda_adv
        self.adv_feature_layer = adv_feature_layer

        self.last_loss_dict = {
            "loss_total": torch.tensor(0.0),
            "loss_flow": torch.tensor(0.0),
            "loss_distill_out": torch.tensor(0.0),
            "loss_distill_hidden": torch.tensor(0.0),
            "loss_ctc": torch.tensor(0.0),
            "loss_adv": torch.tensor(0.0),
        }

        self.teacher_transformer = teacher_transformer
        if self.teacher_transformer is not None:
            self.teacher_transformer.eval()
            for p in self.teacher_transformer.parameters():
                p.requires_grad = False

        need_hidden = (self.use_distill and self.lambda_distill_hidden > 0) or self.use_ctc or self.use_accent_adv
        if hasattr(self.transformer, "set_capture_hidden_for_distill"):
            self.transformer.set_capture_hidden_for_distill(need_hidden)
        if self.teacher_transformer is not None and hasattr(self.teacher_transformer, "set_capture_hidden_for_distill"):
            self.teacher_transformer.set_capture_hidden_for_distill(need_hidden)

        vocab_size = len(vocab_char_map) if exists(vocab_char_map) else 256
        self.ctc_head = CTCAuxHead(self.dim, vocab_size) if self.use_ctc else None
        self.accent_head = (
            AccentAdversarialHead(self.dim, adv_num_classes, pooling=adv_pooling)
            if self.use_accent_adv and adv_num_classes > 0
            else None
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
        accent_id: torch.Tensor | None = None,
        lang_id: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )

        # flow matching loss
        flow_loss = F.mse_loss(pred, flow, reduction="none")
        flow_loss = flow_loss[rand_span_mask].mean()

        zero = flow_loss.new_zeros(())
        distill_out_loss = zero
        distill_hidden_loss = zero
        ctc_loss = zero
        adv_loss = zero

        student_hidden = getattr(self.transformer, "last_hidden_states", None)

        if (
            self.use_distill
            and (self.lambda_distill_out > 0 or self.lambda_distill_hidden > 0)
            and self.teacher_transformer is not None
        ):
            with torch.no_grad():
                teacher_pred = self.teacher_transformer(
                    x=φ,
                    cond=cond,
                    text=text,
                    time=time,
                    drop_audio_cond=drop_audio_cond,
                    drop_text=drop_text,
                    mask=mask,
                )
                teacher_hidden = getattr(self.teacher_transformer, "last_hidden_states", None)
            if self.lambda_distill_out > 0:
                T = self.distill_temperature
                pred_m = pred[rand_span_mask]              # [N, mel_bins]
                teacher_m = teacher_pred[rand_span_mask]   # [N, mel_bins]
                s_log = F.log_softmax(pred_m / T, dim=-1)
                t_sft = F.softmax(teacher_m.detach() / T, dim=-1)
                distill_out_loss = F.kl_div(s_log, t_sft, reduction="batchmean") * (T**2)

            if self.lambda_distill_hidden > 0 and self.distill_hidden_layers and student_hidden and teacher_hidden:
                parts = []
                for layer_idx in self.distill_hidden_layers:
                    if -len(student_hidden) <= layer_idx < len(student_hidden) and -len(teacher_hidden) <= layer_idx < len(
                        teacher_hidden
                    ):
                        parts.append(F.mse_loss(student_hidden[layer_idx], teacher_hidden[layer_idx]))
                if parts:
                    distill_hidden_loss = torch.stack(parts).mean()

        if self.use_ctc and self.lambda_ctc > 0 and self.ctc_head is not None:
            ctc_source = None
            if student_hidden and -len(student_hidden) <= self.ctc_layer_index < len(student_hidden):
                ctc_source = student_hidden[self.ctc_layer_index]
            if ctc_source is not None and ctc_source.shape[-1] == self.dim:
                generated_mask = rand_span_mask if self.ctc_on_generated_only else None
                ctc_loss = self.ctc_head(ctc_source, text, lens, generated_mask=generated_mask)

        if self.use_accent_adv and self.lambda_adv > 0 and self.accent_head is not None:
            labels = accent_id
            if labels is None:
                labels = lang_id
            if labels is None:
                labels = domain_id

            if labels is not None:
                adv_source = None
                if student_hidden and -len(student_hidden) <= self.adv_feature_layer < len(student_hidden):
                    adv_source = student_hidden[self.adv_feature_layer]
                valid = labels >= 0
                if adv_source is not None and adv_source.shape[-1] == self.dim and valid.any():
                    logits = self.accent_head(adv_source, lambda_adv=1.0, mask=mask)
                    adv_loss = F.cross_entropy(logits, labels.long(), ignore_index=-1)

        total_loss = (
            flow_loss
            + self.lambda_distill_out * distill_out_loss
            + self.lambda_distill_hidden * distill_hidden_loss
            + self.lambda_ctc * ctc_loss
            + self.lambda_adv * adv_loss
        )

        if not torch.isfinite(total_loss):
            total_loss = flow_loss
            distill_out_loss = zero
            distill_hidden_loss = zero
            ctc_loss = zero
            adv_loss = zero

        self.last_loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_flow": flow_loss.detach(),
            "loss_distill_out": distill_out_loss.detach(),
            "loss_distill_hidden": distill_hidden_loss.detach(),
            "loss_ctc": ctc_loss.detach(),
            "loss_adv": adv_loss.detach(),
        }

        return total_loss, cond, pred
