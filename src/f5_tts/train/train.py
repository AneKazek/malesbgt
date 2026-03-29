# training script.

import inspect
import os
from importlib.resources import files
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)


def _to_plain_dict(cfg_obj):
    if cfg_obj is None:
        return {}
    if isinstance(cfg_obj, dict):
        return dict(cfg_obj)
    return OmegaConf.to_container(cfg_obj, resolve=True)


def _filter_constructor_kwargs(cls, kwargs: dict):
    signature = inspect.signature(cls.__init__)
    allowed = set(signature.parameters) - {"self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def _load_checkpoint_blob(path: str):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(blob):
    if isinstance(blob, dict):
        for key in ("ema_model_state_dict", "model_state_dict", "state_dict"):
            value = blob.get(key)
            if isinstance(value, dict):
                return value
        if blob and all(torch.is_tensor(v) for v in blob.values()):
            return blob

    raise ValueError("Unsupported checkpoint format: no state_dict found")


def _normalize_teacher_key(key: str) -> str:
    prefixes = (
        "module.",
        "ema_model.",
        "model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True

    if key.startswith("transformer."):
        key = key[len("transformer.") :]

    return key


def _load_teacher_transformer(teacher_transformer, checkpoint_path: str):
    blob = _load_checkpoint_blob(checkpoint_path)
    raw_state_dict = _extract_state_dict(blob)

    normalized_state_dict = {}
    for key, value in raw_state_dict.items():
        if not torch.is_tensor(value):
            continue
        norm_key = _normalize_teacher_key(key)
        normalized_state_dict[norm_key] = value

    teacher_sd = teacher_transformer.state_dict()
    compatible = {}
    shape_mismatch = 0
    for key, value in normalized_state_dict.items():
        if key in teacher_sd:
            if teacher_sd[key].shape == value.shape:
                compatible[key] = value
            else:
                shape_mismatch += 1

    teacher_sd.update(compatible)
    teacher_transformer.load_state_dict(teacher_sd, strict=False)

    missing = len(teacher_sd) - len(compatible)
    unexpected = len(normalized_state_dict) - len([k for k in normalized_state_dict if k in teacher_sd])

    return {
        "loaded": len(compatible),
        "total_teacher": len(teacher_sd),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = _to_plain_dict(model_cfg.model.arch)
    cfm_experiment = _to_plain_dict(model_cfg.model.get("cfm_experiment", {}))
    tokenizer = model_cfg.model.tokenizer
    mel_spec_cfg = _to_plain_dict(model_cfg.model.mel_spec)
    mel_spec_type = mel_spec_cfg["mel_spec_type"]

    wandb_project = model_cfg.ckpts.get("wandb_project", "CFM-TTS")
    wandb_run_name = model_cfg.ckpts.get(
        "wandb_run_name",
        f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}",
    )
    wandb_resume_id = model_cfg.ckpts.get("wandb_resume_id", None)

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # set transformer model(s)
    student_transformer = model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=mel_spec_cfg["n_mel_channels"])

    use_distill = bool(cfm_experiment.get("use_distill", False))
    teacher_ckpt_path = cfm_experiment.pop("teacher_ckpt_path", None)
    teacher_backbone = cfm_experiment.pop("teacher_backbone", "DiT")

    if use_distill:
        if not teacher_ckpt_path:
            raise ValueError(
                "Distillation is enabled but model.cfm_experiment.teacher_ckpt_path is not set. "
                "Provide a teacher checkpoint path."
            )

        teacher_ckpt_path = str(Path(teacher_ckpt_path).expanduser())
        if not os.path.isabs(teacher_ckpt_path):
            teacher_ckpt_path = str((Path.cwd() / teacher_ckpt_path).resolve())
        if not os.path.isfile(teacher_ckpt_path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt_path}")

        teacher_cls = hydra.utils.get_class(f"f5_tts.model.{teacher_backbone}")
        teacher_arch_cfg = _to_plain_dict(model_cfg.model.get("teacher_arch", model_arc))
        teacher_arch = _filter_constructor_kwargs(teacher_cls, teacher_arch_cfg)
        teacher_transformer = teacher_cls(
            **teacher_arch,
            text_num_embeds=vocab_size,
            mel_dim=mel_spec_cfg["n_mel_channels"],
        )

        load_stats = _load_teacher_transformer(teacher_transformer, teacher_ckpt_path)
        print(
            "[Distill Teacher] "
            f"backbone={teacher_backbone} checkpoint={teacher_ckpt_path} "
            f"loaded={load_stats['loaded']}/{load_stats['total_teacher']} "
            f"missing={load_stats['missing']} unexpected={load_stats['unexpected']} "
            f"shape_mismatch={load_stats['shape_mismatch']}"
        )

        cfm_experiment["teacher_transformer"] = teacher_transformer
    elif teacher_ckpt_path:
        print("[Distill Teacher] teacher_ckpt_path provided but use_distill is false; checkpoint will be ignored.")

    # set model
    model = CFM(
        transformer=student_transformer,
        mel_spec_kwargs=mel_spec_cfg,
        vocab_char_map=vocab_char_map,
        **cfm_experiment,
    )

    # init trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    train_dataset = load_dataset(model_cfg.datasets.name, tokenizer, mel_spec_kwargs=mel_spec_cfg)
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
