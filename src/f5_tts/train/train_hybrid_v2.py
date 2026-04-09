import os
from pathlib import Path
import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf, DictConfig
from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer

def print_banner(text):
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60 + "\n")

def setup_teacher(model_cfg, vocab_size, mel_dim):
    """Initialize and load the pretrained DiT teacher."""
    teacher_cfg = model_cfg.model.get("teacher_arch", model_cfg.model.arch)
    teacher_backbone = model_cfg.model.get("teacher_backbone", "DiT")
    teacher_cls = hydra.utils.get_class(f"f5_tts.model.backbones.{teacher_backbone.lower()}.{teacher_backbone}")
    
    teacher = teacher_cls(
        **teacher_cfg,
        text_num_embeds=vocab_size,
        mel_dim=mel_dim
    )
    
    ckpt_path = model_cfg.model.cfm_experiment.get("teacher_ckpt_path")
    if not ckpt_path:
        raise ValueError("teacher_ckpt_path must be set in config for distillation.")
    
    print(f"Loading teacher from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Handle both raw state_dict and trainer-wrapped checkpoints
    if "ema_model_state_dict" in checkpoint:
        state_dict = checkpoint["ema_model_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    # Remove 'transformer.' or 'model.' prefixes if present
    clean_sd = {}
    for k, v in state_dict.items():
        new_k = k.replace("transformer.", "").replace("model.", "").replace("module.", "")
        clean_sd[new_k] = v
        
    info = teacher.load_state_dict(clean_sd, strict=False)
    print(f"Teacher load info: {info}")
    teacher.eval()
    return teacher

@hydra.main(version_base="1.3", config_path="../configs", config_name="F5TTS_SparseBiMamba_V2")
def train(cfg: DictConfig):
    print_banner("HYBRID SPARSE BIMAMBA-DIT V2 TRAINING")
    
    # 1. Setup Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(cfg.datasets.name, cfg.model.tokenizer)
    mel_dim = cfg.model.mel_spec.n_mel_channels
    
    # 2. Build Student (HybridDiT)
    model_cls = hydra.utils.get_class(f"f5_tts.model.backbones.hybrid_dit.HybridDiT")
    student_transformer = model_cls(
        **cfg.model.arch,
        text_num_embeds=vocab_size,
        mel_dim=mel_dim
    )
    
    # 3. Setup Teacher & Distillation
    teacher = setup_teacher(cfg, vocab_size, mel_dim)
    
    # Warm-start from teacher: copy all compatible weights (AdaLN, FFN, etc.)
    # Mamba layers in student will keep their random/zero-init mixers.
    from f5_tts.model.backbones.hybrid_dit import load_partial_state_dict_safely
    num_copied = load_partial_state_dict_safely(student_transformer, teacher.state_dict())
    print(f"Warm-started student with {num_copied} tensors from teacher.")
    
    # 4. Wrap with CFM
    cfm_experiment = OmegaConf.to_container(cfg.model.cfm_experiment, resolve=True)
    cfm_experiment["teacher_transformer"] = teacher
    
    model = CFM(
        transformer=student_transformer,
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
        **cfm_experiment
    )
    
    # 5. Dataset & Trainer
    train_dataset = load_dataset(cfg.datasets.name, cfg.model.tokenizer, mel_spec_kwargs=cfg.model.mel_spec)
    
    save_dir = Path(cfg.ckpts.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(
        model,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        checkpoint_path=str(save_dir),
        batch_size_per_gpu=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_project=cfg.ckpts.wandb_project,
        wandb_run_name=cfg.ckpts.wandb_run_name,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        mel_spec_type=cfg.model.mel_spec.mel_spec_type,
        model_cfg_dict=OmegaConf.to_container(cfg, resolve=True),
    )
    
    print_banner("STARTING TRAINING")
    trainer.train(
        train_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666
    )

if __name__ == "__main__":
    train()
