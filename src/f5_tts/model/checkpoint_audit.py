from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import torch
from torch import nn


def _module_prefix(key: str) -> str:
    parts = key.split(".")
    if len(parts) >= 4 and parts[0] in {"transformer", "teacher_transformer"} and parts[1] == "transformer_blocks":
        return ".".join(parts[:4])
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return key


def _group_keys(keys: Iterable[str]) -> dict[str, int]:
    counter = Counter(_module_prefix(k) for k in keys)
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _get_mamba_layers(model: nn.Module) -> set[int]:
    tf = getattr(model, "transformer", None)
    if tf is None:
        return set()
    mamba_layers = getattr(tf, "mamba_layers", None)
    if mamba_layers is None:
        return set()
    return {int(x) for x in mamba_layers}


def _is_expected_missing(key: str, mamba_layers: set[int]) -> bool:
    if key.startswith("ctc_head.") or key.startswith("accent_head.") or key.startswith("teacher_transformer."):
        return True

    parts = key.split(".")
    if len(parts) >= 5 and parts[0] == "transformer" and parts[1] == "transformer_blocks":
        try:
            block_idx = int(parts[2])
        except ValueError:
            return False
        if block_idx in mamba_layers and parts[3] == "mixer":
            return True

    return False


def _is_expected_unexpected(key: str, mamba_layers: set[int]) -> bool:
    parts = key.split(".")
    if len(parts) >= 5 and parts[0] == "transformer" and parts[1] == "transformer_blocks":
        try:
            block_idx = int(parts[2])
        except ValueError:
            return False
        if block_idx in mamba_layers and parts[3] == "attn":
            return True
    return False


def _categorize_loaded_key(key: str, mamba_layers: set[int]) -> str:
    if key.startswith("transformer.text_embed.") or key.startswith("transformer.input_embed."):
        if ".text_blocks." in key:
            return "text refinement"
        return "embeddings"

    if key.startswith("transformer.transformer_blocks."):
        parts = key.split(".")
        if len(parts) >= 5:
            try:
                block_idx = int(parts[2])
            except ValueError:
                block_idx = -1
            submodule = parts[3]
            if block_idx in mamba_layers:
                if submodule == "mixer":
                    return "Mamba-only params"
                if submodule in {"attn_norm", "ff_norm", "ff"}:
                    return "AdaLN / FFN in replaced blocks"
            else:
                return "DiT non-mamba blocks"

    if key.startswith("ctc_head.") or key.startswith("accent_head."):
        return "aux heads"

    return "other shared params"


def summarize_checkpoint_load(
    model: nn.Module,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> dict:
    student_sd = model.state_dict()
    mamba_layers = _get_mamba_layers(model)

    loaded_exact = []
    missing = []
    unexpected = []

    for key, student_tensor in student_sd.items():
        ckpt_tensor = checkpoint_state_dict.get(key)
        if ckpt_tensor is not None and tuple(student_tensor.shape) == tuple(ckpt_tensor.shape):
            loaded_exact.append(key)
        else:
            missing.append(key)

    for key, ckpt_tensor in checkpoint_state_dict.items():
        student_tensor = student_sd.get(key)
        if student_tensor is None or tuple(student_tensor.shape) != tuple(ckpt_tensor.shape):
            unexpected.append(key)

    total_params_student = sum(int(v.numel()) for v in student_sd.values())
    total_params_loaded_exact = sum(int(student_sd[k].numel()) for k in loaded_exact)
    total_params_new_or_missing = total_params_student - total_params_loaded_exact
    load_ratio_percent = 100.0 * total_params_loaded_exact / max(total_params_student, 1)

    expected_missing = [k for k in missing if _is_expected_missing(k, mamba_layers)]
    suspicious_missing = [k for k in missing if k not in set(expected_missing)]
    expected_unexpected = [k for k in unexpected if _is_expected_unexpected(k, mamba_layers)]
    suspicious_unexpected = [k for k in unexpected if k not in set(expected_unexpected)]

    copied_module_summary = Counter(_categorize_loaded_key(k, mamba_layers) for k in loaded_exact)
    new_init_summary = Counter(_categorize_loaded_key(k, mamba_layers) for k in missing)

    return {
        "total_params_student": total_params_student,
        "total_params_loaded_exact": total_params_loaded_exact,
        "total_params_new_or_missing": total_params_new_or_missing,
        "load_ratio_percent": load_ratio_percent,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "expected_missing_keys": expected_missing,
        "suspicious_missing_keys": suspicious_missing,
        "expected_unexpected_keys": expected_unexpected,
        "suspicious_unexpected_keys": suspicious_unexpected,
        "missing_grouped": _group_keys(missing),
        "unexpected_grouped": _group_keys(unexpected),
        "copied_module_summary": dict(sorted(copied_module_summary.items())),
        "new_init_module_summary": dict(sorted(new_init_summary.items())),
    }


def format_checkpoint_audit(summary: dict, *, max_group_lines: int = 20) -> str:
    def _top_groups(grouped: dict[str, int]) -> str:
        if not grouped:
            return "none"
        lines = []
        for idx, (name, count) in enumerate(grouped.items()):
            if idx >= max_group_lines:
                lines.append(f"  ... (+{len(grouped) - max_group_lines} more groups)")
                break
            lines.append(f"  - {name}: {count}")
        return "\n".join(lines)

    copied = summary.get("copied_module_summary", {})
    new_init = summary.get("new_init_module_summary", {})

    return "\n".join(
        [
            "[Checkpoint Audit]",
            (
                "  total_params_student="
                f"{summary.get('total_params_student', 0):,} "
                "total_params_loaded_exact="
                f"{summary.get('total_params_loaded_exact', 0):,} "
                "total_params_new_or_missing="
                f"{summary.get('total_params_new_or_missing', 0):,} "
                "load_ratio_percent="
                f"{summary.get('load_ratio_percent', 0.0):.2f}%"
            ),
            (
                "  missing_keys="
                f"{len(summary.get('missing_keys', []))} "
                "(expected="
                f"{len(summary.get('expected_missing_keys', []))}, suspicious="
                f"{len(summary.get('suspicious_missing_keys', []))})"
            ),
            (
                "  unexpected_keys="
                f"{len(summary.get('unexpected_keys', []))} "
                "(expected="
                f"{len(summary.get('expected_unexpected_keys', []))}, suspicious="
                f"{len(summary.get('suspicious_unexpected_keys', []))})"
            ),
            "  missing_keys grouped by module:",
            _top_groups(summary.get("missing_grouped", {})),
            "  unexpected_keys grouped by module:",
            _top_groups(summary.get("unexpected_grouped", {})),
            "  copied module summary (num keys):",
            "\n".join(f"  - {k}: {v}" for k, v in copied.items()) if copied else "  - none",
            "  new init module summary (num keys):",
            "\n".join(f"  - {k}: {v}" for k, v in new_init.items()) if new_init else "  - none",
        ]
    )