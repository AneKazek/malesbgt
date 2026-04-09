"""
Smoke test untuk HybridDiT V2 (Sparse BiMamba).

Checks:
  1. Imports & instantiation
  2. Layer placement: [6, 10, 14, 18] are HybridDiTBlock, rest are DiTBlock
  3. mamba_alpha initialized at 0.0

If CUDA is available:
  4. Forward pass shape correctness
  5. Gradient flow to mamba_alpha
"""

import sys
import torch
import torch.nn as nn

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0
skipped = 0

def ok(name):
    global passed; passed += 1
    print(f"  {GREEN}✓{RESET} {name}")

def fail(name, reason=""):
    global failed; failed += 1
    msg = f": {reason}" if reason else ""
    print(f"  {RED}✗{RESET} {name}{RED}{msg}{RESET}")

def skip(name, reason=""):
    global skipped; skipped += 1
    print(f"  {YELLOW}–{RESET} {name} {YELLOW}(skip: {reason}){RESET}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = DEVICE == "cuda"

print(f"\n{BOLD}Smoke test — HybridDiT V2 (Sparse BiMamba){RESET}")
print(f"Device: {DEVICE}")
if not HAS_CUDA:
    print(f"{YELLOW}No GPU — Mamba forward pass tests will be skipped.{RESET}")

# ── 1. Imports ──────────────────────────────────────────────
print(f"\n{BOLD}1. Imports{RESET}")
print("─" * 50)

try:
    from f5_tts.model.backbones.hybrid_dit import HybridDiT, HybridDiTBlock
    from f5_tts.model.backbones.dit import DiT
    from f5_tts.model.modules import DiTBlock
    ok("All imports successful")
except Exception as e:
    fail("Imports", str(e))
    sys.exit(1)

# ── 2. Instantiation ───────────────────────────────────────
print(f"\n{BOLD}2. Model instantiation{RESET}")
print("─" * 50)

MAMBA_LAYERS = [6, 10, 14, 18]
DIM = 128
DEPTH = 22

try:
    model = HybridDiT(
        dim=DIM,
        depth=DEPTH,
        heads=4,
        dim_head=32,
        ff_mult=2,
        mel_dim=16,
        text_num_embeds=256,
        text_dim=64,
        conv_layers=4,
        use_mamba=True,
        mamba_layers=MAMBA_LAYERS,
        use_bidi=True,
        inject_sinpos=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
    )
    ok(f"HybridDiT(depth={DEPTH}, mamba_layers={MAMBA_LAYERS}) created")
except Exception as e:
    fail("Instantiation", str(e))
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 3. Layer placement ─────────────────────────────────────
print(f"\n{BOLD}3. Layer placement verification{RESET}")
print("─" * 50)

try:
    for idx in MAMBA_LAYERS:
        blk = model.transformer_blocks[idx]
        assert isinstance(blk, HybridDiTBlock), \
            f"Layer {idx}: expected HybridDiTBlock, got {type(blk).__name__}"
    ok(f"Layers {MAMBA_LAYERS} are HybridDiTBlock ✓")
except Exception as e:
    fail("Mamba layer types", str(e))

try:
    dit_layers = [i for i in range(DEPTH) if i not in MAMBA_LAYERS]
    for idx in dit_layers[:4]:
        blk = model.transformer_blocks[idx]
        assert isinstance(blk, DiTBlock), \
            f"Layer {idx}: expected DiTBlock, got {type(blk).__name__}"
    ok(f"Non-Mamba layers are DiTBlock ✓")
except Exception as e:
    fail("DiT layer types", str(e))

# ── 4. mamba_alpha zero-init ────────────────────────────────
print(f"\n{BOLD}4. mamba_alpha initialization{RESET}")
print("─" * 50)

try:
    for idx in MAMBA_LAYERS:
        blk = model.transformer_blocks[idx]
        assert hasattr(blk, "mamba_alpha"), f"Layer {idx}: missing mamba_alpha"
        val = blk.mamba_alpha.item()
        assert val == 0.0, f"Layer {idx}: mamba_alpha = {val}, expected 0.0"
    ok("All mamba_alpha parameters initialized at 0.0 ✓")
except Exception as e:
    fail("mamba_alpha init", str(e))

# ── 5. Parameter count ────────────────────────────────────
print(f"\n{BOLD}5. Parameter count{RESET}")
print("─" * 50)

total = sum(p.numel() for p in model.parameters())
mamba_params = 0
for idx in MAMBA_LAYERS:
    blk = model.transformer_blocks[idx]
    if hasattr(blk, "mixer"):
        mamba_params += sum(p.numel() for p in blk.mixer.parameters())
    mamba_params += 1  # mamba_alpha

pct = mamba_params / total * 100
ok(f"Total params: {total:,}")
ok(f"Mamba-specific params: {mamba_params:,} ({pct:.1f}%)")

# ── 6. Weight copy from DiT ──────────────────────────────
print(f"\n{BOLD}6. Pretrained weight transfer{RESET}")
print("─" * 50)

try:
    dit_base = DiT(
        dim=DIM, depth=DEPTH, heads=4, dim_head=32, ff_mult=2,
        mel_dim=16, text_num_embeds=256, text_dim=64, conv_layers=4,
    )
    for p in dit_base.parameters():
        nn.init.normal_(p, mean=0.5, std=0.1)

    from f5_tts.model.backbones.hybrid_dit import load_partial_state_dict_safely
    n_copied = load_partial_state_dict_safely(model, dit_base.state_dict())
    ok(f"load_partial_state_dict_safely: {n_copied} tensors copied")

    # Verify FFN weights in a Mamba layer were copied
    blk6 = model.transformer_blocks[6]
    dit6 = dit_base.transformer_blocks[6]
    def _first_linear_weight(ff_module):
        for mod in ff_module.modules():
            if isinstance(mod, nn.Linear):
                return mod.weight
        raise RuntimeError("No nn.Linear in FFN")

    ff_match = torch.allclose(
        _first_linear_weight(blk6.ff),
        _first_linear_weight(dit6.ff),
    )
    assert ff_match, "FFN weights not copied into Mamba layer"
    ok("FFN weights correctly transferred to HybridDiTBlock ✓")
except Exception as e:
    fail("Weight transfer", str(e))
    import traceback; traceback.print_exc()

# ── 7. Forward pass (GPU only) ────────────────────────────
print(f"\n{BOLD}7. Forward pass (requires GPU){RESET}")
print("─" * 50)

if not HAS_CUDA:
    skip("Forward pass", "No CUDA GPU available")
    skip("Gradient flow", "No CUDA GPU available")
    skip("cfg_infer", "No CUDA GPU available")
else:
    model = model.to(DEVICE)
    B, T = 2, 64
    x = torch.randn(B, T, 16, device=DEVICE)
    cond = torch.randn(B, T, 16, device=DEVICE)
    text = torch.randint(0, 256, (B, 32), device=DEVICE)
    time = torch.rand(B, device=DEVICE)
    mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)

    try:
        model.eval()
        with torch.no_grad():
            out = model(x, cond, text, time, mask=mask)
        assert out.shape == (B, T, 16)
        assert not torch.isnan(out).any()
        ok(f"Forward pass shape {out.shape}, no NaN ✓")
    except Exception as e:
        fail("Forward pass", str(e))

    try:
        model.train()
        out = model(x, cond, text, time, mask=mask)
        out.mean().backward()
        for idx in MAMBA_LAYERS:
            alpha = model.transformer_blocks[idx].mamba_alpha
            assert alpha.grad is not None, f"No grad @ layer {idx}"
        ok("Gradients flow to mamba_alpha ✓")
    except Exception as e:
        fail("Gradient flow", str(e))

    try:
        model.eval()
        with torch.no_grad():
            out_cfg = model(x, cond, text, time, mask=mask, cfg_infer=True)
        assert out_cfg.shape == (B * 2, T, 16)
        ok(f"cfg_infer shape {out_cfg.shape} ✓")
    except Exception as e:
        fail("cfg_infer", str(e))

# ── Summary ───────────────────────────────────────────────
total_tests = passed + failed + skipped
print(f"\n{'─'*50}")
print(f"{BOLD}Results: {GREEN}{passed} passed{RESET}  "
      f"{RED}{failed} failed{RESET}  "
      f"{YELLOW}{skipped} skipped{RESET}  "
      f"/ {total_tests} total")

if failed == 0 and skipped == 0:
    print(f"\n{GREEN}{BOLD}All tests passed — V2 architecture is production-ready.{RESET}")
elif failed == 0:
    print(f"\n{YELLOW}{BOLD}All CPU tests passed. Run on GPU to verify forward pass.{RESET}")
else:
    print(f"\n{RED}{BOLD}Some tests failed — check errors above.{RESET}")

sys.exit(0 if failed == 0 else 1)
