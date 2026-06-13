"""LENS model components: MLLM conditioner, grounding head, and SAM-based decoder."""

import torch


def load_full_state_dict(model, checkpoint_path, map_location="cpu"):
    """Resume a ``FullModel`` checkpoint, skipping tensors whose shapes don't match.

    This is identical to ``load_state_dict(..., strict=False)`` when every tensor matches
    (the original LLaVA + vit_h case). It additionally tolerates *shape* mismatches, which
    lets a checkpoint warm-start an experiment that swaps a differently sized component --
    e.g. a different SAM image encoder (``vit_h`` -> ``vit_l``/``vit_b``), whose frozen
    weights are rebuilt from the SAM checkpoint regardless. Skipped keys are logged.
    """
    state = torch.load(checkpoint_path, map_location=map_location)
    model_state = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"[resume] {checkpoint_path}: loaded {len(filtered)} tensors"
        + (f", skipped {len(skipped)} shape-mismatch/unknown (e.g. {skipped[:3]})" if skipped else "")
    )
    return missing, unexpected
