"""
genpara_edit_example.py

Example:
1) Sample a chair extrinsic latent from SALAD (Phase 1).
2) Call a fine-tuned GenPara model (OpenAI) to modify specific parts
   using textual adjectives (e.g., "short", "open").
3) Reconstruct and visualize the modified chair using SPAGHETTI.

Requirements
-----------
- This repo cloned with SALAD + SPAGHETTI as submodules.
- Checkpoints placed under ./checkpoints/chairs/phase1, phase2
  (hparams.yaml + state_only.ckpt).
- OPENAI_API_KEY environment variable set.
"""

import os
import sys
from typing import Literal, List

import torch
from omegaconf import OmegaConf
import hydra
from pytorch_lightning import seed_everything

from openai import OpenAI

# ---------------------------------------------------------------------
# 0. Repo-relative imports (SALAD / SPAGHETTI)
# ---------------------------------------------------------------------

# 필요하면 여기서 sys.path 정리 (예: 다른 프로젝트 경로 제거)
# sys.path = [p for p in sys.path if "salad" not in p and "bomani" not in p]

# 이 스크립트 기준으로 SALAD, SPAGHETTI가 import 가능하다고 가정
from salad.models.phase1 import Phase1Model
from salad.models.phase2 import Phase2Model
from salad.utils import visutil, imageutil
from salad.utils.spaghetti_util import (
    load_spaghetti,
    load_mesher,
    generate_zc_from_sj_gaus,
    batch_gaus_to_gmms,
)
from salad.utils import gm_utils
from salad.utils.paths import SPAGHETTI_DIR

# ---------------------------------------------------------------------
# 1. Model loading (SALAD phase1/phase2)
# ---------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_ROOT = "./checkpoints/chairs"


def load_model(model_class: Literal["phase1", "phase2"], device: str):
    """
    Load SALAD phase1/phase2 model from ./checkpoints/chairs/<model_class>/.
    """
    cfg_path = os.path.join(CHECKPOINT_ROOT, model_class, "hparams.yaml")
    ckpt_path = os.path.join(CHECKPOINT_ROOT, model_class, "state_only.ckpt")

    c = OmegaConf.load(cfg_path)
    model = hydra.utils.instantiate(c)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    return model


def split_gmm(v, f, extrinsics):
    """
    Utility to split a mesh into Gaussian parts (for visualization).
    """
    v_tensor = torch.tensor(v)
    f_tensor = torch.tensor(f)
    extrinsics_tensor_ = batch_gaus_to_gmms(extrinsics)
    meshes = [v_tensor, f_tensor]
    splited_mesh = gm_utils.split_mesh_by_gmm(meshes, extrinsics_tensor_)
    splited_dictionary = {}
    for i in range(16):
        try:
            splited_mesh_dict = {"faces": splited_mesh[i].tolist()}
        except Exception:
            splited_mesh_dict = {"faces": splited_mesh[i]}
        splited_dictionary[f"part_{i}"] = splited_mesh_dict
    splited_dictionary["v"] = v.tolist()
    return splited_dictionary


# ---------------------------------------------------------------------
# 2. OpenAI fine-tuned model 호출 부분
# ---------------------------------------------------------------------

# 여러분이 실제 fine-tuning 후 받는 모델 이름으로 교체
FT_MODEL_NAME = "ft:gpt-4o-mini:YOUR_ORG:genpara-latent-edit"  # TODO: change


def edit_chair_latent(
    client: OpenAI,
    extrinsics: torch.Tensor,
    part_indices: List[int],
    adjective: str,
):
    """
    Call fine-tuned GenPara model to edit extrinsic latents.

    extrinsics: shape (1, 16, 16) or (16, 16)
    part_indices: e.g., [4, 11]
    adjective: e.g., "short", "wide and enclosed"
    """
    import json
    import time

    if extrinsics.ndim == 3:
        extrinsics = extrinsics[0]

    data = extrinsics.tolist()
    # 숫자를 적당히 반올림해서 프롬프트 크기 줄이기 (선택 사항)
    rounded_data = [
        [round(float(v), 4) for v in row] for row in data
    ]
    json_latents = json.dumps(rounded_data)

    indices_str = "[" + ", ".join(str(i) for i in part_indices) + "]"

    system_msg = (
        "You work with a 3D chair model represented by 16 parts, "
        "each encoded as a 16-dimensional latent vector. "
        "Each part vector consists of: 3 values for Mu (position), "
        "9 values for a 3x3 eigenvector matrix (shape directions), "
        "1 value Pi (importance), and 3 eigenvalues (scale). "
        "Given the full 16x16 latent array and a list of part indices, "
        "you must edit only the specified parts according to the user's "
        "adjectives (e.g., 'short back', 'thin legs'). "
        "Return the full modified 16x16 latent as a valid JSON array and "
        "do not add any natural-language explanation."
    )

    user_msg = (
        f"The extrinsic latent vectors of this chair are: {json_latents}\n"
        f"I want to see the version of this chair with {adjective}.\n"
        f"To achieve this, you must modify the parts with indices {indices_str} "
        "by adjusting Mu, eigenvectors, Pi, and eigenvalues as necessary, "
        "while keeping the other parts consistent. "
        "Return only the full modified 16x16 latent as JSON."
    )

    attempts = 0
    while attempts < 5:
        attempts += 1
        start_time = time.time()
        try:
            resp = client.chat.completions.create(
                model=FT_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            content = resp.choices[0].message.content
            modified = json.loads(content)

            if len(modified) != 16:
                raise ValueError(f"Expected 16 parts, got {len(modified)}")

            modified_tensor = torch.tensor(modified, dtype=torch.float32)
            elapsed = time.time() - start_time
            print(f"[OK] LLM editing done in {elapsed:.2f}s")
            return modified_tensor

        except Exception as e:
            print(f"[WARN] LLM call failed (attempt {attempts}/5): {e}")
            time.sleep(1.0)

    raise RuntimeError("Failed to obtain valid edited latents after 5 attempts.")


# ---------------------------------------------------------------------
# 3. End-to-end example
# ---------------------------------------------------------------------


def main():
    seed_everything(63)

    # 1) Load SALAD & SPAGHETTI
    phase1_model = load_model("phase1", DEVICE)
    phase2_model = load_model("phase2", DEVICE)

    if SPAGHETTI_DIR not in sys.path:
        sys.path.append(SPAGHETTI_DIR)

    spaghetti = load_spaghetti(device=DEVICE)
    mesher = load_mesher(device=DEVICE)

    # 2) Sample one chair extrinsic latent
    extrinsics = phase1_model.sampling_gaussians(1)  # shape ~ (1, 16, 16)
    extrinsics = torch.tensor(extrinsics, device=DEVICE, dtype=torch.float32)

    # 3) Visualize original chair
    intrinsics = phase2_model.sample(extrinsics)
    zcs = generate_zc_from_sj_gaus(spaghetti, intrinsics, extrinsics)

    v, f = None, None
    try:
        v, f = visutil.get_mesh_from_spaghetti(spaghetti, mesher, zcs[0], res=256)
    except Exception as e:
        print(f"[WARN] mesh reconstruction failed for original chair: {e}")

    # 4) Edit with fine-tuned LLM
    client = OpenAI()  # uses OPENAI_API_KEY

    highlight_indices = [4, 11]  # 예: 등받이 일부
    adjective = "short back"     # 예시

    edited_extrinsics = edit_chair_latent(
        client=client,
        extrinsics=extrinsics[0],
        part_indices=highlight_indices,
        adjective=adjective,
    ).to(DEVICE)

    edited_extrinsics = edited_extrinsics.unsqueeze(0)  # (1, 16, 16)

    # 5) Reconstruct edited chair
    edited_intrinsics = phase2_model.sample(edited_extrinsics)
    edited_zcs = generate_zc_from_sj_gaus(
        spaghetti, edited_intrinsics, edited_extrinsics
    )

    try:
        v_edit, f_edit = visutil.get_mesh_from_spaghetti(
            spaghetti, mesher, edited_zcs[0], res=256
        )
    except Exception as e:
        print(f"[WARN] mesh reconstruction failed for edited chair: {e}")
        v_edit, f_edit = None, None

    # 6) Render Gaussian + mesh (before/after)
    images = []

    # original
    gaus_img_orig = visutil.render_gaussians(extrinsics[0], highlight_indices=[highlight_indices])
    mesh_img_orig = visutil.render_mesh(v, f) if v is not None else gaus_img_orig
    images.append([gaus_img_orig, mesh_img_orig])

    # edited
    gaus_img_edit = visutil.render_gaussians(edited_extrinsics[0], highlight_indices=[highlight_indices])
    mesh_img_edit = visutil.render_mesh(v_edit, f_edit) if v_edit is not None else gaus_img_edit
    images.append([gaus_img_edit, mesh_img_edit])

    final_img = imageutil.merge_images(images)
    final_img.save("genpara_edit_result.png")
    print("[OK] Saved comparison image → genpara_edit_result.png")


if __name__ == "__main__":
    main()
