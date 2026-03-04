from pathlib import Path

import torch
from generate import load_from_pretrained
from hnet.utils.data import create_dataloaders

torch.set_printoptions(sci_mode=False)

model_1L = load_from_pretrained(
    model_path="/scratch-shared/mivanov1/hnet/checkpoints/hnet_1stage_XL/hnet_1stage_XL.pt",
    model_config_path="./configs/hnet_1stage_XL.json"
)

model_2L = load_from_pretrained(
    model_path="/scratch-shared/mivanov1/hnet/checkpoints/hnet_2stage_XL/hnet_2stage_XL.pt",
    model_config_path="./configs/hnet_2stage_XL.json"
)

Q = model_1L.backbone.routing_module.q_proj_layer.weight.type(torch.float32)
K = model_1L.backbone.routing_module.k_proj_layer.weight.type(torch.float32)

Q_2_1 = model_2L.backbone.routing_module.q_proj_layer.weight.type(torch.float32)
K_2_1 = model_2L.backbone.routing_module.k_proj_layer.weight.type(torch.float32)

print(model_2L)

Q_2_2 = model_2L.backbone.main_network.routing_module.q_proj_layer.weight.type(torch.float32)
K_2_2 = model_2L.backbone.main_network.routing_module.k_proj_layer.weight.type(torch.float32)

def get_metrics_for_matrix(A):
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    dist_frobenius = torch.norm(A - I, p="fro").item()

    U, S, V = torch.linalg.svd(A)
    print(f"Singular values — min: {S.min():.4f}, max: {S.max():.4f}, mean: {S.mean():.4f}, std: {S.std():.4f}")

    # How many singular values deviate significantly from 1?
    threshold = 0.1
    deviated = (S - 1).abs() > threshold
    print(f"Singular values deviating >0.1 from 1: {deviated.sum().item()} / {S.shape[0]}")

    # Also check if it's still approximately orthogonal
    # (identity is orthogonal; did it stay that way or become non-orthogonal?)
    ATA = A.T @ A
    ortho_dist = torch.norm(ATA - I, p='fro').item()
    print(f"Distance from orthogonality (||A^T A - I||_F): {ortho_dist:.4f}")

    # Effective rank / how many dimensions carry the signal
    cumulative_energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
    eff_rank_90 = (cumulative_energy < 0.90).sum().item() + 1
    eff_rank_99 = (cumulative_energy < 0.99).sum().item() + 1
    eff_rank_999 = (cumulative_energy < 0.999).sum().item() + 1
    print(f"Dimensions for 90% energy: {eff_rank_90} / 1024")
    print(f"Dimensions for 99% energy: {eff_rank_99} / 1024")
    print(f"Dimensions for 99.9% energy: {eff_rank_999} / 1024")
    print(f"Top 10 singular values: {S[:10].tolist()}")

    # How many are essentially dead?
    print(f"Singular values < 0.01: {(S < 0.01).sum().item()} / 1024")

    print(f"Frobenius norm: {dist_frobenius:.6f}")

def get_metrics(Q, K):
    k = 20
    U_q, S_q, _ = torch.linalg.svd(Q)
    U_k, S_k, _ = torch.linalg.svd(K)

    # Principal angles between the top-k subspaces
    # (1.0 = aligned, 0.0 = orthogonal)
    cos_angles = torch.linalg.svdvals(U_q[:, :k].T @ U_k[:, :k])
    print(f"Cosine of principal angles (top-{k}): {cos_angles.tolist()}")

    k = 5
    # Pairwise cosine similarities between individual singular vectors
    alignment = (U_q[:, :k].T @ U_k[:, :k]).abs()  # shape [k, k]
    print("Pairwise alignment (|cos| between individual singular vectors):")
    print(alignment)

    print('*' * 55)
    print("Metrics for Q")
    get_metrics_for_matrix(Q)

    print('*' * 55)
    print("Metrics for K")
    get_metrics_for_matrix(K)

print('-' * 50)
print("Metrics for 1stage_XL")
get_metrics(Q, K)
# print('-' * 50)
# print("Metrics for 2stage_XL_1")
# get_metrics(Q_2_1, K_2_1)
# print('-' * 50)
# print("Metrics for 2stage_XL_2")
# get_metrics(Q_2_2, K_2_2)

# dataloader, val_dataloader = create_dataloaders(
#     Path("/scratch-shared/mivanov1/hnet/data"),
#     "sample-10BT",
#     seq_len=8192,
#     seed=42,
#     val_batches=50,
#     batch_size=1,
#     num_workers=8
# )

# # for batch_idx, batch in enumerate(val_dataloader):


# for step in range(500, 10500, 500):
#     print(f"Loading train weights for step: {step}")
#     model_1L = load_from_pretrained(
#         model_path=f"/scratch-shared/mivanov1/hnet/checkpoints/train/train_hnet_1stage_L_700M_baseline/model_step{step}.pt",
#         model_config_path="./configs/hnet_1stage_L.json"
#     )

#     Q = model_1L.backbone.routing_module.q_proj_layer.weight.type(torch.float32)
#     K = model_1L.backbone.routing_module.k_proj_layer.weight.type(torch.float32)

#     print('-' * 50)
#     print(f"Metrics for 1stage_L_step_{step}")
#     get_metrics(Q, K)