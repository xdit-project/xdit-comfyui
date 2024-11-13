from typing import Optional
import torch
from einops import rearrange
from torch import Tensor
from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management

from xdit_comfyui_private.envs import HAS_FLASH_ATTN


hybrid_seq_parallel_attn = None
hybrid_seq_parallel_attn_joint = None

def init_seq_parallel_attn():
    global hybrid_seq_parallel_attn
    global hybrid_seq_parallel_attn_joint
    # if HAS_FLASH_ATTN:
    #     from xdit_comfyui_private.modules.long_ctx_attention.hybrid import xFuserFluxLongContextAttention, xFuserLongContextAttention
    #     hybrid_seq_parallel_attn_joint = xFuserFluxLongContextAttention()
    #     hybrid_seq_parallel_attn = xFuserLongContextAttention()
    # else:
    #     from xdit_comfyui_private.modules.long_ctx_attention.ulysses import xFuserUlyssesAttention
    #     hybrid_seq_parallel_attn_joint = xFuserUlyssesAttention(use_fa=False)
    #     hybrid_seq_parallel_attn = hybrid_seq_parallel_attn_joint
    from xdit_comfyui_private.modules.long_ctx_attention.ulysses import xFuserUlyssesAttention
    from yunchang.kernels import FlashAttentionImpl
    hybrid_seq_parallel_attn_joint = xFuserUlyssesAttention(use_sync=True, attn_type=FlashAttentionImpl.FA3)
    hybrid_seq_parallel_attn = hybrid_seq_parallel_attn_joint



def attention(img_q: Tensor, img_k: Tensor, img_v: Tensor, pe: Tensor, txt_q: Optional[Tensor] = None, txt_k: Optional[Tensor] = None, txt_v: Optional[Tensor] = None, joint_strategy = 'front') -> Tensor:
    bs, heads, _, head_dim = img_q.shape
    if joint_strategy == 'none':
        img_q, img_k = apply_rope(img_q, img_k, pe)
        img_q = img_q.transpose(1,2)
        img_k = img_k.transpose(1,2)
        img_v = img_v.transpose(1,2)
        x = hybrid_seq_parallel_attn(img_q, img_k, img_v, joint_strategy=joint_strategy)
    else:
        txt_seq_len = txt_q.shape[2] if txt_q is not None else 0
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        q, k = apply_rope(q, k, pe)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        img_v = img_v.transpose(1,2)
        txt_v = txt_v.transpose(1,2)
        txt_q = q[:, :txt_seq_len, ...]
        img_q = q[:, txt_seq_len:, ...]
        txt_k = k[:, :txt_seq_len, ...]
        img_k = k[:, txt_seq_len:, ...]
        x = hybrid_seq_parallel_attn_joint(img_q, img_k, img_v, joint_tensor_query = txt_q, joint_tensor_key=txt_k, joint_tensor_value=txt_v, joint_strategy=joint_strategy)

    x = x.reshape(bs, -1, heads * head_dim)

    return x

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    if comfy.model_management.is_device_mps(pos.device) or comfy.model_management.is_intel_xpu():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
