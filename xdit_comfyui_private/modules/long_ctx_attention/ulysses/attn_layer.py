from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from yunchang import UlyssesAttention
from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.kernels import FlashAttentionImpl, select_flash_attn_impl


class xFuserUlyssesAttention(UlyssesAttention):
    def __init__(
        self,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
        attn_type: FlashAttentionImpl = FlashAttentionImpl.FA,
        use_kv_cache: bool = False,
    ) -> None:

        super(UlyssesAttention, self).__init__()
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.attn_type = attn_type
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(device)
        if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
            self.attn_type = FlashAttentionImpl.TORCH
        self.use_kv_cache = use_kv_cache
        self.fn = select_flash_attn_impl(self.attn_type, stage="fwd-bwd")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        if (
            joint_tensor_key is not None
            and joint_tensor_value is not None
            and joint_tensor_query is not None
        ):
            if joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
            elif joint_strategy == "front":
                query = torch.cat([joint_tensor_query, query], dim=1)
            elif joint_strategy == "none":
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supported when joint tensors is not None."
                )
            else:
                raise ValueError(f"joint_strategy: {joint_strategy} not supported.")
            ulysses_world_size = torch.distributed.get_world_size(self.spg)
            ulysses_rank = torch.distributed.get_rank(self.spg)
            attn_heads_per_ulysses_rank = (
                joint_tensor_key.shape[-2] // ulysses_world_size
            )
            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]

        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)

        # scatter 2, gather 1
        q = SeqAllToAll4D.apply(
            self.spg, query, self.scatter_idx, self.gather_idx, self.use_sync,
        )
        k = SeqAllToAll4D.apply(
            self.spg, key, self.scatter_idx, self.gather_idx, self.use_sync,
        )
        v = SeqAllToAll4D.apply(
            self.spg, value, self.scatter_idx, self.gather_idx, self.use_sync,
        )

        if joint_strategy != "none":
            if joint_strategy == "rear":
                k = torch.cat([k, joint_tensor_key], dim=1)
                v = torch.cat([v, joint_tensor_value], dim=1)

            elif joint_strategy == "front":
                k = torch.cat([joint_tensor_key, k], dim=1)
                v = torch.cat([joint_tensor_value, v], dim=1)

        context_layer = self.fn(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_attn_probs,
        )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.spg, context_layer, self.gather_idx, self.scatter_idx, self.use_sync,
        )

        # out e.g., [s/p::h]
        return output
