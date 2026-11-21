from typing import Callable, Optional
import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode
from e3nn.nn import Gate, Activation

def mask_head(x: torch.Tensor, head: torch.Tensor, num_heads: int) -> torch.Tensor:
    mask = torch.zeros(x.shape[0], x.shape[1] // num_heads, num_heads, device=x.device)
    idx = torch.arange(mask.shape[0], device=x.device)
    mask[idx, :, head] = 1
    mask = mask.permute(0, 2, 1).reshape(x.shape)
    return x * mask

@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irrep_out: o3.Irreps = o3.Irreps("0e")):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irrep_out)

    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        # Robust handling of acts
        if callable(gate):
            acts = [gate] +[None] * int(len(self.hidden_irreps) -1)
        else:
            raise ValueError("gate must be a callable or a list/tuple of callables")
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=acts)
        self.linear_2 = o3.Linear(irreps_in=self.hidden_irreps, irreps_out=irrep_out)

    def forward(
        self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)  # [n_nodes, len(heads)]

@compile_mode("script")
class NonLinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,                    # 前一层输出
        mlp_irreps: o3.Irreps,                   # 线性 + Gate 之间的隐藏空间
        gate_fn: Callable,                       # 激活函数, 例: torch.nn.SiLU()
        irreps_out: o3.Irreps = o3.Irreps("1x0e + 1x1o"),
    ):
        super().__init__()

        # ---------------------------------------------------------------------
        # 1) 仅保留与最终输出同类型的通道 -> 减少“无用”通路
        #    - scalars : l == 0 & 出现在 irreps_out 中
        #    - gated   : l > 0 & 出现在 irreps_out 中
        # ---------------------------------------------------------------------
        irreps_scalars = o3.Irreps([
            (mul, ir) for mul, ir in mlp_irreps
            if ir.l == 0 and ir in irreps_out
        ])

        irreps_gated = o3.Irreps([
            (mul, ir) for mul, ir in mlp_irreps
            if ir.l > 0 and ir in irreps_out
        ])

        # 统一用 0e 做门控通道；如果想用 0o，把 "0e" 改成 "0o" 并选奇函数即可
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])

        # ---------------------------------------------------------------------
        # 2) Gate：为每个标量 / gate 分配激活
        #    - 偶标量 (0e) 可用 SiLU/ReLU 等
        #    - 奇标量 (0o) 必须用奇/偶明确的函数，如 tanh/abs
        # ---------------------------------------------------------------------
        def _act_scalar(ir):
            return gate_fn if ir.p == +1 else torch.tanh     # 0o → tanh

        act_scalars = [_act_scalar(ir) for _, ir in irreps_scalars]
        act_gates   = [torch.sigmoid] * len(irreps_gates)    # 0e gate → sigmoid

        self.gate = Gate(
            irreps_scalars = irreps_scalars,
            act_scalars    = act_scalars,
            irreps_gates   = irreps_gates,
            act_gates      = act_gates,
            irreps_gated   = irreps_gated,
        )

        # ---------------------------------------------------------------------
        # 3) 各层 linear
        # ---------------------------------------------------------------------
        # linear_1 把 input -> Gate 输入
        self.linear_1 = o3.Linear(
            irreps_in  = irreps_in,
            irreps_out = self.gate.irreps_in,    # == irreps_scalars + irreps_gates + irreps_gated
        )

        # linear_2 把 Gate 输出 -> 最终 dipole 表示
        self.linear_2 = o3.Linear(
            irreps_in  = self.gate.irreps_out,   # 关键修正！
            irreps_out = irreps_out,
        )

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., irreps_in]"""
        x = self.linear_1(x)      # -> Gate 输入
        x = self.gate(x)          # -> Gate 输出
        x = self.linear_2(x)      # -> 1x0e + 1x1o
        return x
