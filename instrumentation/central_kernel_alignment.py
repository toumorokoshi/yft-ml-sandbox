import torch

def central_kernel_alignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pass


def _center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    """
    Centering of a kernel is:

    K_c = C_n @ K @ C_n^T

    where

    C_n = I_n - (1 / n) * 1_n

    for a given n.
    """
    n = kernel.size(0)
    one_n = torch.ones((n, n), device=kernel.device) / n
    return kernel - one_n @ kernel - kernel @ one_n + one_n @ kernel @ one_n