from typing import Optional
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.augmentation import gaussian_kernel_2d_opencv


class DDN:
    """
    DDN attack: decoupling the direction and norm of the perturbation to achieve a small L2 norm in few steps.
    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    gamma : float, optional
        Factor by which the norm will be modified. new_norm = norm * (1 + or - gamma).
    init_norm : float, optional
        Initial value for the norm.
    quantize : bool, optional
        If True, the returned adversarials will have quantized values to the specified number of levels.
    levels : int, optional
        Number of levels to use for quantization (e.g. 256 for 8 bit images).
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.
    callback : object, optional
        Visdom callback to display various metrics.
    """

    def __init__(self,
                 steps: int = 200,
                 gamma: float = 0.05,
                 init_norm: float = 1.,
                 quantize: bool = True,
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu'),
                 callback: Optional = None) -> None:
        self.steps = steps
        self.gamma = gamma
        self.init_norm = init_norm

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm

        self.device = device
        self.callback = callback

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               targeted: bool = False, use_post_process: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.
        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)
        norm = torch.full((batch_size,), self.init_norm, device=self.device, dtype=torch.float)
        worst_norm = F.interpolate(torch.max(inputs, 1 - inputs), (299, 299)).view(batch_size, -1).norm(p=2, dim=1)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps, eta_min=0.01)

        best_l2 = worst_norm.clone()
        best_delta = torch.zeros_like(inputs)
        adv_found = torch.zeros(inputs.size(0), dtype=torch.uint8, device=self.device)

        for i in range(self.steps):
            scheduler.step()

            l2 = F.interpolate(delta, (299, 299)).data.view(batch_size, -1).norm(p=2, dim=1)
#             l2 = torch.clamp(torch.sqrt(torch.clamp((delta ** 2).sum(dim=1), min=1e-6)), min=10).mean(dim=-1).mean(dim=-1)
            adv = inputs + delta
            if use_post_process:
                kernel_size = 5
                kernel = torch.tensor(gaussian_kernel_2d_opencv(kernel_size)).cuda()
                adv_processed = F.conv2d(adv, kernel, padding=kernel_size//2)
                logits = [m(adv_processed) for m in model] if isinstance(model, list) else model(adv_processed)
            else:
                logits = [m(adv) for m in model] if isinstance(model, list) else model(adv)
                
            pred_labels = [l.argmax(dim=1) for l in logits] if isinstance(logits, list) else logits.argmax(dim=1)
            ce_loss = sum([F.cross_entropy(l, labels, reduction='sum') for l in logits]) / len(logits) if isinstance(logits, list) else F.cross_entropy(logits, labels, reduction='sum')
            loss = multiplier * ce_loss

            if isinstance(pred_labels, list):
                is_adv = [(p == labels) if targeted else (p != labels) for p in pred_labels]
                is_adv = reduce(lambda x, y: x & y, is_adv)
            else:
                is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            is_smaller = l2 < best_l2
            is_both = is_adv * is_smaller
            adv_found[is_both] = 1
            best_l2[is_both] = l2[is_both]
            best_delta[is_both] = delta.data[is_both]

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            if self.callback:
                cosine = F.cosine_similarity(-delta.grad.view(batch_size, -1),
                                             delta.data.view(batch_size, -1), dim=1).mean().item()
                for l in ce_loss:
                    self.callback.scalar('ce', i, l.item() / batch_size)
                self.callback.scalars(
                    ['max_norm', 'l2', 'best_l2'], i,
                    [norm.mean().item(), l2.mean().item(),
                     best_l2[adv_found].mean().item() if adv_found.any() else norm.mean().item()]
                )
                self.callback.scalars(['cosine', 'lr', 'success'], i,
                                      [cosine, optimizer.param_groups[0]['lr'], adv_found.float().mean().item()])

            optimizer.step()
#             print('success: {}'.format(adv_found.float().mean().item()))

            norm.mul_(1 - (2 * is_adv.float() - 1) * self.gamma)
            norm = torch.min(norm, worst_norm)

            delta.data.mul_((norm / delta.data.view(batch_size, -1).norm(2, 1)).view(-1, 1, 1, 1))
            delta.data.add_(inputs)
            if self.quantize:
                delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)
            delta.data.clamp_(0, 1).sub_(inputs)

        if self.max_norm:
            best_delta.renorm_(p=2, dim=0, maxnorm=self.max_norm)
            if self.quantize:
                best_delta.mul_(self.levels - 1).round_().div_(self.levels - 1)
#         print('success: {}'.format(adv_found.float().mean().item()))
        return inputs + best_delta
