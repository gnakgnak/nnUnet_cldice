import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.utilities.helpers import softmax_helper_dim1


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    # img: b, c, x, y, z. Min-pooling along each axis separately, as in Shit et al. (CVPR 2021)
    p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def _skel_step(img: torch.Tensor, skel: torch.Tensor):
    img = soft_erode(img)
    delta = F.relu(img - soft_open(img))
    return img, skel + F.relu(delta - skel * delta)


def soft_skel(img: torch.Tensor, num_iter: int, use_checkpoint: bool = False) -> torch.Tensor:
    # use_checkpoint: recompute each iteration during backward instead of storing its intermediates
    # (max-pool keeps int64 indices, so stored intermediates cost several GB at nnU-Net patch sizes)
    skel = F.relu(img - soft_open(img))
    for _ in range(num_iter):
        if use_checkpoint:
            img, skel = checkpoint(_skel_step, img, skel, use_reentrant=False)
        else:
            img, skel = _skel_step(img, skel)
    return skel


class SoftclDiceLoss(nn.Module):
    def __init__(self, classes, num_iter: int = 5, smooth: float = 1.):
        """
        1 - clDice, computed with soft skeletons on softmax probabilities.
          Tprec = |S_P ∩ V_L| / |S_P|,  Tsens = |S_L ∩ V_P| / |S_L|,  clDice = 2 Tprec Tsens / (Tprec + Tsens)
        Sums run over the whole batch (like nnU-Net's batch_dice). Returns the mean over `classes`.
        """
        super().__init__()
        self.classes = list(classes)
        self.num_iter = num_iter
        self.smooth = smooth

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # net_output: b, c, x, y, z logits. target: b, 1, x, y, z label map
        with torch.autocast(net_output.device.type, enabled=False):
            v_p = softmax_helper_dim1(net_output.float())[:, self.classes]
            v_l = torch.cat([(target == c) for c in self.classes], 1).float()
            # skeletonize one (sample, class) volume at a time (checkpointed for the prediction), so only the
            # intermediates of a single iteration of a single volume are ever alive
            with torch.no_grad():
                s_l = torch.stack([torch.stack([
                    soft_skel(v_l[b:b + 1, c:c + 1], self.num_iter)[0, 0]
                    for c in range(v_l.shape[1])]) for b in range(v_l.shape[0])])
            s_p = torch.stack([torch.stack([
                soft_skel(v_p[b:b + 1, c:c + 1], self.num_iter, use_checkpoint=True)[0, 0]
                for c in range(v_p.shape[1])]) for b in range(v_p.shape[0])])
            axes = [0] + list(range(2, v_p.ndim))
            tprec = ((s_p * v_l).sum(axes) + self.smooth) / (s_p.sum(axes) + self.smooth)
            tsens = ((s_l * v_p).sum(axes) + self.smooth) / (s_l.sum(axes) + self.smooth)
            cl_dice = 2 * tprec * tsens / (tprec + tsens)
            return 1 - cl_dice.mean()


class DC_CE_and_clDice_loss(nn.Module):
    def __init__(self, dc_ce_loss: DC_and_CE_loss, cldice_loss: SoftclDiceLoss, weight_cldice: float):
        super().__init__()
        self.dc_ce = dc_ce_loss
        self.cldice = cldice_loss
        self.weight_cldice = weight_cldice

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dc_ce(net_output, target) + self.weight_cldice * self.cldice(net_output, target)
