import numpy as np

from nnunetv2.training.loss.cldice import DC_CE_and_clDice_loss, SoftclDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_clDice(nnUNetTrainer):
    """
    Dice + CE + weight_cldice * (1 - soft clDice).
    clDice is computed on cldice_classes only (Dataset001/002: 7 = Coronaries, 8 = Fat).
    """
    cldice_classes = (7, 8)
    weight_cldice = 0.5
    cldice_iter = 5

    def _build_loss(self):
        assert not self.label_manager.has_regions, 'clDice trainer only supports label (non-region) training'
        dc_ce = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                               ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        loss = DC_CE_and_clDice_loss(dc_ce, SoftclDiceLoss(self.cldice_classes, self.cldice_iter),
                                     self.weight_cldice)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 1e-6 if self.is_ddp else 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
