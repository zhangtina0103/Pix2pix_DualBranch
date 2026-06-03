"""ASP on paired HEMIT: CUT + adaptive patch NCE weights from target L1."""
import torch
from .cut_model import CUTModel
from .nce_losses import AdaptivePatchNCELoss, get_patch_l1_weights


class ASPModel(CUTModel):
    """Paired ASP: same pipeline as CUTModel with adaptive NCE weighting."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CUTModel.modify_commandline_options(parser, is_train)
        if is_train:
            parser.add_argument('--lambda_ASP', type=float, default=1.0, help='adaptive PatchNCE weight')
        return parser

    def __init__(self, opt):
        CUTModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'G_NCE', 'D_real', 'D_fake']
        if self.isTrain:
            self.criterionNCE = AdaptivePatchNCELoss()

    def _nce_loss(self):
        fake_n, real_n = self._nce_spatial_inputs()
        n_p = self.nce_patches
        feats_fake, patch_idx = self.netF.get_features(fake_n, n_patches=n_p)
        with torch.no_grad():
            feats_real, _ = self.netF.get_features(real_n, n_patches=n_p)
        if not feats_fake:
            return torch.tensor(0.0, device=self.device)
        if self.nce_size > 0:
            s = self.nce_size
            real_b = torch.nn.functional.interpolate(
                self.real_B, size=(s, s), mode='bilinear', align_corners=False)
            fake_b = torch.nn.functional.interpolate(
                self.fake_B, size=(s, s), mode='bilinear', align_corners=False)
        else:
            real_b, fake_b = self.real_B, self.fake_B
        l1_weights = get_patch_l1_weights(real_b, fake_b, patch_idx)
        losses = [
            self.criterionNCE(fq, fk, w)
            for fq, fk, w in zip(feats_fake, feats_real, l1_weights)
        ]
        return sum(losses) / len(losses)

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        nce = self._nce_loss()
        lam = getattr(self.opt, 'lambda_ASP', self.opt.lambda_NCE)
        self.loss_G_NCE = nce * lam
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_NCE
        self.loss_G.backward()
