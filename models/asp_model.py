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

    def _nce_loss_batch(self, fake_b, real_a, real_gt):
        fake_n, real_n = self._nce_tensors(fake_b, real_a)
        n_p = self.nce_patches
        feats_fake, patch_idx = self.netF.get_features(fake_n, n_patches=n_p)
        with torch.no_grad():
            feats_real, _ = self.netF.get_features(real_n, n_patches=n_p)
        if not feats_fake:
            return torch.tensor(0.0, device=self.device)
        fake_w, real_w = fake_b, real_gt
        if self.nce_size > 0:
            s = self.nce_size
            real_w = torch.nn.functional.interpolate(
                real_gt, size=(s, s), mode='bilinear', align_corners=False)
            fake_w = torch.nn.functional.interpolate(
                fake_b, size=(s, s), mode='bilinear', align_corners=False)
        l1_weights = get_patch_l1_weights(real_w, fake_w, patch_idx)
        losses = [
            self.criterionNCE(fq, fk, w)
            for fq, fk, w in zip(feats_fake, feats_real, l1_weights)
        ]
        return sum(losses) / len(losses)

    def _nce_loss(self):
        b = self.fake_B.size(0)
        if b <= 1:
            return self._nce_loss_batch(self.fake_B, self.real_A, self.real_B)
        loss = torch.tensor(0.0, device=self.device)
        for i in range(b):
            loss = loss + self._nce_loss_batch(
                self.fake_B[i:i + 1], self.real_A[i:i + 1], self.real_B[i:i + 1])
            if torch.cuda.is_available() and i + 1 < b:
                torch.cuda.empty_cache()
        return loss / b

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
