"""CUT on paired HEMIT: joint 3-channel output, same train/test/post_process as pix2pix."""
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from .nce_losses import (
    PatchNCELoss,
    EncoderFeatureExtractor,
    generator_module,
)


class CUTModel(BaseModel):
    """Paired CUT: conditional GAN + PatchNCE + L1 on full multiplex (3 ch)."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            dataset_mode='aligned',
            norm='instance',
            netG='resnet_9blocks',
            no_dropout=True,
        )
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='L1 weight on fake vs real B')
            parser.add_argument('--lambda_NCE', type=float, default=1.0, help='PatchNCE weight')
            parser.add_argument('--nce_patches', type=int, default=64,
                                help='PatchNCE samples per layer (lower = less VRAM)')
            parser.add_argument('--nce_size', type=int, default=0,
                                help='PatchNCE hook resolution; 0=same as train (1024²). '
                                     '512 only if OOM (set NCE_SIZE=512 on sbatch).')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.nce_size = int(getattr(opt, 'nce_size', 0) or 0)
        self.nce_patches = int(getattr(opt, 'nce_patches', 64))
        if self.isTrain:
            res = self.nce_size if self.nce_size > 0 else 'full'
            print(f'CUT PatchNCE: nce_size={res} nce_patches={self.nce_patches}')
        self.loss_names = ['G_GAN', 'G_L1', 'G_NCE', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D', 'F']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
        )
        if self.isTrain:
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
            )
            g_core = generator_module(self.netG)
            self.netF = EncoderFeatureExtractor(g_core).to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionNCE = PatchNCELoss()
            params_g = list(self.netG.parameters()) + list(self.netF.projectors.parameters())
            self.optimizer_G = torch.optim.Adam(params_g, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def _nce_spatial_inputs(self):
        """Downsample for PatchNCE hooks — full 1024² through netF OOMs on L40S."""
        if self.nce_size <= 0:
            return self.fake_B, self.real_A
        s = self.nce_size
        fake = F.interpolate(self.fake_B, size=(s, s), mode='bilinear', align_corners=False)
        real = F.interpolate(self.real_A, size=(s, s), mode='bilinear', align_corners=False)
        return fake, real

    def _nce_loss(self):
        fake_n, real_n = self._nce_spatial_inputs()
        n_p = self.nce_patches
        with torch.no_grad():
            feats_real, _ = self.netF.get_features(real_n, n_patches=n_p)
        feats_fake, _ = self.netF.get_features(fake_n, n_patches=n_p)
        if not feats_fake:
            return torch.tensor(0.0, device=self.device)
        losses = [self.criterionNCE(fq, fk) for fq, fk in zip(feats_fake, feats_real)]
        return sum(losses) / len(losses)

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_NCE = self._nce_loss() * self.opt.lambda_NCE
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_NCE
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
