"""
Vanilla conditional flow matching in the pix2pix framework.

Integrates mentor reference scripts:
  - flow_matching.py:     MONAI UNet, x1 L1 + perceptual, logit-normal t, tanh, Heun ODE via x1->v
  - flow_matching_v.py:   MONAI UNet, velocity MSE, uniform t, Heun ODE on raw v

Default backbone: MONAI DiffusionModelUNet (~11M params with 64,128,192 + res=2 + attn 0,0,1).
Legacy: --fm_backbone custom (skip U-Net, spatial t channel).
"""
import os
import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import networks
from .base_model import BaseModel
from .fm_perceptual import build_fm_perceptual
from .mentor_flow_net import MentorFlowNet


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(x + h)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_res: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x):
        x = self.conv(x)
        return self.res(x)


class _UpSkipBilinear(nn.Module):
    """Bilinear upsample + conv (decoder_only ablation; keys up.N.weight in state_dict)."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, n_res: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.res(x)


class _UpSkipConvTranspose(nn.Module):
    """ConvTranspose2d upsample (joint_perc / perc_strong checkpoints; keys up.weight)."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, n_res: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.res(x)


def _init_he_proj_conv(conv: nn.Conv2d, mode: str = "gray") -> None:
    """Map H&E RGB to multiplex-shaped x0. gray=mean(R,G,B) per marker (avoids stain color leak)."""
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        if mode == "identity":
            for c in range(min(conv.in_channels, conv.out_channels)):
                conv.weight[c, c, 0, 0] = 1.0
        elif mode == "gray":
            n_in = conv.in_channels
            for oc in range(conv.out_channels):
                for ic in range(n_in):
                    conv.weight[oc, ic, 0, 0] = 1.0 / n_in
        else:
            raise ValueError(f"unknown fm_he_proj_init={mode!r}")


def _make_up_skip(up_mode: str, in_ch: int, out_ch: int, skip_ch: int, n_res: int) -> nn.Module:
    if up_mode == "conv_transpose":
        return _UpSkipConvTranspose(in_ch, out_ch, skip_ch, n_res)
    if up_mode == "bilinear":
        return _UpSkipBilinear(in_ch, out_ch, skip_ch, n_res)
    raise ValueError(f"unknown fm_up_mode={up_mode!r}; use bilinear or conv_transpose")


class FiLMHeadGenerator(nn.Module):
    """Pooled H&E -> per output channel (gamma, beta); identity init (DAPI/CD3/panCK)."""

    def __init__(self, cond_nc: int, out_ch: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cond_nc, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_dim, 2 * out_ch)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.head.bias.data[:out_ch] = 1.0

    def forward(self, cond: torch.Tensor):
        h = self.trunk(self.pool(cond).flatten(1))
        gb = self.head(h)
        c = gb.shape[1] // 2
        gamma = gb[:, :c, None, None]
        beta = gb[:, c:, None, None]
        return gamma, beta


class FiLMGenerator(nn.Module):
    """Global H&E embedding -> (gamma, beta) per decoder level (identity init: gamma=1, beta=0)."""

    def __init__(self, cond_nc: int, decoder_dims: Tuple[int, ...], hidden_dim: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cond_nc, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2 * dim) for dim in decoder_dims])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            head.bias.data[: head.out_features // 2] = 1.0

    def forward(self, cond: torch.Tensor):
        h = self.trunk(self.pool(cond).flatten(1))
        out = []
        for head in self.heads:
            gb = head(h)
            dim = gb.shape[1] // 2
            gamma = gb[:, :dim, None, None]
            beta = gb[:, dim:, None, None]
            out.append((gamma, beta))
        return out


class _CondUNet(nn.Module):
    """Legacy custom U-Net (spatial t channel). Use --fm_backbone monai for mentor parity."""

    def __init__(self, in_ch: int = 7, out_ch: int = 3,
                 channels: Tuple[int, int, int] = (32, 64, 96),
                 num_res_blocks: int = 1,
                 up_mode: str = "bilinear",
                 use_film: bool = False,
                 film_where: str = "decoder",
                 use_learned_null: bool = False,
                 cond_nc: int = 3,
                 film_hidden: int = 128,
                 use_he_proj: bool = False,
                 he_proj_init: str = "gray"):
        super().__init__()
        self.he_proj = None
        if use_he_proj:
            self.he_proj = nn.Conv2d(3, 3, 1)
            _init_he_proj_conv(self.he_proj, he_proj_init)
        c1, c2, c3 = channels
        self.use_film = use_film
        self.film_where = film_where if use_film else ""
        self.film = None
        if use_film:
            if film_where == "head":
                self.film = FiLMHeadGenerator(cond_nc, out_ch, film_hidden)
            else:
                self.film = FiLMGenerator(cond_nc, (c2, c1), film_hidden)
        self.cfg_null = (
            nn.Parameter(torch.zeros(1, cond_nc, 1, 1)) if use_learned_null else None
        )
        self.stem = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.res1 = nn.Sequential(*[_ResBlock(c1) for _ in range(num_res_blocks)])
        self.down1 = _Down(c1, c2, num_res_blocks)
        self.down2 = _Down(c2, c3, num_res_blocks)
        self.mid = nn.Sequential(*[_ResBlock(c3) for _ in range(max(1, num_res_blocks))])
        self.up2 = _make_up_skip(up_mode, c3, c2, c2, num_res_blocks)
        self.up1 = _make_up_skip(up_mode, c2, c1, c1, num_res_blocks)
        self.head = nn.Conv2d(c1, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, cond_img: Optional[torch.Tensor] = None) -> torch.Tensor:
        film_params = None
        if (
            self.film is not None
            and cond_img is not None
            and self.film_where == "decoder"
        ):
            film_params = self.film(cond_img)
        h1 = self.res1(self.stem(x))
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h = self.mid(h3)
        h = self.up2(h, h2)
        if film_params is not None:
            g, b = film_params[0]
            h = h * g + b
        h = self.up1(h, h1)
        if film_params is not None:
            g, b = film_params[1]
            h = h * g + b
        out = self.head(h)
        if self.film_where == "head" and self.film is not None and cond_img is not None:
            g, b = self.film(cond_img)
            out = out * g + b
        return out


class VanillaFMModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode="aligned", no_dropout=True)
        parser.add_argument("--fm_backbone", type=str, default="monai",
                            choices=["monai", "custom"],
                            help="monai=mentor DiffusionModelUNet; custom=legacy skip U-Net")
        parser.add_argument("--fm_loss", type=str, default="x1",
                            choices=["x1", "velocity"],
                            help="x1=flow_matching.py; velocity=flow_matching_v.py")
        parser.add_argument("--fm_channels", type=str, default="64,128,192",
                            help="UNet channels (monai or custom)")
        parser.add_argument("--fm_attn_levels", type=str, default="0,0,0",
                            help="MONAI attention per level; use 0,0,0 at 1024² (OOM if 0,0,1)")
        parser.add_argument("--fm_num_head_channels", type=int, default=32,
                            help="MONAI attention head channels (monai only)")
        parser.add_argument("--fm_num_res_blocks", type=int, default=2,
                            help="ResBlocks per UNet level")
        parser.add_argument("--fm_up_mode", type=str, default="bilinear",
                            choices=["bilinear", "conv_transpose"],
                            help="custom U-Net decoder: bilinear (decoder_only) or "
                                 "conv_transpose (joint_perc / perc_strong checkpoints)")
        parser.add_argument("--fm_use_tanh", action="store_true",
                            help="tanh on net output (flow_matching.py; x1 + monai)")
        parser.add_argument("--fm_time_dist", type=str, default="logit_normal",
                            choices=["uniform", "logit_normal"],
                            help="t sampling: logit_normal (x1 mentor) or uniform (velocity)")
        parser.add_argument("--fm_P_mean", type=float, default=-0.8,
                            help="Logit-normal t: mean (flow_matching.py)")
        parser.add_argument("--fm_P_std", type=float, default=0.8,
                            help="Logit-normal t: std")
        parser.add_argument("--fm_steps", type=int, default=25,
                            help="ODE steps at test")
        parser.add_argument("--fm_val_steps", type=int, default=8,
                            help="ODE steps during training validation")
        parser.add_argument("--fm_sample_method", type=str, default="heun",
                            choices=["euler", "heun"],
                            help="ODE solver at val / test")
        parser.add_argument("--fm_use_cfg", action="store_true",
                            help="Classifier-free guidance: cond dropout at train, guided ODE at test")
        parser.add_argument("--fm_cfg_dropout", type=float, default=0.1,
                            help="Train: prob of zeroing H&E cond (CFG)")
        parser.add_argument("--fm_cfg_scale", type=float, default=1.5,
                            help="Test: guidance scale w (1.0 = no guidance)")
        parser.add_argument("--fm_use_film", action="store_true",
                            help="FiLM on custom U-Net decoder from pooled H&E (additive to concat cond)")
        parser.add_argument("--fm_film_where", type=str, default="decoder",
                            choices=["decoder", "head"],
                            help="decoder=legacy global FiLM on up2/up1; head=per-marker on output")
        parser.add_argument("--fm_film_hidden", type=int, default=128,
                            help="FiLM MLP hidden dim (custom backbone only)")
        parser.add_argument("--fm_film_reg", type=float, default=0.0,
                            help="L2 on (gamma-1)^2 and beta^2 (FiLM finetune stability)")
        parser.add_argument("--fm_null_mode", type=str, default="zero",
                            choices=["zero", "learned"],
                            help="CFG null cond: zeros or learnable 1x3 (CFG v2)")
        parser.add_argument("--fm_use_seg", action="store_true",
                            help="Concat 1ch pseudo seg with H&E cond (custom U-Net; dataset aligned_cond)")
        parser.add_argument("--fm_flow_path", type=str, default="noise",
                            choices=["noise", "bridge"],
                            help="noise=Gaussian x0; bridge=x0=proj(H&E) straight-line path")
        parser.add_argument("--fm_init_from_cond", action="store_true",
                            help="ODE start: sigma*noise + (1-sigma)*proj(H&E) (noise path only)")
        parser.add_argument("--fm_init_noise_sigma", type=float, default=0.3,
                            help="Noise fraction when fm_init_from_cond (0=pure proj)")
        parser.add_argument("--fm_he_proj_init", type=str, default="",
                            choices=["", "gray", "identity"],
                            help="he_proj init: gray (bridge default) or identity RGB copy")
        parser.add_argument("--fm_bridge_x0_sigma", type=float, default=0.0,
                            help="Bridge train: add sigma*noise to proj(H&E) x0")
        parser.add_argument("--fm_bridge_noise_prob", type=float, default=0.0,
                            help="Bridge train: prob of Gaussian x0 (joint_perc-style mix)")
        if is_train:
            parser.add_argument("--fm_lambda_perc", type=float, default=0.1,
                                help="Perceptual weight (x1 mode; flow_matching.py; 0=off)")
            parser.add_argument("--fm_perc_size", type=int, default=256,
                                help="Downsample for perceptual at 1024² (0=full res; 256 saves VRAM)")
            parser.add_argument("--fm_lambda_vel", type=float, default=0.0,
                                help="Aux velocity MSE in x1 mode (0=off)")
            parser.add_argument("--fm_lambda_l1", type=float, default=0.0,
                                help="Extra L1 on x1_hat (usually 0)")
            parser.add_argument("--fm_lambda_sample_l1", type=float, default=0.0,
                                help="Optional ODE sample L1 in train loop")
            parser.add_argument("--fm_train_sample_steps", type=int, default=0,
                                help="ODE steps when fm_lambda_sample_l1>0 (0=fm_steps)")
            parser.add_argument("--fm_sample_l1_prob", type=float, default=1.0,
                                help="Fraction of iters with ODE sample L1")
            parser.add_argument("--fm_train_sample_method", type=str, default="euler",
                                choices=["euler", "heun"],
                                help="ODE solver for train sample L1")
            parser.add_argument("--fm_use_gan", action="store_true",
                                help="PatchGAN on ODE samples (pix2pix-style sharpness)")
            parser.add_argument("--fm_lambda_gan", type=float, default=1.0,
                                help="Weight for GAN loss on ODE fake_B")
            parser.add_argument("--fm_gan_sample_prob", type=float, default=0.5,
                                help="Fraction of iters that run ODE for GAN / sample L1")
            parser.add_argument("--fm_gan_sample_steps", type=int, default=12,
                                help="ODE steps for train GAN/sample (0=fm_train_sample_steps or fm_steps)")
            parser.add_argument("--fm_channel_weights", type=str, default="1,2,1",
                                help="Per-channel weights on ODE L1 (DAPI,CD3,panCK) like pix2pix Focal")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.fm_loss_mode = str(getattr(opt, "fm_loss", "x1"))
        self.use_monai = str(getattr(opt, "fm_backbone", "monai")) == "monai"
        self.fm_use_cfg = bool(getattr(opt, "fm_use_cfg", False))
        self.fm_cfg_dropout = float(getattr(opt, "fm_cfg_dropout", 0.1))
        self.fm_cfg_scale = float(getattr(opt, "fm_cfg_scale", 1.5))
        self.fm_null_mode = str(getattr(opt, "fm_null_mode", "zero"))
        self.fm_film_reg = float(getattr(opt, "fm_film_reg", 0.0))
        self.fm_use_seg = bool(getattr(opt, "fm_use_seg", False))
        self.fm_flow_path = str(getattr(opt, "fm_flow_path", "noise"))
        self.fm_init_from_cond = bool(getattr(opt, "fm_init_from_cond", False))
        self.fm_init_noise_sigma = float(getattr(opt, "fm_init_noise_sigma", 0.3))
        he_init = str(getattr(opt, "fm_he_proj_init", "") or "")
        if not he_init:
            he_init = "gray" if self.fm_flow_path == "bridge" else "identity"
        self.fm_he_proj_init = he_init
        self.fm_bridge_x0_sigma = float(getattr(opt, "fm_bridge_x0_sigma", 0.0))
        self.fm_bridge_noise_prob = float(getattr(opt, "fm_bridge_noise_prob", 0.0))
        self.fm_cond_extra = 1 if self.fm_use_seg else 0
        use_he_proj = (
            self.fm_flow_path == "bridge" or self.fm_init_from_cond
        )
        if self.fm_use_seg or use_he_proj:
            if self.use_monai:
                raise NotImplementedError(
                    "fm_use_seg / fm_flow_path bridge / fm_init_from_cond "
                    "require --fm_backbone custom"
                )

        self.loss_names = ["FM"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.model_names = ["G"]

        ch = tuple(int(x) for x in opt.fm_channels.split(","))
        n_res = int(opt.fm_num_res_blocks)
        use_tanh = bool(getattr(opt, "fm_use_tanh", False)) and self.fm_loss_mode == "x1"

        if self.use_monai:
            attn = tuple(bool(int(a)) for a in opt.fm_attn_levels.split(","))
            self.netG: Union[MentorFlowNet, _CondUNet] = MentorFlowNet(
                in_ch=opt.input_nc,
                out_ch=opt.output_nc,
                channels=ch,
                attention_levels=attn,
                num_res_blocks=n_res,
                num_head_channels=int(opt.fm_num_head_channels),
                use_tanh=use_tanh,
            )
        else:
            up_mode = str(getattr(opt, "fm_up_mode", "bilinear"))
            use_film = bool(getattr(opt, "fm_use_film", False))
            film_where = str(getattr(opt, "fm_film_where", "decoder"))
            use_learned_null = self.fm_use_cfg and self.fm_null_mode == "learned"
            cond_nc = opt.input_nc + self.fm_cond_extra
            stem_in = opt.output_nc + cond_nc + 1
            self.netG = _CondUNet(
                in_ch=stem_in,
                out_ch=opt.output_nc,
                channels=ch,  # type: ignore[arg-type]
                num_res_blocks=n_res,
                up_mode=up_mode,
                use_film=use_film,
                film_where=film_where,
                use_learned_null=use_learned_null,
                cond_nc=cond_nc,
                film_hidden=int(getattr(opt, "fm_film_hidden", 128)),
                use_he_proj=use_he_proj,
                he_proj_init=self.fm_he_proj_init,
            )
            if self.fm_use_seg:
                print(f"FM cond: H&E + seg -> {cond_nc}ch, stem in_ch={stem_in}")
            if use_he_proj:
                print(
                    f"FM he_proj: init={self.fm_he_proj_init} flow={self.fm_flow_path} "
                    f"init_cond={self.fm_init_from_cond} sigma={self.fm_init_noise_sigma} "
                    f"bridge_x0_sigma={self.fm_bridge_x0_sigma} "
                    f"bridge_noise_prob={self.fm_bridge_noise_prob}"
                )
            if use_film:
                print(
                    f"FM FiLM: where={film_where} hidden={opt.fm_film_hidden} "
                    f"reg={self.fm_film_reg}"
                )
            if use_learned_null:
                print("FM CFG v2: learned null cond (cfg_null parameter)")

        lam_perc = float(getattr(opt, "fm_lambda_perc", 0.0)) if self.isTrain else 0.0
        perc_size = int(getattr(opt, "fm_perc_size", 256))
        self.perceptual_loss_fn = None
        self._perc_backend = "off"
        if self.isTrain and self.fm_loss_mode == "x1" and lam_perc > 0:
            self.perceptual_loss_fn, self._perc_backend = build_fm_perceptual(
                self.device, lam_perc, perc_size=perc_size,
            )
            if self.perceptual_loss_fn is None:
                print(
                    "Warning: no perceptual backend (monai/lpips/torchvision); "
                    "install lpips or fix monai — fm_lambda_perc ignored"
                )
            else:
                print(
                    f"FM perceptual: backend={self._perc_backend} "
                    f"lambda={lam_perc} downsample={perc_size if perc_size > 0 else 'full'}"
                )

        self.fm_use_gan = bool(getattr(opt, "fm_use_gan", False)) and self.isTrain
        cw = [float(x) for x in str(getattr(opt, "fm_channel_weights", "1,2,1")).split(",")]
        self.fm_channel_weights = torch.tensor(cw, device=self.device, dtype=torch.float32)

        if self.isTrain:
            if self.perceptual_loss_fn is not None:
                self.loss_names.append("Perc")
            if float(getattr(opt, "fm_lambda_vel", 0.0)) > 0:
                self.loss_names.append("Vel")
            if float(getattr(opt, "fm_lambda_l1", 0.0)) > 0:
                self.loss_names.append("L1")
            if float(getattr(opt, "fm_lambda_sample_l1", 0.0)) > 0:
                self.loss_names.append("L1s")
            if self.fm_film_reg > 0 and bool(getattr(opt, "fm_use_film", False)):
                self.loss_names.append("FilmR")
            if self.fm_use_gan:
                self.loss_names.extend(["G_GAN", "D_real", "D_fake"])
                self.model_names.append("D")
                self.netD = networks.define_D(
                    opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                )
                self.criterionGAN = networks.GANLoss(getattr(opt, "gan_mode", "lsgan")).to(self.device)
                self.optimizer_D = torch.optim.Adam(
                    self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.netG.to(self.device)
            if len(self.gpu_ids) > 1:
                self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        self._cfg_null_inited = False

    def _maybe_init_cfg_null(self) -> None:
        """Seed learned null to batch-mean H&E (not zeros) so CFG uncond path is trainable."""
        if self._cfg_null_inited or not self.isTrain or not self.fm_use_cfg:
            return
        if self.fm_null_mode != "learned" or self.use_monai:
            return
        net = self._unwrap_netg()
        if getattr(net, "cfg_null", None) is None:
            return
        with torch.no_grad():
            mean_he = self._effective_cond().mean(dim=(0, 2, 3), keepdim=True)
            net.cfg_null.data.copy_(mean_he)
        self._cfg_null_inited = True
        print(
            "FM CFG v2: cfg_null init from mean H&E "
            f"(per-channel {net.cfg_null.data.flatten().tolist()})"
        )

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]
        seg = input.get("seg")
        self.real_seg = seg.to(self.device) if seg is not None else None
        self._maybe_init_cfg_null()

    def _cond_tensor(self, he: torch.Tensor,
                    seg: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.fm_use_seg:
            s = seg if seg is not None else getattr(self, "real_seg", None)
            if s is None:
                s = torch.zeros(he.shape[0], 1, he.shape[2], he.shape[3],
                                device=he.device, dtype=he.dtype)
            return torch.cat([he, s], dim=1)
        return he

    def _effective_cond(self) -> torch.Tensor:
        return self._cond_tensor(self.real_A, getattr(self, "real_seg", None))

    def _film_cond_img(self, cond: torch.Tensor) -> torch.Tensor:
        return cond[:, : self.opt.input_nc]

    def _he_proj(self, he: torch.Tensor) -> torch.Tensor:
        net = self._unwrap_netg()
        if getattr(net, "he_proj", None) is not None:
            return net.he_proj(he)
        return he

    def _sample_x0(self, like: torch.Tensor) -> torch.Tensor:
        if self.fm_flow_path == "bridge":
            if (
                self.isTrain
                and self.fm_bridge_noise_prob > 0
                and random.random() < self.fm_bridge_noise_prob
            ):
                return torch.randn_like(like)
            x0 = self._he_proj(self.real_A)
            if self.isTrain and self.fm_bridge_x0_sigma > 0:
                x0 = x0 + self.fm_bridge_x0_sigma * torch.randn_like(like)
            return x0
        return torch.randn_like(like)

    def _ode_x_init(self, cond_A: torch.Tensor) -> torch.Tensor:
        if self.fm_flow_path == "bridge":
            return self._he_proj(cond_A)
        shape = (cond_A.shape[0], self.opt.output_nc, cond_A.shape[2], cond_A.shape[3])
        if self.fm_init_from_cond:
            sigma = self.fm_init_noise_sigma
            noise = torch.randn(shape, device=cond_A.device, dtype=cond_A.dtype)
            return sigma * noise + (1.0 - sigma) * self._he_proj(cond_A)
        return torch.randn(shape, device=cond_A.device, dtype=cond_A.dtype)

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        dist = str(getattr(self.opt, "fm_time_dist", "logit_normal"))
        if self.fm_loss_mode == "velocity":
            return torch.rand(batch_size, device=device)
        if dist == "logit_normal":
            p_mean = float(getattr(self.opt, "fm_P_mean", -0.8))
            p_std = float(getattr(self.opt, "fm_P_std", 0.8))
            return torch.sigmoid(p_mean + p_std * torch.randn(batch_size, device=device))
        return torch.rand(batch_size, device=device)

    def _net_forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.use_monai:
            return self.netG(x_t, t, cond)
        B, _, H, W = x_t.shape
        t_ch = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        inp = torch.cat([x_t, cond, t_ch], dim=1)
        net = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        if getattr(net, "use_film", False):
            return net(inp, cond_img=self._film_cond_img(cond))
        return self.netG(inp)

    def load_networks(self, epoch):
        """Load G; strict=False when FiLM layers are new (finetune from joint_perc)."""
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            load_path = os.path.join(self.save_dir, "%s_net_%s.pth" % (epoch, name))
            net = getattr(self, "net" + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            if not os.path.isfile(load_path):
                if name == "D" and self.fm_use_gan:
                    print(
                        "no checkpoint for %s — init PatchGAN D from scratch "
                        "(expected when finetuning G from joint_perc)" % load_path
                    )
                    continue
                raise FileNotFoundError(load_path)
            print("loading the model from %s" % load_path)
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, "_metadata"):
                del state_dict._metadata
            for key in list(state_dict.keys()):
                self._patch_instance_norm_state_dict(state_dict, net, key.split("."))
            loose = bool(getattr(self.opt, "fm_use_film", False))
            if getattr(self.opt, "fm_use_cfg", False) and getattr(
                self.opt, "fm_null_mode", "zero"
            ) == "learned":
                loose = True
            if getattr(self.opt, "fm_use_seg", False):
                loose = True
            if getattr(self.opt, "fm_flow_path", "noise") == "bridge":
                loose = True
            if getattr(self.opt, "fm_init_from_cond", False):
                loose = True
            incompatible = net.load_state_dict(state_dict, strict=not loose)
            if loose:
                n_miss = len(incompatible.missing_keys)
                n_unexp = len(incompatible.unexpected_keys)
                if n_miss or n_unexp:
                    print(
                        f"  load strict=False (finetune): missing={n_miss} unexpected={n_unexp}"
                    )
                    if n_miss and n_miss <= 20:
                        print("  missing:", incompatible.missing_keys)

    def _unwrap_netg(self) -> nn.Module:
        net = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        return net

    def _null_cond(self, like: torch.Tensor) -> torch.Tensor:
        if self.fm_null_mode == "learned" and not self.use_monai:
            net = self._unwrap_netg()
            if getattr(net, "cfg_null", None) is not None:
                null = net.cfg_null.to(like.device, like.dtype)
                return null.expand(like.shape[0], -1, like.shape[2], like.shape[3])
        return torch.zeros_like(like)

    def _cond_for_train(self, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CFG train: replace H&E with null embedding on dropped samples."""
        if cond is None:
            cond = self._effective_cond()
        if not self.fm_use_cfg or not self.isTrain:
            return cond
        B = cond.shape[0]
        keep = torch.rand(B, device=cond.device) > self.fm_cfg_dropout
        null = self._null_cond(cond)
        return torch.where(keep.view(B, 1, 1, 1), cond, null)

    def _pred_x1(self, x_t: torch.Tensor, t: torch.Tensor,
                 cond: Optional[torch.Tensor] = None,
                 cfg_guidance: bool = False,
                 cfg_scale: Optional[float] = None) -> torch.Tensor:
        cond_a = self._effective_cond() if cond is None else cond
        w = self.fm_cfg_scale if cfg_scale is None else float(cfg_scale)
        if cfg_guidance and self.fm_use_cfg and w != 1.0:
            x1_c = self._net_forward(x_t, t, cond_a)
            x1_u = self._net_forward(x_t, t, self._null_cond(cond_a))
            return x1_u + w * (x1_c - x1_u)
        return self._net_forward(x_t, t, cond_a)

    def _film_regularizer(self) -> torch.Tensor:
        if self.fm_film_reg <= 0 or self.use_monai:
            return torch.tensor(0.0, device=self.device)
        net = self._unwrap_netg()
        if not getattr(net, "use_film", False) or net.film is None:
            return torch.tensor(0.0, device=self.device)
        reg = torch.tensor(0.0, device=self.device)
        film = net.film
        if isinstance(film, FiLMHeadGenerator):
            reg = reg + (film.head.weight ** 2).mean() + (film.head.bias ** 2).mean()
            out_ch = film.head.out_features // 2
            reg = reg + ((film.head.bias[:out_ch] - 1.0) ** 2).mean()
        elif isinstance(film, FiLMGenerator):
            for head in film.heads:
                half = head.out_features // 2
                reg = reg + ((head.bias.data[:half] - 1.0) ** 2).mean()
                reg = reg + (head.bias.data[half:] ** 2).mean()
        return reg * self.fm_film_reg

    def _velocity_from_x1(self, x_t: torch.Tensor, t: torch.Tensor,
                          x1_hat: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1_hat = x1_hat if x1_hat is not None else self._pred_x1(x_t, t)
        denom = (1.0 - t).view(-1, 1, 1, 1).clamp(min=1e-5)
        return (x1_hat - x_t) / denom

    def _velocity_raw(self, x_t: torch.Tensor, t: torch.Tensor,
                      cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond_a = self._effective_cond() if cond is None else cond
        return self._net_forward(x_t, t, cond_a)

    def _ode_step_update(self, x: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor,
                         dt: torch.Tensor, method: str, is_last: bool,
                         cfg_guidance: bool = False,
                         cfg_scale: Optional[float] = None) -> torch.Tensor:
        B = x.shape[0]
        if self.fm_loss_mode == "velocity":
            v0 = self._velocity_raw(x, t0)
            if method == "euler" or is_last:
                return x + dt * v0
            x_mid = x + dt * v0
            v1 = self._velocity_raw(x_mid, t1)
            return x + dt * (v0 + v1) * 0.5

        x1_0 = self._pred_x1(x, t0, cfg_guidance=cfg_guidance, cfg_scale=cfg_scale)
        v0 = self._velocity_from_x1(x, t0, x1_0)
        if method == "euler" or is_last:
            return x + dt * v0
        x_mid = x + dt * v0
        x1_1 = self._pred_x1(x_mid, t1, cfg_guidance=cfg_guidance, cfg_scale=cfg_scale)
        v1 = self._velocity_from_x1(x_mid, t1, x1_1)
        return x + dt * (v0 + v1) * 0.5

    def _integrate(self, cond_A: torch.Tensor, x: torch.Tensor, steps: int,
                   method: str, cfg_guidance: bool = False,
                   cfg_scale: Optional[float] = None) -> torch.Tensor:
        real_A_prev = getattr(self, "real_A", None)
        self.real_A = cond_A
        B = cond_A.shape[0]
        ts = torch.linspace(0.0, 1.0, steps + 1, device=cond_A.device, dtype=cond_A.dtype)

        for i in range(steps):
            t0 = ts[i].expand(B)
            t1 = ts[i + 1].expand(B)
            dt = ts[i + 1] - ts[i]
            x = self._ode_step_update(
                x, t0, t1, dt, method, is_last=(i == steps - 1),
                cfg_guidance=cfg_guidance, cfg_scale=cfg_scale,
            )

        if real_A_prev is None:
            delattr(self, "real_A")
        else:
            self.real_A = real_A_prev
        return x.clamp(-1, 1)

    def _integrate_train_ode(self, cond_A, x, steps, method):
        from torch.utils.checkpoint import checkpoint

        real_A_prev = getattr(self, "real_A", None)
        self.real_A = cond_A
        B = cond_A.shape[0]
        ts = torch.linspace(0.0, 1.0, steps + 1, device=cond_A.device, dtype=cond_A.dtype)

        def step_ckpt(x_in, t0, t1, dt, is_last):
            return self._ode_step_update(x_in, t0, t1, dt, method, is_last)

        for i in range(steps):
            t0 = ts[i].expand(B)
            t1 = ts[i + 1].expand(B)
            dt = ts[i + 1] - ts[i]
            is_last = i == steps - 1
            if x.requires_grad:
                x = checkpoint(step_ckpt, x, t0, t1, dt, is_last, use_reentrant=False)
            else:
                x = step_ckpt(x, t0, t1, dt, is_last)

        if real_A_prev is None:
            delattr(self, "real_A")
        else:
            self.real_A = real_A_prev
        return x.clamp(-1, 1)

    def _ode_config_test(self):
        return int(self.opt.fm_steps), str(getattr(self.opt, "fm_sample_method", "heun"))

    def forward(self):
        steps, method = self._ode_config_test()
        self.fake_B = self.sample(self.real_A, steps=steps, method=method)

    def compute_visuals(self):
        with torch.no_grad():
            steps, method = self._ode_config_test()
            self.fake_B = self.sample(self.real_A, steps=steps, method=method)

    def _loss_fm(self):
        x1 = self.real_B
        x0 = self._sample_x0(x1)
        t = self._sample_t(x1.shape[0], x1.device)
        t_bc = t.view(-1, 1, 1, 1)
        xt = (1.0 - t_bc) * x0 + t_bc * x1

        if self.fm_loss_mode == "velocity":
            cond = self._cond_for_train()
            v_tgt = x1 - x0
            v_pred = self._velocity_raw(xt, t, cond=cond)
            self.loss_FM = F.mse_loss(v_pred, v_tgt)
            loss = self.loss_FM
            lam_perc = float(getattr(self.opt, "fm_lambda_perc", 0.0))
            if self.perceptual_loss_fn is not None:
                x1_hat = xt + (1.0 - t_bc) * v_pred
                self.loss_Perc = self.perceptual_loss_fn(x1_hat, x1) * lam_perc
                loss = loss + self.loss_Perc
        else:
            cond = self._cond_for_train()
            x1_hat = self._pred_x1(xt, t, cond=cond)
            self.loss_FM = self._channel_weighted_l1(x1_hat, x1)
            loss = self.loss_FM
            lam_perc = float(getattr(self.opt, "fm_lambda_perc", 0.0))
            if self.perceptual_loss_fn is not None:
                self.loss_Perc = self.perceptual_loss_fn(x1_hat, x1) * lam_perc
                loss = loss + self.loss_Perc

        lam_vel = float(getattr(self.opt, "fm_lambda_vel", 0.0))
        if lam_vel > 0 and self.fm_loss_mode == "x1":
            v_star = x1 - x0
            v_pred = self._velocity_from_x1(xt, t)
            self.loss_Vel = F.mse_loss(v_pred, v_star) * lam_vel
            loss = loss + self.loss_Vel

        lam_path = float(getattr(self.opt, "fm_lambda_l1", 0.0))
        if lam_path > 0 and self.fm_loss_mode == "x1":
            self.loss_L1 = F.l1_loss(x1_hat, x1) * lam_path
            loss = loss + self.loss_L1

        if self.fm_film_reg > 0:
            self.loss_FilmR = self._film_regularizer()
            loss = loss + self.loss_FilmR

        # Ensure loggable scalars exist for every registered loss name.
        for name in self.loss_names:
            attr = "loss_" + name
            if not hasattr(self, attr):
                setattr(self, attr, loss.new_tensor(0.0))
        return loss

    def _ode_aux_steps(self) -> int:
        gan_steps = int(getattr(self.opt, "fm_gan_sample_steps", 0))
        if gan_steps > 0:
            return gan_steps
        override = int(getattr(self.opt, "fm_train_sample_steps", 0))
        if override > 0:
            return override
        return int(self.opt.fm_steps)

    def _should_run_ode_aux(self) -> bool:
        lam_sample = float(getattr(self.opt, "fm_lambda_sample_l1", 0.0))
        prob_sample = float(getattr(self.opt, "fm_sample_l1_prob", 1.0))
        prob_gan = float(getattr(self.opt, "fm_gan_sample_prob", 0.5))
        need_sample = lam_sample > 0 and random.random() < prob_sample
        need_gan = self.fm_use_gan and random.random() < prob_gan
        return need_sample or need_gan

    def _channel_weighted_l1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.fm_channel_weights.view(1, -1, 1, 1).to(pred.device)
        return (w * F.l1_loss(pred, target, reduction="none")).mean()

    def _sample_train_fake(self) -> torch.Tensor:
        x1 = self.real_B
        steps = self._ode_aux_steps()
        method = str(getattr(self.opt, "fm_train_sample_method", "heun"))
        x_init = self._ode_x_init(self.real_A)
        return self._integrate_train_ode(self.real_A, x_init, steps, method)

    def _loss_ode_sample_l1(self, fake: torch.Tensor) -> torch.Tensor:
        lam = float(getattr(self.opt, "fm_lambda_sample_l1", 0.0))
        self.loss_L1s = self._channel_weighted_l1(fake, self.real_B) * lam
        return self.loss_L1s

    def _backward_D_fm(self, fake_B: torch.Tensor) -> None:
        fake_AB = torch.cat((self.real_A, fake_B.detach()), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def _backward_G_gan_fm(self, fake_B: torch.Tensor) -> None:
        lam_gan = float(getattr(self.opt, "fm_lambda_gan", 1.0))
        fake_AB = torch.cat((self.real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * lam_gan
        self.loss_G_GAN.backward()

    def optimize_parameters(self):
        lam_sample = float(getattr(self.opt, "fm_lambda_sample_l1", 0.0))
        fake_ode: Optional[torch.Tensor] = None
        run_ode = self._should_run_ode_aux()

        if run_ode:
            fake_ode = self._sample_train_fake()

        if self.fm_use_gan and fake_ode is not None:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad(set_to_none=True)
            self._backward_D_fm(fake_ode)
            self.optimizer_D.step()

        self.optimizer_G.zero_grad(set_to_none=True)
        self._loss_fm().backward()
        if fake_ode is not None and lam_sample > 0:
            self._loss_ode_sample_l1(fake_ode).backward()
        if self.fm_use_gan and fake_ode is not None:
            self.set_requires_grad(self.netD, False)
            self._backward_G_gan_fm(fake_ode)
        self.optimizer_G.step()

    @torch.no_grad()
    def sample(self, cond_A: torch.Tensor, steps: Optional[int] = None,
               method: Optional[str] = None,
               cfg_scale: Optional[float] = None) -> torch.Tensor:
        default_steps, default_method = self._ode_config_test()
        steps = int(steps if steps is not None else default_steps)
        method = method or default_method
        w = self.fm_cfg_scale if cfg_scale is None else float(cfg_scale)
        # Guided ODE only at test (test.py: isTrain=False). Train val stays unguided + fast.
        cfg_guidance = self.fm_use_cfg and (not self.isTrain) and w != 1.0
        x_init = self._ode_x_init(cond_A)
        return self._integrate(
            cond_A, x_init, steps, method,
            cfg_guidance=cfg_guidance, cfg_scale=w,
        )
