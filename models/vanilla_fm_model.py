import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


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


class _UpSkip(nn.Module):
    """Upsample, concat encoder skip, fuse, then ResBlocks."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, n_res: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.res(x)


class _CondUNet(nn.Module):
    """
    Conditional U-Net for (vanilla) flow matching (encoder skips on decode).

    Input: concat([x_t (B,3,H,W), cond_A (B,3,H,W), t_channel (B,1,H,W)]) => 7ch
    Output: velocity field v_theta (B,3,H,W)
    """

    def __init__(self, in_ch: int = 7, out_ch: int = 3,
                 channels: Tuple[int, int, int] = (32, 64, 96),
                 num_res_blocks: int = 1):
        super().__init__()
        c1, c2, c3 = channels
        self.stem = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.res1 = nn.Sequential(*[_ResBlock(c1) for _ in range(num_res_blocks)])
        self.down1 = _Down(c1, c2, num_res_blocks)
        self.down2 = _Down(c2, c3, num_res_blocks)
        self.mid = nn.Sequential(*[_ResBlock(c3) for _ in range(max(1, num_res_blocks))])
        self.up2 = _UpSkip(c3, c2, c2, num_res_blocks)
        self.up1 = _UpSkip(c2, c1, c1, num_res_blocks)
        self.head = nn.Conv2d(c1, out_ch, 3, padding=1)

    def forward(self, x):
        h1 = self.res1(self.stem(x))
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h = self.mid(h3)
        h = self.up2(h, h2)
        h = self.up1(h, h1)
        return self.head(h)


class VanillaFMModel(BaseModel):
    """
    Vanilla conditional rectified flow matching (pix2pix framework).

    Training: one UNet forward — MSE(v_theta(x_t,t|A), x_1 - x_0) at random t.
    Inference / val: ODE integrate dx/dt = v_theta from noise (fm_steps, no grad).
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode="aligned", no_dropout=True)
        parser.add_argument("--fm_channels", type=str, default="32,64,96",
                            help="UNet channels e.g. 32,64,96 (tune params)")
        parser.add_argument("--fm_num_res_blocks", type=int, default=1,
                            help="ResBlocks per level (tune params)")
        parser.add_argument("--fm_steps", type=int, default=25,
                            help="ODE steps at test (and val if fm_val_steps unset)")
        parser.add_argument("--fm_val_steps", type=int, default=8,
                            help="ODE steps during training-time validation only")
        parser.add_argument("--fm_sample_method", type=str, default="heun",
                            choices=["euler", "heun"],
                            help="ODE solver at val / test (not used in standard train)")
        if is_train:
            parser.add_argument("--fm_lambda_l1", type=float, default=0.0,
                                help="Optional path L1 on x1_hat (0 = off; not standard FM)")
            parser.add_argument("--fm_lambda_sample_l1", type=float, default=0.0,
                                help="Optional ODE sample L1 in train loop (0 = standard FM)")
            parser.add_argument("--fm_train_sample_steps", type=int, default=0,
                                help="If fm_lambda_sample_l1>0: ODE steps (0 = use fm_steps)")
            parser.add_argument("--fm_sample_l1_prob", type=float, default=1.0,
                                help="With sample L1: fraction of iters that run ODE loss")
            parser.add_argument("--fm_train_sample_method", type=str, default="euler",
                                choices=["euler", "heun"],
                                help="ODE solver when fm_lambda_sample_l1>0 only")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["FM"]
        if self.isTrain:
            if float(getattr(opt, "fm_lambda_l1", 0.0)) > 0:
                self.loss_names.append("L1")
            if float(getattr(opt, "fm_lambda_sample_l1", 0.0)) > 0:
                self.loss_names.append("L1s")
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.model_names = ["G"]

        ch = tuple(int(x) for x in opt.fm_channels.split(","))
        self.netG = _CondUNet(
            in_ch=opt.input_nc + opt.output_nc + 1,
            out_ch=opt.output_nc,
            channels=ch,
            num_res_blocks=int(opt.fm_num_res_blocks),
        )
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.netG.to(self.device)
            if len(self.gpu_ids) > 1:
                self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def _ode_config_test(self):
        return int(self.opt.fm_steps), str(getattr(self.opt, "fm_sample_method", "heun"))

    def _ode_config_val(self):
        return int(getattr(self.opt, "fm_val_steps", 8)), str(
            getattr(self.opt, "fm_sample_method", "heun"))

    def _v(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x_t.shape
        t_ch = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        inp = torch.cat([x_t, self.real_A, t_ch], dim=1)
        return self.netG(inp)

    def _integrate(self, cond_A: torch.Tensor, x: torch.Tensor, steps: int,
                   method: str) -> torch.Tensor:
        """ODE solve dx/dt = v(x,t|A). Inference / val only (no grad)."""
        real_A_prev = getattr(self, "real_A", None)
        self.real_A = cond_A
        B = cond_A.shape[0]
        ts = torch.linspace(0.0, 1.0, steps + 1, device=cond_A.device, dtype=cond_A.dtype)

        for i in range(steps):
            t0 = ts[i].expand(B)
            t1 = ts[i + 1].expand(B)
            dt = ts[i + 1] - ts[i]
            v0 = self._v(x, t0)
            if method == "euler" or i == steps - 1:
                x = x + dt * v0
            else:
                x_mid = x + dt * v0
                v1 = self._v(x_mid, t1)
                x = x + dt * (v0 + v1) * 0.5

        if real_A_prev is None:
            delattr(self, "real_A")
        else:
            self.real_A = real_A_prev
        return x.clamp(-1, 1)

    def forward(self):
        steps, method = self._ode_config_test()
        self.fake_B = self.sample(self.real_A, steps=steps, method=method)

    def compute_visuals(self):
        with torch.no_grad():
            steps, method = self._ode_config_test()
            self.fake_B = self.sample(self.real_A, steps=steps, method=method)

    def _loss_fm_velocity(self):
        """Standard FM: one forward, ||v_theta - (x1 - x0)||^2."""
        x1 = self.real_B
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1.0 - t).view(-1, 1, 1, 1) * x0 + t.view(-1, 1, 1, 1) * x1
        v_star = x1 - x0
        v_pred = self._v(xt, t)
        self.loss_FM = F.mse_loss(v_pred, v_star)
        loss = self.loss_FM

        lam_path = float(getattr(self.opt, "fm_lambda_l1", 0.0))
        if lam_path > 0:
            t_bc = t.view(-1, 1, 1, 1)
            x1_hat = xt + (1.0 - t_bc) * v_pred
            self.loss_L1 = F.l1_loss(x1_hat, x1) * lam_path
            loss = loss + self.loss_L1
        return loss

    def _loss_ode_sample_l1(self):
        """Non-standard: differentiable ODE + L1 (opt-in via fm_lambda_sample_l1)."""
        x1 = self.real_B
        override = int(getattr(self.opt, "fm_train_sample_steps", 0))
        steps = override if override > 0 else int(self.opt.fm_steps)
        method = str(getattr(self.opt, "fm_train_sample_method", "euler"))
        x_init = torch.randn_like(x1)
        # Requires grad through ODE — only used when explicitly enabled.
        fake = self._integrate_train_ode(self.real_A, x_init, steps, method)
        lam = float(getattr(self.opt, "fm_lambda_sample_l1", 0.0))
        self.loss_L1s = F.l1_loss(fake, x1) * lam
        return self.loss_L1s

    def _integrate_train_ode(self, cond_A, x, steps, method):
        """Grad-enabled ODE for optional fm_lambda_sample_l1 > 0."""
        from torch.utils.checkpoint import checkpoint

        real_A_prev = getattr(self, "real_A", None)
        self.real_A = cond_A
        B = cond_A.shape[0]
        ts = torch.linspace(0.0, 1.0, steps + 1, device=cond_A.device, dtype=cond_A.dtype)

        def v_ckpt(x_t, t):
            return self._v(x_t, t)

        for i in range(steps):
            t0 = ts[i].expand(B)
            t1 = ts[i + 1].expand(B)
            dt = ts[i + 1] - ts[i]
            v0 = checkpoint(v_ckpt, x, t0, use_reentrant=False) if x.requires_grad else self._v(x, t0)
            if method == "euler" or i == steps - 1:
                x = x + dt * v0
            else:
                x_mid = x + dt * v0
                v1 = checkpoint(v_ckpt, x_mid, t1, use_reentrant=False) if x.requires_grad else self._v(x_mid, t1)
                x = x + dt * (v0 + v1) * 0.5

        if real_A_prev is None:
            delattr(self, "real_A")
        else:
            self.real_A = real_A_prev
        return x.clamp(-1, 1)

    def backward_G(self):
        self._loss_fm_velocity().backward()
        lam_sample = float(getattr(self.opt, "fm_lambda_sample_l1", 0.0))
        prob = float(getattr(self.opt, "fm_sample_l1_prob", 1.0))
        if lam_sample > 0 and random.random() < prob:
            self._loss_ode_sample_l1().backward()

    def optimize_parameters(self):
        self.optimizer_G.zero_grad(set_to_none=True)
        self._loss_fm_velocity().backward()

        lam_sample = float(getattr(self.opt, "fm_lambda_sample_l1", 0.0))
        prob = float(getattr(self.opt, "fm_sample_l1_prob", 1.0))
        if lam_sample > 0 and random.random() < prob:
            self._loss_ode_sample_l1().backward()

        self.optimizer_G.step()

    @torch.no_grad()
    def sample(self, cond_A: torch.Tensor, steps: Optional[int] = None,
               method: Optional[str] = None) -> torch.Tensor:
        default_steps, default_method = self._ode_config_test()
        steps = int(steps if steps is not None else default_steps)
        method = method or default_method
        x_init = torch.randn(
            cond_A.shape[0], self.opt.output_nc, cond_A.shape[2], cond_A.shape[3],
            device=cond_A.device, dtype=cond_A.dtype,
        )
        return self._integrate(cond_A, x_init, steps, method)
