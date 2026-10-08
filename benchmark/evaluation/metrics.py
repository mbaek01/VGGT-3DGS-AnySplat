import numpy as np


def psnr_ssim(pred, gt):
    from skimage.metrics import structural_similarity
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError("Metrics require matching HWC RGB images")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Non-finite pixels")
    mse = float(np.mean((pred.astype(np.float64)-gt.astype(np.float64))**2))
    # Finite, explicit 120 dB cap for identical images keeps results valid JSON.
    psnr = float(-10*np.log10(max(mse, 1e-12)))
    ssim = float(structural_similarity(gt, pred, data_range=1., channel_axis=-1,
                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
    return {"psnr": psnr, "ssim": ssim}


class Metrics:
    def __init__(self, net="alex", device="cpu", skip_lpips=False):
        self.lpips = None
        self.device = device
        if not skip_lpips:
            import lpips
            self.lpips = lpips.LPIPS(net=net, version="0.1").to(device).eval()

    def __call__(self, pred, gt):
        result = psnr_ssim(pred, gt)
        result["lpips"] = None
        if self.lpips is not None:
            import torch
            def tensor(x):
                return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)*2-1
            with torch.no_grad():
                result["lpips"] = float(self.lpips(tensor(pred), tensor(gt)).item())
        return result
