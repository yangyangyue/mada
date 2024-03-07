import random
from typing import Union

import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table

from LilyVAE.vaenet import VaeNet
from util.config import UkdaleConfig, ReddConfig
from util.load_data import get_loaders
from util.metric import Metric
import numpy as np


def test(app_name: str, config: Union[ReddConfig, UkdaleConfig]):
    """ train and val for app specified by app_name"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path = f"./LilyVAE/weights/{app_name}.pth"
    # data loader
    test_loader = get_loaders(app_name, config, False)
    # build model and load its weight through saved weights
    checkpoint = torch.load(weight_path)
    if config.method == 'vae':
        (model := VaeNet(*checkpoint['stat']).to(device)).load_state_dict(checkpoint['model'])
    else:
        raise ValueError("method must in (vae, s2p)")

    mmain= np.empty((0, config.window_size))

    model.eval()
    metric = Metric(threshold := config.threshold[app_name], config.window_size, config.window_stride)
    for aggs, apps, status in track(test_loader, description=f"{app_name}: "):
        aggs, apps, status = aggs.to(device), apps.to(device), status.to(device)
        apps_pred = model(aggs)
        apps_pred[apps_pred < threshold] = 0
        metric.add(apps, apps_pred)
        mmain = np.concatenate([mmain, aggs.squeeze().cpu().detach().numpy()])
    result = np.stack([metric.merge(mmain, False), metric.merge(metric.apps, False), metric.merge(metric.apps_pred, False)]).transpose(1, 0)
    np.savetxt(f"{app_name}.csv", result)
    return metric.get_metrics()


def test_all_app():
    table = Table(title="the experiments of all appliances")
    table.add_column("appliance", style="blue")
    table.add_column("mae", style="magenta")
    table.add_column("mae_on", style="cyan")
    table.add_column("acc", style="green")
    table.add_column("pre", style="green")
    table.add_column("rec", style="green")
    table.add_column("f1", style="green")
    # ukdale
    config = UkdaleConfig()
    for app_name in config.app_names:
        table.add_row(app_name, *[str(index) for index in test(app_name, config)])
    # redd
    # config = ReddConfig()
    # for app_name in config.app_names:
    #     table.add_row(app_name, *[str(index) for index in test(app_name, config)])

    console = Console()
    console.print(table)


if __name__ == "__main__":
    # fix the random with particular seed
    seed = 12345
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.set_default_tensor_type(torch.FloatTensor)

    test_all_app()
