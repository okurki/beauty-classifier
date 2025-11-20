import platform
import subprocess
import warnings

import asyncclick as click

from src.infrastructure.ml_models import attractiveness_model
from src.infrastructure.ml_models import celebrity_matcher
from src.config import config


@click.group
def cli():
    pass


@cli.command
@click.option("--train", is_flag=True, default=False, help="Train the model")
@click.option("--eval", is_flag=True, default=False, help="Evaluate the model")
def attractiveness(train: bool, eval: bool):
    if train:
        attractiveness_model.train()
    if eval:
        attractiveness_model.evaluate()


@cli.command
@click.option("--train", is_flag=True, default=False, help="Train the model")
@click.option("--eval", is_flag=True, default=False, help="Evaluate the model")
def celebrity(train: bool, eval: bool):
    if train:
        celebrity_matcher.train()
    if eval:
        celebrity_matcher.evaluate()


def check_vm_reachable():
    ml_remote_ip = config.ml.remote_ip
    if not ml_remote_ip:
        warnings.warn("ML__REMOTE_IP not set. Skipping VM reachability check")
        return True
    param = "-n" if platform.system() == "Windows" else "-c"
    command = ["ping", param, "1", ml_remote_ip]
    result = subprocess.run(command, capture_output=True, timeout=5).returncode
    if not result == 0:
        warnings.warn(
            f"Remote VM ({config.ml.remote_ip}) is not reachable. "
            "Make sure it is running or try turning off VPN"
        )


if __name__ == "__main__":
    check_vm_reachable()
    cli()
