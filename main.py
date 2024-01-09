import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(type(cfg.model.num_classes))


if __name__ == "__main__":
    my_app()
