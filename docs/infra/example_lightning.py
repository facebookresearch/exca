import sys
import typing as tp

import exca
import pydantic
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18


class ResNet(pl.LightningModule):
    def __init__(self, pretrained: bool=True, learning_rate: float=0.001):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.model = resnet18(pretrained=pretrained)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def _step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class Mnist(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def _dataloader(self, train: bool):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dset = datasets.MNIST('', train=train, download=True, transform=transform)
        return torch.utils.data.DataLoader(dset, batch_size=self.batch_size)

    def train_dataloader(self):
        return self._dataloader(train=True)

    def val_dataloader(self):
        return self._dataloader(train=False)


class AutoConfig(pydantic.BaseModel):  # TODO move to exca.helpers?
    model_config = pydantic.ConfigDict(extra="forbid")
    _cls: tp.ClassVar[tp.Any]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        exca.helpers.validate_kwargs(self._cls, self._public_params())
    
    def _public_params(self):
        return {k: v for k, v in self.dict().items() if not k.startswith("_")}

    def build(self, **kwargs):
        return self._cls(**self._public_params(), **kwargs)

def args_to_nested_dict(args: list[str]) -> tp.Dict[str, tp.Any]:  # TODO move to exca.helpers?
    """
    Parses a list of Bash-style arguments (e.g., --key=value) into a nested dict.
    """
    nested_dict = {}
    for arg in args:
        # Split argument into key and value
        key, value = arg.lstrip("--").split("=", 1)
        # Convert flat key into a nested dictionary
        keys = key.split(".")
        current_level = nested_dict
        for k in keys[:-1]:
            current_level = current_level.setdefault(k, {})
        current_level[keys[-1]] = value
    return nested_dict


class ModelConfig(AutoConfig):  # question: right design?
    pretrained: bool = True
    learning_rate: float = 0.001
    _cls = ResNet
    

class MnistConfig(AutoConfig):
    batch_size: int = 64  # question: uid change if add new param, but corresponds to default trainer?
    _cls = Mnist


class TrainerConfig(AutoConfig):
    max_epochs: int = 5
    _cls = Trainer


class Experiment(pydantic.BaseModel):
    model: ModelConfig = ModelConfig()
    data: MnistConfig = MnistConfig()
    trainer: TrainerConfig = TrainerConfig()

    infra: exca.TaskInfra = exca.TaskInfra()

    def build(self):
        mnist = self.data.build()
        model = self.model.build()
        callbacks = None
        if self.infra.folder:
            callbacks = [ModelCheckpoint(
                dirpath=self.infra.uid_folder() / 'checkpoint',
                save_top_k=1,
                monitor="val_loss",
                mode="min")
            ]
        trainer = self.trainer.build(callbacks=callbacks)
        return mnist, model, trainer

    @infra.apply
    def fit(self):
        data_loaders, model, trainer = self.build()
        # Define the checkpoint directory
        checkpoint_dir = self.infra.uid_folder() / 'checkpoint'

        # Find the latest checkpoint if it exists
        checkpoints = sorted(checkpoint_dir.glob('*.ckpt'))
        ckpt_path = sorted(checkpoints)[-1] if checkpoints else None
        
        # Fit model
        trainer.fit(model, data_loaders, ckpt_path=ckpt_path)
        return model

    def validate(self):
        data_loaders, _, trainer = self.build()
        model = self.fit()
        return trainer.validate(model, dataloaders=data_loaders.val_dataloader())


if __name__ == '__main__':
    config = args_to_nested_dict(sys.argv[1:])
    exp = Experiment(**config)
    score = exp.validate()
    print(score)
