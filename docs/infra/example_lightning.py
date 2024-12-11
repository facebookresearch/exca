import inspect
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

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        """Checks that the config default values match the _cls defaults"""
        super().__pydantic_init_subclass__(**kwargs)
        super().__init_subclass__()

        if isinstance(cls._cls, type):
            func_or_class = cls._cls.__init__
        else:
            func_or_class = cls._cls
        
        # Get the function signature
        signature = inspect.signature(func_or_class)
        func_params = signature.parameters
        
        # Iterate through the class fields and verify their defaults
        for field_name, field_info in cls.model_fields.items():
            # Check if the field has a default value or is required
            model_default = field_info.default
            model_required = field_info.is_required()
            
            # Check if the parameter exists in the function signature
            if field_name not in func_params:
                raise ValueError(f"Field '{field_name}' is missing in the function parameters.")
            
            func_param = func_params[field_name]
            func_default = func_param.default
            
            # Check if the field is required in both the function and the model
            if model_required != (func_default is inspect.Parameter.empty):
                raise ValueError(f"Field '{field_name}' is required in the model but not in the function or vice versa.")
            
            # If it has a default in both, compare them
            if model_default != func_default and func_default is not inspect.Parameter.empty:
                raise ValueError(f"Field '{field_name}' default value mismatch: model has '{model_default}', function has '{func_default}'.")
    
    def model_post_init(self, log__: tp.Any) -> None:
        """Check that the parameters are compatible with _cls"""
        super().model_post_init(log__)
        exca.helpers.validate_kwargs(self._cls, self.dict())
    
    def build(self, **kwargs):  #  /!\ **kwargs needed for trainer checkpoint, but bad api for uid?
        return self._cls(**self.dict(), **kwargs)


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
    max_epochs: tp.Optional[int] = None
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
    config = args_to_nested_dict(['--trainer.max_epochs=5'] + sys.argv[1:])
    exp = Experiment(**config)
    score = exp.validate()
    print(score)
