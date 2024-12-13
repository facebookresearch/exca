import typing as tp

import exca
import pydantic
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18


class ModelConfig(pydantic.BaseModel):
    pretrained: bool = True
    learning_rate: float = 0.001
    model_config = pydantic.ConfigDict(extra="forbid")


class DataConfig(pydantic.BaseModel):
    batch_size: int = 64
    model_config = pydantic.ConfigDict(extra="forbid")


class TrainerConfig(pydantic.BaseModel):
    max_epochs: tp.Optional[int] = None
    model_config = pydantic.ConfigDict(extra="forbid")


class Model(pl.LightningModule):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super(Model, self).__init__()
        self.config = config
        self.model = resnet18(pretrained=config.pretrained)
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
        return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)


class Data(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config

    def _dataloader(self, train: bool):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dset = datasets.MNIST('', train=train, download=True, transform=transform)
        return torch.utils.data.DataLoader(dset, batch_size=self.config.batch_size)

    def train_dataloader(self):
        return self._dataloader(train=True)

    def val_dataloader(self):
        return self._dataloader(train=False)
    

class Experiment(pydantic.BaseModel):
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    trainer: TrainerConfig = TrainerConfig()
    infra: exca.TaskInfra = exca.TaskInfra(folder='.cache/')

    def build(self):
        data = Data(self.data)
        model = Model(self.model)
        
        # Add checkpoint callback based on infra folder
        callbacks = None
        if self.infra.folder:
            callbacks = [ModelCheckpoint(
                dirpath=self.infra.uid_folder() / 'checkpoint',
                save_top_k=1,
                monitor="val_loss",
                mode="min")
            ]

        trainer = Trainer(**self.trainer.dict(), callbacks=callbacks)
        return data, model, trainer

    @property
    def checkpoint_path(self):
        # Define the checkpoint directory
        checkpoint_dir = self.infra.uid_folder() / 'checkpoint'

        # Find the latest checkpoint if it exists
        checkpoints = sorted(checkpoint_dir.glob('*.ckpt'))
        ckpt_path = sorted(checkpoints)[-1] if checkpoints else None
        return ckpt_path

    @infra.apply
    def fit(self):
        # Configure
        data_loaders, model, trainer = self.build()
        
        # Fit model
        trainer.fit(model, data_loaders, ckpt_path=self.checkpoint_path)
        
        # Return model if not saved 
        if self.checkpoint_path is None:
            return model

    def validate(self):
        data_loaders, model, trainer = self.build()
        trained_model = self.fit()
        if trained_model is None:
            trained_model = model.__class__.load_from_checkpoint(self.checkpoint_path)

        return trainer.validate(trained_model, dataloaders=data_loaders.val_dataloader())


if __name__ == '__main__':
    config = dict(
        model={'learning_rate': .01},
        trainer={'max_epochs': 2},
        infra={'folder': '.cache/'}
    )
    exp = Experiment(**config)
    score = exp.validate()
    print(score)
