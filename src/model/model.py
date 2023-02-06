import timm
import torch
import torch.nn as nn

from .layer import get_layer


class EncoderWithHead(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool, 
        layer_name: str, 
        embedding_size: int,
        num_classes: int,
        ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.layer_name = layer_name
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        self.encoder = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.embedding_size,
        )

        self.head = get_layer(
            layer_name = self.layer_name,
            embedding_size=self.embedding_size,
            num_classes=self.num_classes,
        )


    def forward(self, images, targets=None) -> torch.tensor:
        features = self.encoder(images)
        if targets is None:
            return features
        outputs = self.head(features, targets)
        return outputs