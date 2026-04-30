#References 
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
# https://huggingface.co/docs/transformers/en/model_doc/vit

import torch.nn as nn
import timm

class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Load pretrained ViT
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)

        # Start with a frozen backbone
        for param in self.vit.parameters():
            param.requires_grad = False

        # Replace classifier head with a more robust version
        in_features = self.vit.head.in_features
        self.vit.head = nn.Sequential(
            nn.Dropout(0.3), # Helps prevent memorization of small data
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.vit(x)