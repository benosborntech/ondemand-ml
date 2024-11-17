import torch
import torch.nn as nn
from transformers import ViTModel 


class ViTYOLO(nn.Module):
    def __init__(self, num_classes: int, boxes_per_cell: int, grid_size: int, hidden_size: int, vit_model_name="google/vit-base-patch16-224-in21k"):
        super(ViTYOLO, self).__init__()

        self.num_classes = num_classes
        self.boxes_per_cell = boxes_per_cell  # number of bounding boxes per grid cell (for simplicity)
        self.grid_size = grid_size # break the image into a grid and have an output for each cell
        self.hidden_size = hidden_size # the size of the NN hidden layer output
        
        # Load pretrained ViT model
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit_hidden_size = self.vit.config.hidden_size # assume output from ViT model is of shape (batch_size, num_patches, hidden_size)

        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.bbox_head = nn.Sequential(
            nn.Linear(self.vit_hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.grid_size * self.grid_size * self.boxes_per_cell * (4 + self.num_classes + 1))
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        vit_output = self.vit(inputs)  # (batch_size, num_patches, hidden_size)
        features = vit_output.last_hidden_state  # (batch_size, num_patches, hidden_size)
        
        # We can average the patches features or just use the last token (CLS token) for simplicity
        # For simplicity, let's just use the global average pooling from all patches
        pooled_features = features.mean(dim=1)  # (batch_size, hidden_size)
        
        # Pass pooled features through YOLO head
        yolo_output = self.bbox_head(pooled_features)
        
        return yolo_output

    def reshape(self, inputs: torch.Tensor) -> torch.Tensor:
        # Reshape to (batch_size, grid_size * grid_size, bounding boxes, labels)
        return inputs.view(-1, self.grid_size * self.grid_size, self.boxes_per_cell, (4 + self.num_classes + 1))