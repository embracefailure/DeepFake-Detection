import torch 
from PIL import Image
import torch.nn as nn

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 1024,
    "ViT-B/14" : 768
    
}

class DINOModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(DINOModel, self).__init__()

        self.model= torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitl14") #backbone加载，如果想测pretrained，就用vitb14！,我自己训的时候用了vitl14
        #print(self.model)
        self.fc = nn.Linear( CHANNELS[name], num_classes )

    def forward(self, x, return_feature=False):
        features = self.model(x) 
        #print(features)
        if return_feature:
            return features
        return self.fc(features)
