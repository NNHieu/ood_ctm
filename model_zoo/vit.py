import collections
import torch
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification
import torchvision.transforms as T

class ViTWarpper(torch.nn.Module):
    def __init__(self, name='google/vit-base-patch16-224-in21k', num_classes=None, id2label=None, label2id=None) -> None:
        super().__init__()
        self.processor, self.model = _get_model(name, num_classes, id2label, label2id)
        image_mean, image_std = self.processor.image_mean, self.processor.image_std
        size = self.processor.size["height"]
        self.transform = T.Compose(
            [
                T.Resize(size),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=image_mean, std=image_std),
            ]
        )
        
    def load(self, ckpt_path):
        if ckpt_path is None:
            print("No checkpoint path is provided. Use random initialization.")
            return
        tm = torch.load(ckpt_path, map_location="cpu")
        if isinstance(tm, collections.OrderedDict):
            state_dict = tm
        elif isinstance(tm, dict):
            # print(tm.keys())
            if 'state_dict' in tm:
                state_dict = tm['state_dict']
            else:
                state_dict = tm
        else:
            state_dict = tm.state_dict()
        self.model.load_state_dict(state_dict)
    
    def forward(self, *args, return_feature_list=False, **kwargs):
        if return_feature_list:
            output = self.model(*args, output_hidden_states=True, **kwargs)
            return output.logits, [output.hidden_states[-1][:, 0, :],]
        else:
            output = self.model(*args, **kwargs)
            return output.logits
    
    @property
    def fc(self):
        return self.model.classifier

def _get_model(name='google/vit-base-patch16-224-in21k', num_classes=None, id2label=None, label2id=None):
    processor = ViTImageProcessor.from_pretrained(name)
    print("Loaded processor")
    # print(type(id2label))
    # print(label2id)
    id2label = dict(id2label)
    label2id = dict(label2id)
    model = ViTForImageClassification.from_pretrained(name,
                                                  id2label=id2label,
                                                  label2id=label2id)
    print("Loaded model")
    return processor, model

def get_model(name='google/vit-base-patch16-224-in21k', num_classes=None, id2label=None, label2id=None):
    return ViTWarpper(name, num_classes, id2label, label2id)