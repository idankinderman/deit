import torch

import open_clip

import utils
from torch import nn



class ArgsWrapper:
    def __init__(self, model_type):
        self.model = model_type
        self.openclip_cachedir = None

class ImageEncoder(torch.nn.Module):
    def __init__(self, args, pretrained, keep_lang=False, random_init=False):
        super().__init__()

        print(f'Loading {args.model} pre-trained weights.')
        if '__pretrained__' in args.model:
            name, _ = args.model.split('__pretrained__')
        else:
            name = args.model

        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)

        if random_init:
            # Before the transformer
            nn.init.normal_(self.model.visual.conv1.weight, std=0.02)
            nn.init.normal_(self.model.visual.class_embedding, std=0.02)
            nn.init.normal_(self.model.visual.positional_embedding, std=0.01)
            self.model.visual.ln_pre = nn.LayerNorm(self.model.visual.ln_pre.normalized_shape)

            # Inside the transformer
            proj_std = (self.model.visual.transformer.width ** -0.5) * ((2 * self.model.visual.transformer.layers) ** -0.5)
            attn_std = self.model.visual.transformer.width ** -0.5
            fc_std = (2 * self.model.visual.transformer.width) ** -0.5
            for block in self.model.visual.transformer.resblocks:
                block.ln_1 = nn.LayerNorm(block.ln_1.normalized_shape)
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.in_proj_bias, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.attn.out_proj.bias, std=proj_std)

                block.ln_2 = nn.LayerNorm(block.ln_2.normalized_shape)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_fc.bias, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_proj.bias, std=proj_std)

            # After the transformer
            self.model.visual.ln_post = nn.LayerNorm(self.model.visual.ln_post.normalized_shape)
            nn.init.normal_(self.model.visual.proj, std=0.01)

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')
        if keep_lang and hasattr(self.model, 'transformer'):
            for param in self.model.transformer.parameters():
                param.requires_grad = False

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f'Loading image encoder from {filename}')
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)

    @classmethod
    def load_from_state_dict(cls, model_name, state_dict):
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir)
        self.model.load_from_state_dict(state_dict)




class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        # In case the inputs are a merge of few models, U_output is the matrix that reconstruct the original inputs
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

        self.freeze()

    def freeze(self):
        print('Freezing classification head')
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)

    def unfreeze(self):
        print('Unfreezing classification head')
        self.weight.requires_grad_(True)
        self.bias.requires_grad_(True)

    def randomize(self):
        print(f'Randomizing classification head with dim {self.weight.shape[0]}')
        fc = torch.nn.Linear(512, 100, bias=False)
        self.weight = fc.weight

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)

class ModuleWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, inputs, **kwargs):
        return self.module(inputs, **kwargs)