"""
@Author: Du Yunhao
@Filename: model.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/9 11:05
@Discription: model
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision.models import resnet50
from models.SENet import se_resnext50_32x4d

class MultiStreamNetwork(nn.Module):
    def __init__(self, cfg_model):
        super(MultiStreamNetwork, self).__init__()
        self.encoder_img = self.get_img_encoder()
        self.encoder_text = self.get_text_encoder(cfg_model)
        dim_text = 768
        dim_img = 2048
        dim_embedding = cfg_model.EMBED_DIM
        self.fc_text = nn.Sequential(
            nn.Linear(dim_text, dim_text),
            nn.ReLU(),
            nn.Linear(dim_text, dim_embedding)
        )
        self.fc_img = nn.Sequential(
            nn.Conv2d(dim_img, dim_embedding, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.tau = nn.Parameter(torch.ones(1), requires_grad=True)

    @staticmethod
    def get_img_encoder():
        encoder = resnet50(pretrained=True)
        features = list(encoder.children())[:-2]  # 去掉GAP和FC
        encoder = nn.Sequential(*features)
        # encoder = se_resnext50_32x4d(pretrained=None)
        # ckpt = torch.load('/home/zby/AICity2022Track2/pretrain/motion_SE_NOCLS_nlpaug_288.pth')
        # state_dict = dict()
        # for key, value in ckpt['state_dict'].items():
        #     if 'vis_backbone.' in key:
        #         state_dict[key.replace('module.vis_backbone.', '')] = value
        # encoder.load_state_dict(state_dict)
        return encoder

    @staticmethod
    def get_text_encoder(cfg_model):
        encoder = BertModel.from_pretrained(cfg_model.BERT_NAME)
        if cfg_model.FREEZE_TEXT_ENCODER:
            for p in encoder.parameters():
                p.requires_grad = False
        return encoder

    def encode_img(self,x):
        features_crop = self.encoder_img(x['crop'])
        features_crop = self.fc_img(features_crop).squeeze()
        features_crop = F.normalize(features_crop, p=2, dim=-1)
        return features_crop

    def encode_text(self,x):
        features_text = self.encoder_text(x['text_input_ids'], attention_mask=x['text_attention_mask'])
        features_text = torch.mean(features_text.last_hidden_state, dim=1)
        features_text = self.fc_text(features_text)
        features_text = F.normalize(features_text, p=2, dim=-1)
        return features_text

    def forward(self, x):
        """
        keys of x: crop, text_input_ids, text_attention_mask,
        """
        features_crop = self.encoder_img(x['crop'])
        features_crop = self.fc_img(features_crop).squeeze()
        features_text = self.encoder_text(x['text_input_ids'], attention_mask=x['text_attention_mask'])
        features_text = torch.mean(features_text.last_hidden_state, dim=1)
        features_text = self.fc_text(features_text)
        features_crop, features_text = map(lambda t: F.normalize(t, p=2, dim=-1), (features_crop, features_text))
        return [(features_crop, features_text)], self.tau
