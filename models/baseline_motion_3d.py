import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision.models import resnet50
from torchvision.models.video import r2plus1d_18
from models.SENet import se_resnext50_32x4d

class Baseline_motion_3d(nn.Module):
    def __init__(self, cfg_model):
        super(Baseline_motion_3d, self).__init__()
        self.encoder_motion = self.get_motion_encoder()
        self.encoder_img = self.get_img_encoder()
        self.encoder_text = self.get_text_encoder(cfg_model)
        dim_text = 768
        dim_img = 2048
        dim_embedding = cfg_model.EMBED_DIM
        self.fc_text_fusion = nn.Sequential(nn.Linear(dim_text, dim_text), 
                                         nn.ReLU(inplace=True),
                                         nn.Linear(dim_text, dim_embedding))

        self.fc_text_obj = nn.Sequential(nn.Linear(dim_embedding, dim_embedding), 
                                         nn.ReLU(inplace=True),
                                         nn.Linear(dim_embedding, dim_embedding))
        self.fc_text_motion = nn.Sequential(nn.Linear(dim_embedding, dim_embedding),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(dim_embedding, dim_embedding))
        
        self.fc_obj = nn.Sequential(
            nn.Conv2d(dim_img, dim_embedding, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_motion = nn.Sequential(
            nn.Conv3d(512, dim_embedding, kernel_size=1),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc_fusion = nn.Linear(2*dim_embedding,dim_embedding)

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
    def get_motion_encoder():
        encoder = r2plus1d_18(pretrained=True)
        features = list(encoder.children())[:-2]  # 去掉GAP和FC
        encoder = nn.Sequential(*features)
        
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
        features_obj = self.encoder_img(x['crop'])
        features_obj = self.fc_obj(features_obj).squeeze()
        features_motion = self.encoder_motion(x['motion_3d'])
        features_motion = self.fc_motion(features_motion).squeeze()
        features_fusion = self.fc_fusion(torch.cat((features_obj,features_motion),dim=1))

        features_text = self.encoder_text(x['text_input_ids'], attention_mask=x['text_attention_mask'])
        features_text = torch.mean(features_text.last_hidden_state, dim=1)
        features_text_fusion = self.fc_text_fusion(features_text)
        features_text_obj = self.fc_text_obj(features_text_fusion)
        features_text_motion = self.fc_text_motion(features_text_fusion)
        

        features_obj, features_text_obj = map(lambda t: F.normalize(t, p=2, dim=-1), (features_obj, features_text_obj))
        features_motion, features_text_motion = map(lambda t: F.normalize(t, p=2, dim=-1), (features_motion, features_text_motion))
        features_fusion, features_text_fusion = map(lambda t: F.normalize(t, p=2, dim=-1), (features_fusion, features_text_fusion))
        return [(features_obj, features_text_obj), (features_motion, features_text_motion), (features_fusion,features_text_fusion)],self.tau
