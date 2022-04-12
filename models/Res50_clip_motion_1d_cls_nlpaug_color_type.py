import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision.models import resnet50
from models.SENet import se_resnext50_32x4d
from models.Clip import clip
from torch.nn import GRU

class Res50_clip_motion_1d_cls_nlpaug_color_type(nn.Module):
    def __init__(self, cfg_model):
        super(Res50_clip_motion_1d_cls_nlpaug_color_type, self).__init__()
        self.SPLIT_TEXT = cfg_model.SPLIT_TEXT
        self.encoder_motion = self.get_motion_encoder()
        self.encoder_img = self.get_img_encoder()
        self.encoder_text = self.get_text_encoder(cfg_model)
        dim_text = 512
        dim_img = 2048
        dim_embedding = cfg_model.EMBED_DIM
        # Add split text fusion model
        # self.fc_text_fusion = nn.Sequential(nn.Linear(3*dim_text, 3*dim_text), 
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(3*dim_text, dim_embedding))

        self.fc_text_obj = nn.Sequential(nn.Linear(512, dim_embedding), 
                                         nn.ReLU(inplace=True),
                                         nn.Linear(dim_embedding, dim_embedding))
        
        self.fc_obj = nn.Sequential(
            nn.Conv2d(dim_img, dim_embedding, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_color_1 = nn.Sequential(nn.Linear(dim_embedding, dim_embedding), 
                                        nn.ReLU(inplace=True),)
        self.fc_color_2 = nn.Sequential(nn.Linear(dim_embedding, 8),)

        self.fc_type_1 = nn.Sequential(nn.Linear(dim_embedding, dim_embedding),
                                        nn.ReLU(inplace=True),)
        self.fc_type_2 = nn.Sequential(nn.Linear(dim_embedding, 9))

        self.fc_motion = nn.Sequential(nn.Linear(1024, 1024), 
                                         nn.ReLU(inplace=True),
                                         nn.Linear(1024, 2))

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
        encoder = GRU(input_size=4,hidden_size=1024,num_layers=4,batch_first=True)
        return encoder

    @staticmethod
    def get_text_encoder(cfg_model):
        # encoder = BertModel.from_pretrained(cfg_model.BERT_NAME)
        encoder,preprocess = clip.load('/home/zby/AICity2022Track2/pretrained/CLIP_ViT_B_32.pt')
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
        # ENCODERS
        features_obj = self.encoder_img(x['crop'])
        features_obj = self.fc_obj(features_obj).squeeze()
        outputs, features_motion = self.encoder_motion(x['motion_boxes'])

        # For main text data
        features_text = self.encoder_text.encode_text(x['text_input_ids']).to(torch.float)
        features_text_obj = self.fc_text_obj(features_text)

        # For Visual features
        features_color = self.fc_color_1(features_obj)
        output_color = self.fc_color_2(features_color)
        features_type =  self.fc_type_1(features_obj)
        output_type = self.fc_type_2(features_type)
        
        # For motion features
        features_motion = features_motion[-1]
        motion_score = self.fc_motion(features_motion)

        # Normalization for main text
        features_obj_main, features_text_obj_main = map(lambda t: F.normalize(t, p=2, dim=-1), (features_obj, features_text_obj))
        # motion_score = motion_score.softmax(dim=1)

        return [(features_obj_main, features_text_obj_main),],\
                output_color, output_type, self.tau, motion_score,\
                [features_text]

