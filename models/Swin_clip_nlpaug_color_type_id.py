import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision.models import resnet50
from models.SENet import se_resnext50_32x4d
from models.Clip import clip
from torch.nn import GRU
from models.Swin.swin_transformer import SwinTransformer

class Swin_clip_nlpaug_color_type_id(nn.Module):
    def __init__(self, cfg_model):
        super(Swin_clip_nlpaug_color_type_id, self).__init__()
        self.SPLIT_TEXT = cfg_model.SPLIT_TEXT
        # self.encoder_motion = self.get_motion_encoder()
        self.encoder_img = self.get_img_encoder()
        self.encoder_text = self.get_text_encoder(cfg_model)
        dim_text = 512
        dim_img = 2048
        dim_embedding = cfg_model.EMBED_DIM
        # Add split text fusion model
        # self.fc_text_fusion = nn.Sequential(nn.Linear(3*dim_text, 3*dim_text), 
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(3*dim_text, dim_embedding))

        self.fc_text_fusion = nn.Sequential(nn.Linear(dim_text, dim_text), 
                                         nn.ReLU(inplace=True),
                                         nn.Linear(dim_text, dim_embedding))

        self.fc_text_obj = nn.Sequential(nn.Linear(dim_embedding, dim_embedding), 
                                         nn.ReLU(inplace=True),
                                         nn.Linear(dim_embedding, dim_embedding))

        self.fc_text_motion = nn.Sequential(nn.Linear(dim_embedding, dim_embedding),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(dim_embedding, dim_embedding))
        
        self.fc_obj = nn.Sequential(nn.Linear(1024, dim_embedding))
        
        self.fc_color_1 = nn.Sequential(nn.Linear(dim_embedding, dim_embedding), 
                                        nn.ReLU(inplace=True),)
        self.fc_color_2 = nn.Sequential(nn.Linear(dim_embedding, 8),)

        self.fc_type_1 = nn.Sequential(nn.Linear(dim_embedding, dim_embedding),
                                        nn.ReLU(inplace=True),)
        self.fc_type_2 = nn.Sequential(nn.Linear(dim_embedding, 9))

        self.fc_id_1 = nn.Sequential(nn.Linear(dim_embedding, dim_embedding),
                                        nn.ReLU(inplace=True),)
        self.fc_id_2 = nn.Sequential(nn.Linear(dim_embedding, 482))

        # self.fc_motion = nn.Sequential(nn.Linear(1024, 1024), 
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(1024, dim_embedding))

        # self.fc_fusion = nn.Linear(2*dim_embedding,dim_embedding)
        self.fc_fusion = nn.Linear(dim_embedding,dim_embedding)

        self.tau = nn.Parameter(torch.ones(1), requires_grad=True)

    @staticmethod
    def get_img_encoder():
        encoder = SwinTransformer(img_size=384,
                                patch_size=4,
                                in_chans=3,
                                num_classes=21841,
                                embed_dim=128,
                                depths=[2,2,18,2],
                                num_heads=[4,8,16,32],
                                window_size=12,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
        ckpt = torch.load('/home/zby/AICity2022Track2/pretrained/swin_base_patch4_window12_384_22k.pth')
        encoder.load_state_dict(ckpt['model'])
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

    def forward(self, x):
        """
        keys of x: crop, text_input_ids, text_attention_mask,
        """
        # ENCODERS
        features_obj,_ = self.encoder_img(x['crop'])
        features_obj = self.fc_obj(features_obj).squeeze()
        # outputs, features_motion = self.encoder_motion(x['motion_boxes'])
        # features_motion = features_motion[-1]
        # features_motion = self.fc_motion(features_motion)
        # features_fusion = self.fc_fusion(torch.cat((features_obj,features_motion),dim=1))
        features_fusion = self.fc_fusion(features_obj)

        # For main text data
        features_text = self.encoder_text.encode_text(x['text_input_ids']).to(torch.float)
        features_text_fusion = self.fc_text_fusion(features_text)
        features_text_obj = self.fc_text_obj(features_text_fusion)
        # features_text_motion = self.fc_text_motion(features_text_fusion)

        # For Visual features
        features_color = self.fc_color_1(features_obj)
        output_color = self.fc_color_2(features_color)
        features_type =  self.fc_type_1(features_obj)
        output_type = self.fc_type_2(features_type)
        features_id =  self.fc_id_1(features_obj)
        output_id = self.fc_id_2(features_id)

        # Normalization for main text
        features_obj_main, features_text_obj_main = map(lambda t: F.normalize(t, p=2, dim=-1), (features_obj, features_text_obj))
        # features_motion_main, features_text_motion_main = map(lambda t: F.normalize(t, p=2, dim=-1), (features_motion, features_text_motion))
        features_fusion_main, features_text_fusion_main = map(lambda t: F.normalize(t, p=2, dim=-1), (features_fusion, features_text_fusion))

        return [(features_obj_main, features_text_obj_main),
                # (features_motion_main, features_text_motion_main),
                (features_fusion_main,features_text_fusion_main),\
                ], output_color, output_type, output_id, self.tau,\
                [features_text]

