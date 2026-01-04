import torch
import torch.nn as nn
import torch.nn.functional as F
from model.trm import *
import pandas as pd
import json
from model.cross_verification_attention import CrossVerificationCoAttention
from model.guidance_attention import GuidanceAttention
from model.saliency_attention import SaliencyCrossAttention
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
class Adjudicator_Model(torch.nn.Module):
    def __init__(self,dataset):
        super(Adjudicator_Model,self).__init__()
        if dataset=='fakett':
            self.encoded_text_semantic_fea_dim=512 
        elif dataset=='fakesv':
            self.encoded_text_semantic_fea_dim=768 
        self.input_visual_frames=256
        self.mlp_text_emo = nn.Sequential(nn.Linear(768,64),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_text_semantic = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.1)) 
        self.mlp_visual_text = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_img = nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_audio = nn.Sequential(torch.nn.Linear(768, 64), torch.nn.ReLU(),nn.Dropout(0.1))
        self.mlp_audio_text = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_vggish = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1))
        self.cross_attention_audio_vggish = GuidanceAttention(d_model=64, n_heads=8, dropout=0.1)
        self.visual_cross_attention = GuidanceAttention(d_model=64, n_heads=8, dropout=0.1)
        
        self.mlp_analysis = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.1))
        
        self.analysis_cross_attention = SaliencyCrossAttention(d_model=64, n_heads=8, dropout=0.1)
        
        self.co_attention_tv = CrossVerificationCoAttention(d_k=64, d_v=64, n_heads=4, dropout=0.1, d_model=64,
                                        visual_len=self.input_visual_frames, sen_len=1024, fea_v=64, fea_s=64, pos=False)
        self.co_attention_audio_text = CrossVerificationCoAttention(d_k=64, d_v=64, n_heads=4, dropout=0.1, d_model=64,
                                                visual_len=512, sen_len=1024, fea_v=64, fea_s=64, pos=False)
        
        self.trm_emo=nn.TransformerEncoderLayer(d_model = 64, nhead = 2, batch_first = True)
        self.trm_semantic=nn.TransformerEncoderLayer(d_model = 64, nhead = 2, batch_first = True)
        self.trm_audio_text=nn.TransformerEncoderLayer(d_model = 64, nhead = 2, batch_first = True)
        
        self.content_classifier = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.1),nn.Linear(32,2))
        self.audio_text_classifier = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.1),nn.Linear(32,2))
        self.emo_classifier = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.1),nn.Linear(32,2))
        self.final_classifier = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.1),nn.Linear(32,2))
    def forward(self,**kwargs):
        all_combine_text_fea=kwargs['all_combine_text_fea']
        all_text_emo_fea=kwargs['all_text_emo_fea']
        all_visual_analysis_fea=kwargs['all_visual_analysis_fea']
        all_audio_analysis_fea=kwargs['all_audio_analysis_fea']
        all_raw_visual_frames=kwargs['all_raw_visual_frames']
        all_raw_audio_emo=kwargs['all_raw_audio_emo']
        all_review_result_fea=kwargs['all_review_result_fea']
        
        visual_text_proj = self.mlp_visual_text(all_visual_analysis_fea)
        visual_frames_proj = self.mlp_img(all_raw_visual_frames)
        
        enhanced_visual_text = visual_text_proj
        enhanced_visual_frames = visual_frames_proj
        
        enhanced_visual = self.visual_cross_attention(
            query=enhanced_visual_frames,
            key=enhanced_visual_text,
            value=enhanced_visual_text
        )
        
        analysis_proj = self.mlp_analysis(all_review_result_fea)
        
        final_enhanced_visual = self.analysis_cross_attention(
            query=enhanced_visual,
            key=analysis_proj,
            value=analysis_proj
        )
        
        semantic_proj = self.mlp_text_semantic(all_combine_text_fea)
        enhanced_semantic = semantic_proj
        
        content_v, content_t = self.co_attention_tv(
            v=final_enhanced_visual, 
            s=enhanced_semantic, 
            v_len=final_enhanced_visual.shape[1],
            s_len=enhanced_semantic.shape[1]
        )
        
        content_v = torch.mean(content_v, -2) 
        content_t = torch.mean(content_t, -2)
        fusion_semantic_fea = self.trm_semantic(torch.cat((content_t.unsqueeze(1), content_v.unsqueeze(1)), 1))
        fusion_semantic_fea = torch.mean(fusion_semantic_fea, 1)
        logits1_fusion = self.content_classifier(fusion_semantic_fea)
        output_branch1 = logits1_fusion
        
        audio_vggish_proj = self.mlp_vggish(kwargs['all_raw_audio_fea'])
        audio_text_proj = self.mlp_audio_text(all_audio_analysis_fea)
        enhanced_audio_vggish = self.cross_attention_audio_vggish(
            query=audio_vggish_proj,
            key=audio_text_proj,
            value=audio_text_proj
        )
        semantic_proj = self.mlp_text_semantic(all_combine_text_fea)
        content_audio, content_text = self.co_attention_audio_text(
            v=enhanced_audio_vggish,
            s=semantic_proj,
            v_len=enhanced_audio_vggish.shape[1],
            s_len=semantic_proj.shape[1]
        )
        content_audio = torch.mean(content_audio, -2)
        content_text = torch.mean(content_text, -2)
        fusion_audio_text = self.trm_audio_text(torch.cat((content_audio.unsqueeze(1), content_text.unsqueeze(1)), 1))
        fusion_audio_text = torch.mean(fusion_audio_text, 1)
        logits2_fusion = self.audio_text_classifier(fusion_audio_text)
        output_branch2 = logits2_fusion
        
        raw_t_fea_emo = self.mlp_text_emo(all_text_emo_fea).unsqueeze(1)
        raw_a_fea_emo = self.mlp_audio(all_raw_audio_emo).unsqueeze(1) 
        fusion_emo_fea = self.trm_emo(torch.cat((raw_t_fea_emo, raw_a_fea_emo), 1))
        fusion_emo_fea = torch.mean(fusion_emo_fea, 1)
        final_fused_feature = fusion_emo_fea
        logits_fused = self.final_classifier(final_fused_feature)
        output_branch3 =logits_fused
        final_output = output_branch1 + output_branch2 + output_branch3
        return final_output
class Adjudicator_MainModel(torch.nn.Module):
    def __init__(self,dataset):
        super(Adjudicator_MainModel,self).__init__()
        self.adjudicator_branch=Adjudicator_Model(dataset=dataset)
        
    def forward(self,  **kwargs):
        output_adjudicator=self.adjudicator_branch(**kwargs)
        return output_adjudicator