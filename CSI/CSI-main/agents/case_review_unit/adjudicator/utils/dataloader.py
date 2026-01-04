import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video = torch.cat((video, torch.zeros([seq_len-ori_len, video.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)
class Adjudicator_Dataset(Dataset):
    def __init__(self, vid_path, dataset):
        self.dataset = dataset
        if dataset == 'fakesv':
            self.data_all = pd.read_json('./fea/fakesv/shots.json', orient='records', dtype=False, lines=True)
            self.vid = []
            with open(vid_path, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
            self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
            self.data.reset_index(inplace=True)
            self.raw_text_semantic_fea_path = './fea/fakesv/preprocess_text/path/to/file'
            with open(self.raw_text_semantic_fea_path, 'rb') as f:
                self.raw_text_semantic_fea = torch.load(f)
            self.raw_visual_fea_path = './fea/fakesv/path/to/file'
            self.raw_audio_fea_path = './fea/fakesv/path/to/file'
            with open(self.raw_audio_fea_path, 'rb') as f:
                self.raw_audio_fea = torch.load(f)
            self.text_emo_fea_path = './fea/fakesv/preprocess_text/path/to/file'
            with open(self.text_emo_fea_path, 'rb') as f:
                self.raw_text_emo_fea = torch.load(f)
            self.audio_emo_fea_path = './fea/fakesv/path/to/file'
            self.visual_analysis_fea_path = './fea/fakesv/path/to/file'
            with open(self.visual_analysis_fea_path, 'rb') as f:
                self.visual_analysis_fea = torch.load(f)
            self.audio_analysis_fea_path = './fea/fakesv/path/to/file'
            with open(self.audio_analysis_fea_path, 'rb') as f:
                self.audio_analysis_fea = torch.load(f)
            self.structed_title_fea_path = './fea/fakesv/path/to/file'
            with open(self.structed_title_fea_path, 'rb') as f:
                self.structed_title_fea = torch.load(f)
            self.review_result_fea_path = './fea/fakesv/path/to/file'
            with open(self.review_result_fea_path, 'rb') as f:
                self.review_result_fea = torch.load(f)
        elif dataset == 'fakett':
            self.data_all = pd.read_json('./fea/fakett/shots.json', orient='records', lines=True,
                                         dtype={'video_id': str})
            self.vid = []
            with open(vid_path, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
            self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
            self.data.reset_index(inplace=True)
            self.raw_text_semantic_fea_path = './fea/fakett/preprocess_text/path/to/file'
            with open(self.raw_text_semantic_fea_path, 'rb') as f:
                self.raw_text_semantic_fea = torch.load(f)
            
            self.raw_visual_fea_path = './fea/fakett/path/to/file'
            
            self.raw_audio_fea_path = './fea/fakett/path/to/file'
            with open(self.raw_audio_fea_path, 'rb') as f:
                self.raw_audio_fea = torch.load(f)
            self.text_emo_fea_path = './fea/fakett/preprocess_text/path/to/file'
            with open(self.text_emo_fea_path, 'rb') as f:
                self.raw_text_emo_fea = torch.load(f)
            self.audio_emo_fea_path = './fea/fakett/path/to/file'
            self.visual_analysis_fea_path = './fea/fakett/path/to/file'
            with open(self.visual_analysis_fea_path, 'rb') as f:
                self.visual_analysis_fea = torch.load(f)
            self.audio_analysis_fea_path = './fea/fakett/path/to/file'
            with open(self.audio_analysis_fea_path, 'rb') as f:
                self.audio_analysis_fea = torch.load(f)
            self.structed_title_fea_path = './fea/fakett/path/to/file'
            with open(self.structed_title_fea_path, 'rb') as f:
                self.structed_title_fea = torch.load(f)
            self.review_result_fea_path = './fea/fakett/path/to/file'
            with open(self.review_result_fea_path, 'rb') as f:
                self.review_result_fea = torch.load(f)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)
        raw_text_semantic_fea = self.raw_text_semantic_fea['last_hidden_state'][vid]
        raw_audio_fea = self.raw_audio_fea[vid]
        v_fea_path = os.path.join(self.raw_visual_fea_path, vid + '.pkl')
        raw_visual_frames = torch.tensor(torch.load(open(v_fea_path, 'rb')))
        raw_text_emo_fea = self.raw_text_emo_fea['pooler_output'][vid]
        a_e_fea_path = os.path.join(self.audio_emo_fea_path, vid + '.pkl')
        raw_audio_emo = torch.load(open(a_e_fea_path, 'rb'))
        structed_title_fea=self.structed_title_fea[vid]
        review_result_fea=self.review_result_fea[vid]
        visual_analysis_fea = self.visual_analysis_fea[vid]
        audio_analysis_fea = self.audio_analysis_fea[vid]
        return {
            'vid': vid,
            'label': label,
            'raw_text_semantic_fea': raw_text_semantic_fea,
            'raw_audio_fea': raw_audio_fea,
            'raw_visual_frames': raw_visual_frames,
            'raw_text_emo_fea': raw_text_emo_fea,
            'raw_audio_emo': raw_audio_emo,
            'audio_analysis_fea':audio_analysis_fea,
            'visual_analysis_fea':visual_analysis_fea,
            'structed_title_fea':structed_title_fea,
            'review_result_fea':review_result_fea
        }
    def collate_fn_Adjudicator(batch):
        vid = [item['vid'] for item in batch]
        label = torch.stack([item['label'] for item in batch])
        all_text_emo_fea = torch.stack([item['raw_text_emo_fea'] for item in batch])
        all_review_result_fea = torch.stack([item['review_result_fea'] for item in batch])
        all_visual_analysis_fea = torch.stack([item['visual_analysis_fea'] for item in batch])
        all_audio_analysis_fea = torch.stack([item['audio_analysis_fea'] for item in batch])
        all_raw_audio_fea = [item['raw_audio_fea'] for item in batch]
        processed_audio_vggish = []
        for x in all_raw_audio_fea:
            if x.size(0) < 512:
                zeros = torch.zeros((512 - x.size(0), x.size(1)), dtype=torch.float)
                processed_audio_vggish.append(torch.cat([x, zeros], dim=0))
            else:
                processed_audio_vggish.append(x[:512])
        all_raw_audio_fea = torch.stack(processed_audio_vggish)
        raw_visual_frames = [item['raw_visual_frames'] for item in batch]
        max_frames = 256
        all_raw_visual_frames, visual_attention_masks = pad_frame_sequence(max_frames, raw_visual_frames)
        raw_audio_emo = [item['raw_audio_emo'] for item in batch]
        all_raw_audio_emo = torch.cat(raw_audio_emo, dim=0)
        all_text_semantic_fea = [item['raw_text_semantic_fea'] for item in batch]
        all_structed_title_fea = [item['structed_title_fea'] for item in batch]
        all_text_semantic_fea = [
            x if x.shape[0] == 512 else torch.cat((x, torch.zeros([512 - x.shape[0], x.shape[1]], dtype=torch.float)),
                                                dim=0)
            for x in all_text_semantic_fea
        ]
        all_text_semantic_fea = torch.stack(all_text_semantic_fea)
        all_structed_title_fea = [
            x if x.shape[0] == 512 else torch.cat((x, torch.zeros([512 - x.shape[0], x.shape[1]], dtype=torch.float)),
                                                dim=0)
            for x in all_structed_title_fea
        ]
        all_structed_title_fea = torch.stack(all_structed_title_fea)
        all_combine_text_fea = torch.cat((all_structed_title_fea, all_text_semantic_fea), dim=1)
        return {
            'vid': vid,
            'label': label,
            'all_combine_text_fea': all_combine_text_fea,
            'all_raw_audio_fea': all_raw_audio_fea,
            'all_raw_visual_frames': all_raw_visual_frames,
            'all_text_emo_fea': all_text_emo_fea,
            'all_raw_audio_emo': all_raw_audio_emo,
            'all_visual_analysis_fea': all_visual_analysis_fea,
            'all_audio_analysis_fea': all_audio_analysis_fea,
            'all_review_result_fea': all_review_result_fea
        }