import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class FrameTextRelevanceDetector(nn.Module):
    def __init__(self, d_model=64):
        super(FrameTextRelevanceDetector, self).__init__()
        
        self.text_global_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.similarity_net = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.attention_pool = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
    def forward(self, visual_frames, text_features):
        batch_size, num_frames, d_model = visual_frames.shape
        
        text_global, _ = self.attention_pool(
            text_features.mean(dim=1, keepdim=True),
            text_features,
            text_features
        )
        text_global = text_global.squeeze(1)
        
        text_global = self.text_global_encoder(text_global)
        
        text_expanded = text_global.unsqueeze(1).expand(-1, num_frames, -1)
        
        combined_features = torch.cat([visual_frames, text_expanded], dim=-1)
        
        relevance_scores = self.similarity_net(combined_features).squeeze(-1)
        
        return relevance_scores, text_global
class AdaptiveWeightGenerator(nn.Module):
    def __init__(self, d_model=64):
        super(AdaptiveWeightGenerator, self).__init__()
        
        self.weight_modulator = nn.Sequential(
            nn.Linear(d_model + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.min_weight = nn.Parameter(torch.tensor(0.5))
        self.max_weight = nn.Parameter(torch.tensor(5.0))
        
    def forward(self, visual_frames, relevance_scores, text_global):
        batch_size, num_frames, d_model = visual_frames.shape
        
        relevance_expanded = relevance_scores.unsqueeze(-1)
        combined = torch.cat([visual_frames, relevance_expanded], dim=-1)
        
        weight_factors = self.weight_modulator(combined).squeeze(-1)
        
        adaptive_weights = self.min_weight + (self.max_weight - self.min_weight) * weight_factors
        
        final_weights = adaptive_weights * (1 + relevance_scores)
        
        return final_weights
class SaliencyCrossAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=8, dropout=0.1):
        super(SaliencyCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.relevance_detector = FrameTextRelevanceDetector(d_model)
        self.adaptive_weight_generator = AdaptiveWeightGenerator(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None, saliency_weights=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        if saliency_weights is not None:
            weights_expanded = saliency_weights.unsqueeze(1).unsqueeze(-1)
            weights_expanded = weights_expanded.expand(-1, self.n_heads, -1, scores.size(-1))
            scores = scores * weights_expanded
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size, num_frames, _ = query.size()
        text_len = key.size(1)
        
        relevance_scores, text_global = self.relevance_detector(query, key)
        
        saliency_weights = self.adaptive_weight_generator(query, relevance_scores, text_global)
        
        Q = self.W_q(query).view(batch_size, num_frames, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, text_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, text_len, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask, saliency_weights
        )
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, num_frames, self.d_model
        )
        
        output = self.W_o(attention_output)
        
        enhanced_features = self.layer_norm(query + output)
        
        return enhanced_features