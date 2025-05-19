# optimized_inference_pipeline.py
"""Optimized CPU‑only inference pipeline for deep‑fake detection.

Key optimizations
-----------------
1. **Single video pass** – sample all required frames in one sequential read to avoid costly random seeks.
2. **Vectorised transforms** – apply image transforms and feature extraction in a batched fashion.
3. **Single model call** – run the classifier once on the full set of frames, then slice logits for each subset.
4. **Lightweight majority vote** – use `numpy` for fast computation.

Assumptions
-----------
* `transform_`, `featmodel`, `model`, and `CustomNormalize` are already defined.
* The classifier `model` expects a 2‑D tensor `(B, F)` and outputs `(None, logits, None)`.
* All computation is on CPU; set the default tensor device accordingly.
"""

from __future__ import annotations
from typing import List, Sequence
import numpy as np
import torch
from PIL import Image
import torch, torch.nn as nn
import cv2, random
import torchvision.transforms as T
import lightning as L
import collections


__all__ = [
    "sample_frames",
    "frame_indices",
    "pred_cred",
]
device = torch.device("cpu")

class LightningDeepFakeDetection_BB(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.clf_layer = nn.Linear(1000,2)

    def forward(self, inputs):
        inputs = self.model(inputs)
        outputs = self.clf_layer(nn.Dropout(0.5)(nn.GELU()(inputs)))
        return inputs
    
class CustomNormalize(nn.Module):
    def __init__(self):
        super(CustomNormalize, self).__init__()

    def forward(self, x):
        # Calculate max value and standard deviation
        max_val = torch.max(x)
        std_dev = torch.std(x)
        
        # Avoid division by zero
        eps = 1e-8
        
        # Normalize the vector
        normalized = (x - torch.mean(x)) / (std_dev + eps)
        normalized = normalized / (max_val + eps)
        
        return normalized
    

transform_ = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_1 = T.Compose([CustomNormalize()])

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads,batch_first=True)
        self.position_embedding = nn.Parameter(torch.randn(1, 1, input_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Add class token
        batch_size = x.size(0)
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)

        # Add position embedding
        x = x + self.position_embedding

        # Apply attention
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        return x[:, 0]  # Return class token
    

class DF_Detection_V2(nn.Module):
    def __init__(self):
        super(DF_Detection_V2, self).__init__()
        self.project_layer = nn.Linear(1000, 512) 
        self.generator_model = self.Gen_model
        self.classifier_model = self.Classifier_model

        
        self.genlayer1 = nn.Linear(512, 256)
        self.genlayer2 =  nn.Linear(256, 512)
        self.genlayer3 = nn.Linear(512,512)

        self.clflayer1 = nn.Linear(512,256)
        self.attention_layer = AttentionLayer(input_dim=128)
        self.clflayer3 = nn.Linear(128, 2) #3


        self.batchnorm = nn.BatchNorm1d(512)
        self.batchnorm_ = nn.BatchNorm1d(256)
        self.layernorm = nn.LayerNorm(256)
    
    
    def Gen_model(self, x):
        x_1 = nn.GELU()(self.genlayer1(x)) #x + feat_
        x_2 = nn.Dropout(0.25)(nn.GELU()(self.batchnorm(self.genlayer2(x_1))))
        feat = self.genlayer3(x_2 + x)
        return feat
    
    def Classifier_model(self, x_, t):
        x_2 = (self.batchnorm_(self.clflayer1(x_ + nn.Dropout(0.5)(t))))
        x_2 = self.layernorm(nn.Dropout(0.3)(nn.ELU()(x_2)))
        x_3 = self.attention_layer(x_2.view(-1).reshape([len(x_),2,128]))
        out = self.clflayer3(x_3 + x_2[:,:128] + x_2[:,128:])#out = self.clflayer3(x_3)
        return  out
      
    def forward(self, x):
        proj_x = (self.project_layer(x))
        gen_out= self.generator_model(proj_x)
        clf_out = self.classifier_model(proj_x, gen_out)
        return proj_x, clf_out, gen_out


class LightningDeepFakeDetection_V2(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DF_Detection_V2().eval()
        # self.save_hyperparameters()

    def forward(self, inputs):
        return self.model(inputs)
    
path ='DeepFake_Detection_V2-epoch=21-val_acc=0.88-val_loss=2.22_bb_finetuned_w_CMDFD.ckpt' 
model = LightningDeepFakeDetection_V2.load_from_checkpoint(path, map_location=device).eval()

featmodel = LightningDeepFakeDetection_BB()#.load_from_checkpoint('models_lightning/BB/DeepFake_BB-epoch=11-val_acc=0.98-val_loss=0.46_FakeAVCeleb-v1.ckpt')
featmodel.load_state_dict(torch.load('BB_mobilnet_weights_combined.pth', weights_only=False, map_location=device))
featmodel.eval();


def frame_indices(total: int, k: int) -> List[int]:
    """Return *k* unique random indices in the range ``[0, total)``.

    The output is **sorted** so that frames can be read sequentially.
    """
    if total == 0:
        raise RuntimeError("Empty / corrupt video.")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, total)
    return sorted(random.sample(range(total), k))

def sample_frames(video_path: str, idxs: Sequence[int]) -> List[Image.Image]:
    """Read the frames at *idxs* (must be sorted) and return them as ``PIL.Image``s.

    Frames are fetched **sequentially** for maximal throughput on CPU‑only setups.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: %s" % video_path)

    frames: List[Image.Image] = []
    next_idx_iter = iter(idxs)
    next_target = next(next_idx_iter, None)
    cur = 0

    while next_target is not None:
        ok, frame = cap.read()
        if not ok:
            break  # reached EOF prematurely
        if cur == next_target:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            next_target = next(next_idx_iter, None)
        cur += 1

    cap.release()

    if len(frames) != len(idxs):
        raise RuntimeError("Failed to retrieve all requested frames.")
    return frames

def _batched_features(frames: Sequence[Image.Image]) -> torch.Tensor:
    """Apply preprocessing + feature network in one go."""
    batch = torch.stack([transform_(img) for img in frames])  # (B, C, H, W)
    with torch.inference_mode():
        feats = featmodel(batch)  # (B, F)
    return CustomNormalize()(feats)

def pred_cred(video_path: str, subset_sizes: Sequence[int] = (10, 15, 20)) -> int:
    """Return the majority‑voted prediction for *video_path*.

    The video is scanned **once**.  The largest subset size dictates the number of
    sampled frames; smaller subsets reuse the prefix of that sample for voting.
    """
    # open once just to know frame count (cheap)
    total = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    max_k = max(subset_sizes)
    idxs = frame_indices(total, max_k)

    # read frames & extract features
    frames = sample_frames(video_path, idxs)
    feats = _batched_features(frames)  # (max_k, F)

    # classifier – single forward pass
    with torch.inference_mode():
        _, logits, _ = model(feats)
    preds = logits.argmax(1).cpu().numpy()  # (max_k,)

    # majority vote per subset
    majorities = [collections.Counter(preds[:k]).most_common(1)[0][0] for k in subset_sizes]
    # final decision – majority of majorities
    final_pred = int(round(np.mean(majorities)))
    return final_pred
