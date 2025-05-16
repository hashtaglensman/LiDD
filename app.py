# app.py
import torch, torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
import dlib, cv2, random
from PIL import Image


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


transform_1 = T.Compose([T.ToTensor(), 
                                    CustomNormalize()])

class LightningDeepFakeDetection_BB(L.LightningModule):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.clf_layer = nn.Linear(1000,2)
        # self.lr = lr

    def forward(self, inputs):
        inputs = self.model(inputs)
        outputs = self.clf_layer(nn.Dropout(0.5)(nn.GELU()(inputs)))
        return outputs, inputs




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
model = LightningDeepFakeDetection_V2.load_from_checkpoint(path, map_location="cpu").eval()


featmodel = LightningDeepFakeDetection_BB()#.load_from_checkpoint('models_lightning/BB/DeepFake_BB-epoch=11-val_acc=0.98-val_loss=0.46_FakeAVCeleb-v1.ckpt')
featmodel.load_state_dict(torch.load('BB_mobilnet_weights_combined.pth', weights_only=True))
featmodel.eval();


# class ImageIn(BaseModel):
#     def __init__(self):
#         # Load dlib's face detector and landmark predictor
#         detector = dlib.get_frontal_face_detector()
#         predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


app = FastAPI()

# ---------- helper ----------
def sample_frames(video_path: str, n: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise RuntimeError("Empty / corrupt video.")
    n = min(n, total)
    idxs = random.sample(range(total), n)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # BGR → RGB
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

# ---------- endpoint ----------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    n: int = Query(10, ge=1, le=100, description="Number of frames to sample")
):
    if file.content_type not in {"video/mp4"}:
        raise HTTPException(400, "Upload an MP4 video.")
    # save to a temp file because OpenCV needs a path
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        pil_frames = sample_frames(tmp_path, n)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    if not pil_frames:
        raise HTTPException(400, "Could not decode any frames.")

    # preprocess + batch
    batch = torch.stack([featmodel(transform_(img)) for img in pil_frames])  # (B,1,28,28)


    with torch.no_grad():
        logits = model(transform_1(batch))                       # (B,10)
        preds = logits.argmax(1).tolist()           # list[int]

    # majority vote for overall prediction
    majority = collections.Counter(preds).most_common(1)[0][0]

    return {"frame_predictions": preds, "majority_digit": majority}
    
    
    
    # x = preprocess(img).unsqueeze(0)     # (1,1,28,28)
    # with torch.no_grad():
    #     pred = model(x).argmax(1).item()
    # return {"digit": pred}