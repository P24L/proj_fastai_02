import requests
from PIL import Image
from pathlib import Path

from fastai.vision.all import *
from fastai.data.all import *

model_dropbox_url = 'https://dl.dropboxusercontent.com/s/a60n15qujnjbon5/export.pkl?dl=0'


bear_types = 'grizzly','black','teddy'
path = Path('bears')

def bear_type(x): 
    return x.parent.name

model_dest = Path('model_dropbox.pkl')
download_url(model_dropbox_url,model_dest)
learn_inf = load_learner(model_dest)

img = Path('grizzly.jpg')

pred,pred_idx,probs = learn_inf.predict(img)

print(f'Prediction: {pred}; Probability: {probs[pred_idx]*100:.02f}%')

