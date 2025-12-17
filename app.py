import torch
from PIL import Image
from models.caption_model import CaptionModel
from datasets.flickr8k_dataset import Flickr8kDataset
from utils.vocabulary import Vocabulary
from train.trainer import Trainer
from torch.utils.data import DataLoader
from config import Config
from main import create_dataloader
from datasets.transforms import get_transforms
import os

save_dir = "saved/model/Model_epoch_9.pth"
state = torch.load(save_dir,map_location=torch.device("cpu"))
image_test = Image.open("data/Images/72964268_d532bb8ec7.jpg")
transforms = get_transforms()
img = transforms(image_test).unsqueeze(0).to("cuda")
cfg = Config()
_, vocab = create_dataloader(cfg)
cfg.vocab_size = len(vocab)
model = CaptionModel(cfg.vocab_size,cfg.embed_size,cfg.num_heads,cfg.num_layers).to(cfg.device)
model.load_state_dict(state)
print(model.generate_caption(img,vocab,device="cuda"))
