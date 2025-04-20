import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN
def train():
    transform = transforms.Compose([
        transforms.Resize((356,356)),
        transforms.RandomCrop((299,299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    train_loader,dataset =get_loader(
        root_folder = r"D:\git\Image_Captioning\dataset\Images",
        annotation_file = r"D:\git\Image_Captioning\dataset\captions.txt",
        transform = transform,
        num_workers = 2,
    )
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    embed_size = 256
    hidden_size = 256
    num_layers = 1
    vocab_size = len(dataset.vocab)
    learning_rate = 1e-3
    num_epochs = 100
    writer = SummaryWriter("runs/flickr")
    step = 0

    model = CNNtoRNN(embed_size = embed_size, hidden_size = hidden_size,vocab_size = vocab_size,num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if load_model:
        step = (torch.load("my_checkpoint.pth.tar"),model,optimizer)
    model.train()
    for epoch in range(num_epochs):
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)
        for idx, (images, captions) in enumerate(train_loader):
            images, captions = images.to(device), captions.to(device)
            output = model(images,captions[:-1])
            loss = criterion(output.reshape(-1,output.shape[2]),captions.reshape(-1))
            writer.add_scalar("loss",loss.item(),step)
            step += 1
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
if __name__ == "__main__":
    train()





