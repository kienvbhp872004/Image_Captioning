import torchvision.transforms as t

def get_transforms():
    return t.Compose([
        t.Resize((224, 224)),
        t.ToTensor(),
        t.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
