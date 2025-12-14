class Config:
    # Model
    embed_size = 512
    num_heads = 8
    num_layers = 6

    vocab_path = "saved/vocab.pkl"

    # Paths
    image_dir = "data/Images"
    caption_path = "data/captions.txt"
    feature_dir = "data/features"
    # Training
    batch_size = 64
    lr = 1e-4
    num_epochs = 20

    device = "cuda"
