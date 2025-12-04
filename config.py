class Config:
    # Model
    embed_size = 512
    num_heads = 8
    num_layers = 6

    vocab_path = "saved/vocab.pkl"

    # Paths
    image_dir = "data/images"
    caption_path = "data/captions.json"

    # Training
    batch_size = 32
    lr = 1e-4
    num_epochs = 20

    device = "cuda"
