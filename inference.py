import torch

def generate_caption(model, image, word2idx, idx2word, max_len=30):
    model.eval()
    with torch.no_grad():
        memory = model.encoder(image)

        caption = [word2idx["<start>"]]

        for _ in range(max_len):
            tokens = torch.tensor([caption]).to(image.device)
            logits = model.decoder(memory, tokens)
            next_word = logits[0, -1].argmax().item()

            if next_word == word2idx["<end>"]:
                break

            caption.append(next_word)

        return " ".join(idx2word[w] for w in caption[1:])
