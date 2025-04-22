import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(Encoder, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)  # (2048 -> embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Set requires_grad
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = train_CNN

    def forward(self, images):
        # images: (batch_size, 3, 299, 299)
        features = self.inception(images)  # (batch_size, embed_size)
        return self.dropout(self.relu(features))  # (batch_size, embed_size)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # captions input -> embedded vectors
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # features: (batch_size, embed_size)
        # captions: (batch_size, caption_length)

        embeddings = self.dropout(self.embed(captions))  # (batch_size, caption_length, embed_size)

        features = features.unsqueeze(1)  # (batch_size, 1, embed_size)

        inputs = torch.cat((features, embeddings), dim=1)  # (batch_size, caption_length + 1, embed_size)

        hidden, _ = self.lstm(inputs)  # (batch_size, caption_length + 1, hidden_size)

        output = self.linear(hidden)  # (batch_size, caption_length + 1, vocab_size)

        return output

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = Encoder(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        # images: (batch_size, 3, 299, 299)
        # captions: (batch_size, caption_length)

        features = self.encoderCNN(images)  # (batch_size, embed_size)
        output = self.decoderRNN(features, captions)  # (batch_size, caption_length + 1, vocab_size)

        return output
    def caption_image(self,images,vocabulary,max_len = 50):
        result_captions = []
        with torch.no_grad():
            x = self.encoderCNN(images)
            states = None
            for i in range(max_len):
                hidden, states = self.decoderRNN(x, states)
                output = self.decoderRNN.linear(hidden.unsqueeze(0))
                prediction = output.argmax(1)
                result_captions.append(prediction.item())
                x = self.decoderRNN.embed(prediction).unsqueeze(0)
                if vocabulary.itos[prediction.item()] == "<END>":
                    break
        return [vocabulary.itos[x] for x in result_captions]

