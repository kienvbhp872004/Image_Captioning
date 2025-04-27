import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        # Loại bỏ fully connected layer cuối
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Đóng băng trọng số ResNet
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Linear layer để chuyển đặc trưng sang kích thước embed_size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.1)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # captions input -> embedded vectors
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions,lengths):
        # features: (batch_size, embed_size)
        # captions: (batch_size, caption_length)
        lengths = torch.tensor([lengths], dtype=torch.int64)
        lengths =  lengths.view(-1,1)
        embeddings = self.dropout(self.embed(captions))  # (batch_size, caption_length, embed_size)
        features = features.unsqueeze(1)  # (batch_size, 1, embed_size)
        inputs = torch.cat((features, embeddings), dim=1)  # (batch_size, caption_length + 1, embed_size)
        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = self.lstm(packed)  # (batch_size, caption_length + 1, hidden_size)
        output = self.linear(hidden[0])  # (batch_size, caption_length + 1, vocab_size)

        return output
    def sample(self, features, lengths):
        sample_ids = []
        input = features.unsqueeze(1) ## (batch_size,1,embed_size)
        states = None

        for _ in range(lengths):
            hidden, state = self.lstm(input,states)
            output = self.linear(hidden[0])
            _,predicted = output.max(1)
            sample_ids.append(predicted.item())
            input = self.embed(predicted).unsqueeze(1)
        return sample_ids


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    def forward(self, images, captions):
        # images: (batch_size, 3, 299, 299)
        # captions: (batch_size, caption_length)

        features = self.encoderCNN(images)  # (batch_size, embed_size)
        output = self.decoderRNN(features, captions,lengths = 25)  # (batch_size, caption_length + 1, vocab_size)
        return output
    def caption_image(self,images,vocabulary,max_len = 25):
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

