import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels=16):
        super(InceptionBlock1D, self).__init__()
        self.branch1 = nn.Sequential(
            nn.LazyConv1d(out_channels, kernel_size=2, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )
        self.branch2 = nn.Sequential(
            nn.LazyConv1d(out_channels, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )
        self.branch3 = nn.Sequential(
            nn.LazyConv1d(out_channels, kernel_size=8, stride=2, padding=4),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)


    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)

class ChronoNet(nn.Module):
    """
    ChronoNet: Inception-style Conv1D + GRUs with optional binary or multiclass output
    """

    def __init__(self, in_channels=32, conv_out_channels=32, gru_hidden_size=64, num_classes=4, is_binary=False):
        super(ChronoNet, self).__init__()
        self.is_binary = is_binary
        self.num_classes = num_classes

        self.conv_channels = conv_out_channels * 3

        self.incept1 = InceptionBlock1D(in_channels, out_channels=conv_out_channels)
        self.incept2 = InceptionBlock1D(self.conv_channels, out_channels=conv_out_channels)
        self.incept3 = InceptionBlock1D(self.conv_channels, out_channels=conv_out_channels)
        self.incept4 = InceptionBlock1D(self.conv_channels, out_channels=conv_out_channels)
        self.incept5 = InceptionBlock1D(self.conv_channels, out_channels=conv_out_channels)

        self.gru1 = nn.GRU(self.conv_channels, gru_hidden_size, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(gru_hidden_size, gru_hidden_size, batch_first=True, bidirectional=False)
        self.gru3 = nn.GRU(gru_hidden_size * 2, gru_hidden_size, batch_first=True, bidirectional=True)
        self.gru4 = nn.GRU(gru_hidden_size * 4, gru_hidden_size, batch_first=True, bidirectional=True)
        self.gru5 = nn.GRU(gru_hidden_size * 6, gru_hidden_size, batch_first=True, bidirectional=True)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(20, 1 if is_binary else num_classes)
        )

    def forward(self, x):
        x = self.incept1(x)
        # x = F.dropout(x, 0.1)
        x = self.incept2(x)
        x = self.incept3(x)
        x = self.incept4(x)
        x = self.incept5(x)
        # x = F.dropout(x, 0.1)

        x = x.permute(0, 2, 1)  # [B, T, C] for GRU

        out1, _ = self.gru1(x)
        out2, _ = self.gru2(out1)

        cat1 = torch.cat([out1, out2], dim=-1)
        # cat1 = F.dropout(cat1, 0.1)
        out3, _ = self.gru3(cat1)

        cat2 = torch.cat([out1, out2, out3], dim=-1)
        # cat2 = F.dropout(cat2, 0.1)
        out4, _ = self.gru4(cat2)
        
        cat3 = torch.cat([out1, out2, out3, out4], dim=-1)
        # cat2 = F.dropout(cat2, 0.1)
        out5, _ = self.gru5(cat3)
        # out4 = F.dropout(out4, 0.1)
        out5 = out5.permute(0, 2, 1)  # [B, C, T]
        out6 = self.global_pool(out5).squeeze(-1)  # [B, C]
        
        logits = self.fc(out6)
        # Ensure shape compatibility
        # if self.is_binary:
        #     return logits.squeeze(dim=1)  # shape: [B]
        # else:
        return logits  # shape: [B, num_classes]
    
    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'chrononet'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name