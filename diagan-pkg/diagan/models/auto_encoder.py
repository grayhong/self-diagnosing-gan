import torch.nn as nn

class CAE32(nn.Module):
    def __init__(self, in_channels = 3, rep_dim = 256):
        super().__init__()
        nf = 64
        self.nf = nf

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf)
        self.enc_act1 = nn.ReLU(inplace=True)

        self.enc_conv2 = nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf * 2)
        self.enc_act2 = nn.ReLU(inplace=True)

        self.enc_conv3 = nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf * 4)
        self.enc_act3 = nn.ReLU(inplace=True)

        self.enc_fc = nn.Linear(nf * 4 * 4 * 4, rep_dim)
        self.rep_act = nn.Tanh()

        # Decoder
        self.dec_fc = nn.Linear(rep_dim, nf * 4 * 4 * 4)
        self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 4 * 4 *4)
        self.dec_act0 = nn.ReLU(inplace=True)

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf * 2)
        self.dec_act1 = nn.ReLU(inplace=True)

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf)
        self.dec_act2 = nn.ReLU(inplace=True)

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_act = nn.Tanh()

    def encode(self, x):
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return rep

    def decode(self, rep):
        x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        x = x.view(-1, self.nf * 4, 4, 4)
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))

class CAE64(nn.Module):
    def __init__(self, in_channels = 3, rep_dim = 256):
        super().__init__()
        nf = 64
        self.nf = nf

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf)
        self.enc_act1 = nn.ReLU(inplace=True)

        self.enc_conv2 = nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf * 2)
        self.enc_act2 = nn.ReLU(inplace=True)

        self.enc_conv3 = nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf * 4)
        self.enc_act3 = nn.ReLU(inplace=True)

        self.enc_conv4 = nn.Conv2d(in_channels=nf * 4, out_channels=nf * 8, kernel_size=3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(num_features=nf * 8)
        self.enc_act4 = nn.ReLU(inplace=True)

        self.enc_fc = nn.Linear(nf * 8 * 4 * 4, rep_dim)
        self.rep_act = nn.Tanh()
        
        # Decoder
        self.dec_fc = nn.Linear(rep_dim, nf * 8 * 4 * 4)
        self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 8 * 4 * 4)
        self.dec_act0 = nn.ReLU(inplace=True)
        
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf * 8, out_channels=nf * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf * 4)
        self.dec_act1 = nn.ReLU(inplace=True)

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf * 2)
        self.dec_act2 = nn.ReLU(inplace=True)

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn3 = nn.BatchNorm2d(num_features=nf)
        self.dec_act3 = nn.ReLU(inplace=True)

        self.dec_conv4 = nn.ConvTranspose2d(in_channels=nf, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_act = nn.Tanh()

    def encode(self, x):
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_act4(self.enc_bn4(self.enc_conv4(x)))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return rep

    def decode(self, rep):
        x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        x = x.view(-1, self.nf * 8, 4, 4)
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_act3(self.dec_bn3(self.dec_conv3(x)))
        x = self.output_act(self.dec_conv4(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))

class CAE64(nn.Module):
    def __init__(self, in_channels = 3, rep_dim = 256):
        super().__init__()
        nf = 64
        self.nf = nf

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf)
        self.enc_act1 = nn.ReLU(inplace=True)

        self.enc_conv2 = nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf * 2)
        self.enc_act2 = nn.ReLU(inplace=True)

        self.enc_conv3 = nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf * 4)
        self.enc_act3 = nn.ReLU(inplace=True)

        self.enc_conv4 = nn.Conv2d(in_channels=nf * 4, out_channels=nf * 8, kernel_size=3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(num_features=nf * 8)
        self.enc_act4 = nn.ReLU(inplace=True)

        self.enc_fc = nn.Linear(nf * 8 * 4 * 4, rep_dim)
        self.rep_act = nn.Tanh()
        
        # Decoder
        self.dec_fc = nn.Linear(rep_dim, nf * 8 * 4 * 4)
        self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 8 * 4 * 4)
        self.dec_act0 = nn.ReLU(inplace=True)
        
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf * 8, out_channels=nf * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf * 4)
        self.dec_act1 = nn.ReLU(inplace=True)

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf * 2)
        self.dec_act2 = nn.ReLU(inplace=True)

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn3 = nn.BatchNorm2d(num_features=nf)
        self.dec_act3 = nn.ReLU(inplace=True)

        self.dec_conv4 = nn.ConvTranspose2d(in_channels=nf, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_act = nn.Tanh()

    def encode(self, x):
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_act4(self.enc_bn4(self.enc_conv4(x)))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return rep

    def decode(self, rep):
        x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        x = x.view(-1, self.nf * 8, 4, 4)
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_act3(self.dec_bn3(self.dec_conv3(x)))
        x = self.output_act(self.dec_conv4(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


DATASET_DICT = {
    'celeba': CAE64,
    'cifar10': CAE32,
    'color_mnist': CAE32,
    'multi_color_mnist':CAE32,
    'mnist_fmnist': CAE32,
}


def get_ae_model(dataset_name):
    nc = 3
    if dataset_name == 'mnist_fmnist':
        nc = 1
    ae = DATASET_DICT[dataset_name](in_channels=nc)
    return ae