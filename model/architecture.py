# Bu dosya, U-Net segmentasyon mimarisinin PyTorch ile tanımlanmasını içerir.
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """
    İki ardışık Conv2d + BatchNorm + ReLU bloğundan oluşan temel yapı taşı.
    Bu blok, hem encoder hem decoder tarafında tekrar tekrar kullanılır.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Girdiyi ardışık iki konvolüsyon bloğundan geçirir.
        """
        return self.conv(x)

class UNET(nn.Module):
    """
    U-Net segmentasyon mimarisi.
    Encoder (downsampling), bottleneck ve decoder (upsampling) bloklarından oluşur.
    Skip connection'lar ile detay kaybı önlenir.
    """
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: Özellik haritalarını küçültür ve derinleştirir
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder: Özellik haritalarını büyütür ve detayları geri kazandırır
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottleneck: En derin katman
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Girdi görüntüsünü U-Net mimarisi üzerinden geçirir ve çıktı maskesini döndürür.
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            # Boyut uyuşmazlığı varsa upsample edilen çıktıyı skip connection ile aynı boyuta getir
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    """
    U-Net mimarisinin giriş ve çıkış boyutlarını test etmek için kullanılabilir.
    """
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()