import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import build_model, load_clip, tokenize


class StreamFCN(nn.Module):
    def __init__(self, output_channels, device='mps'):
        super(StreamFCN, self).__init__()
        self.up_factor = 2
        self.input_dim = 2048
        self.device = device
        self.output_channels = output_channels
        self.clip_rn50, _ = load_clip("RN50", device="cpu", jit=False)

        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.lang_down1 = nn.Linear(1024, 1024)
        self.lang_down2 = nn.Linear(1024, 512)
        self.lang_down3 = nn.Linear(512, 256)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, self.output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    def forward_heatmap(self, x, l):
        """Returns the raw (B, output_channels, H, W) spatial activation map."""
        x, im = self.encode_image(x)
        l_enc, l_emb, l_mask = self.encode_text(l)

        assert x.shape[1] == self.input_dim
        x = self.conv1(x)
        l_feat = self.lang_down1(l_enc)
        x = self.layer1(x * l_feat.reshape(-1, 1024, 1, 1))
        l_feat = self.lang_down2(l_feat)
        x = self.layer2(x * l_feat.reshape(-1, 512, 1, 1))
        l_feat = self.lang_down3(l_feat)
        x = self.layer3(x * l_feat.reshape(-1, 256, 1, 1))
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    @staticmethod
    def spatial_soft_argmax(heatmap):
        """Differentiable per-channel 2D argmax: softmax over pixels, then
        expected (x, y) under that distribution, normalized to [-0.5, 0.5)
        to match the label convention in generate_data.py."""
        B, C, H, W = heatmap.shape
        prob = F.softmax(heatmap.reshape(B, C, H * W), dim=-1).reshape(B, C, H, W)
        xs = (torch.arange(W, device=heatmap.device, dtype=heatmap.dtype) + 0.5) / W - 0.5
        ys = (torch.arange(H, device=heatmap.device, dtype=heatmap.dtype) + 0.5) / H - 0.5
        expected_x = torch.einsum('bchw,w->bc', prob, xs)
        expected_y = torch.einsum('bchw,h->bc', prob, ys)
        return expected_x, expected_y

    def forward(self, x, l):
        heatmap = self.forward_heatmap(x, l)
        expected_x, expected_y = self.spatial_soft_argmax(heatmap)
        # interleave per-channel (x, y): channel 0 = pick, channel 1 = place
        # -> [pick_x, pick_y, place_x, place_y], matching generate_data.py labels
        return torch.stack([expected_x, expected_y], dim=-1).flatten(1)

    def encode_image(self, img):
        with torch.no_grad():
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
        return img_encoding, img_im

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

class PickAndPlace(nn.Module):
    def __init__(self, crop_size=64, num_dense=10):
        super(PickAndPlace, self).__init__()
        self.crop_size = crop_size
        self.num_dense = num_dense
        self.pick = StreamFCN(output_channels=1)
        self.place_key = StreamFCN(output_channels=num_dense)
        self.place_query = StreamFCN(output_channels=1)
        self.num_rotations = 36
    
    def correlate(self, in0, in1):
        output = F.conv2d(in1, in0)
        output = F.interpolate(output, size=(in1.shape[-2], in1.shape[-1]), mode='bilinear')
        return output
    
    def forward(self, x, l):
        q_pick = self.pick.forward_heatmap(x, l[0])
        _, _, H, W = q_pick.shape
        q_pick = F.softmax(q_pick.reshape(H * W), dim=0)
        t_pick = torch.argmax(q_pick)
        t_pick = torch.tensor([[t_pick // H, t_pick % H]])

        q_place = self.place_query.forward_heatmap(x, l[0])
        q_place = F.softmax(q_place.reshape(H * W), dim=0)
        t_place = torch.argmax(q_place)
        t_place = torch.tensor([[t_place // H, t_place % H]])

        # y = x[:, :, t_pick[0] - self.crop_size:t_pick[0] + self.crop_size, t_pick[1] - self.crop_size:t_pick[1] + self.crop_size]
        # query = self.place_query(y, l)
        # key = self.place_key(x, l)
        # q_place = self.correlate(query, key)
        # t_place = torch.argmax(q_place.reshape(H * W))
        # t_place = torch.tensor([t_place // H, t_place % H])
        return t_pick, t_place
    
if __name__ == '__main__':
    model = StreamFCN(1)
    model.to('cpu')
    model.eval()
    x = torch.randn((1, 3, 224, 224))
    l = 'pick the red circle and keep it over the blue square'
    x = model(x, l)
    print(x.shape)