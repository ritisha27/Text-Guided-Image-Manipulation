from tqdm import tqdm
import cv2
import numpy as np
import torch
import clip
from models.stylegan_generator import StyleGANGenerator
from models.stylegan_encoder import StyleGANEncoder
from models.perceptual_model import PerceptualModel

__all__ = ['StyleGANInverter']

def _get_tensor_value(tensor):
  return tensor.cpu().detach().numpy()

class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cpu")
        self.upsample = torch.nn.Upsample(scale_factor=28)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        image = image.cpu()
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity
    

class StyleGANInverter(object):
    def __init__(self, model_name, mode='man', learning_rate=1e-2, iteration=100, reconstruction_loss_weight=1.0,
               perceptual_loss_weight=5e-5, regularization_loss_weight=2.0, clip_loss_weight=None, description=None,
               logger=None):
        self.text_inputs = torch.cat([clip.tokenize(description)])
        self.clip_loss = CLIPLoss()
        self.mode = mode
        self.logger = logger
        self.model_name = model_name
        self.gan_type = 'stylegan'
        self.G = StyleGANGenerator(self.model_name, self.logger)
        self.E = StyleGANEncoder(self.model_name, self.logger)
        self.F = PerceptualModel(min_val=self.G.min_val, max_val=self.G.max_val)
        self.encode_dim = [self.G.num_layers, self.G.w_space_dim]
        self.run_device = self.G.run_device
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.loss_pix_weight = reconstruction_loss_weight
        self.loss_feat_weight = perceptual_loss_weight
        self.loss_reg_weight = regularization_loss_weight
        self.loss_clip_weight = clip_loss_weight


    def preprocess(self, image):
        if image.shape[2] == 1 and self.G.image_channels == 3:
            image = np.tile(image, (1, 1, 3))
        if self.G.image_channels == 3 and self.G.channel_order == 'BGR':
          image = image[:, :, ::-1]
        if image.shape[1:3] != [self.G.resolution, self.G.resolution]:
            image = cv2.resize(image, (self.G.resolution, self.G.resolution))
        image = image.astype(np.float32)
        image = image / 255.0 * (self.G.max_val - self.G.min_val) + self.G.min_val
        image = image.astype(np.float32).transpose(2, 0, 1)

        return image
    
    def get_init_code(self, image):    
        x = image[np.newaxis]
        x = self.G.to_tensor(x.astype(np.float32))
        z = _get_tensor_value(self.E.net(x).view(1, *self.encode_dim))
        return z.astype(np.float32)
    
    def invert(self, image, num_viz=0):
        image = self.preprocess(image)
        if self.mode == 'gen':
            init_z = self.G.sample(1, latent_space_type='wp',
                                z_space_dim=512, num_layers=14)
            init_z = self.G.preprocess(init_z, latent_space_type='wp')
            z = torch.Tensor(init_z).to(self.run_device)
            z.requires_grad = True
            x = self.G._synthesize(init_z, latent_space_type='wp')['image']
            x = torch.Tensor(x).to(self.run_device)
        else:
            x = image[np.newaxis]
            x = self.G.to_tensor(x.astype(np.float32))
            x.requires_grad = False
            init_z = self.get_init_code(image)
            z = torch.Tensor(init_z).to(self.run_device)            
            z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=self.learning_rate)

        viz_results = []
        viz_results.append(self.G.postprocess(_get_tensor_value(x))[0])
        x_init_inv = self.G.net.synthesis(z)
        viz_results.append(self.G.postprocess(_get_tensor_value(x_init_inv))[0])
        pbar = tqdm(range(1, self.iteration + 1), leave=True)
        for step in pbar:
            loss = 0.0

            x_rec = self.G.net.synthesis(z)
            loss_pix = torch.mean((x - x_rec) ** 2)
            loss = loss + loss_pix * self.loss_pix_weight
            log_message = f'loss_pix: {_get_tensor_value(loss_pix):.3f}'

            if self.loss_feat_weight:
                x_feat = self.F.net(x)
                x_rec_feat = self.F.net(x_rec)
                loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
                loss = loss + loss_feat * self.loss_feat_weight
                log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

            if self.loss_reg_weight:
                z_rec = self.E.net(x_rec).view(1, *self.encode_dim)
                loss_reg = torch.mean((z - z_rec) ** 2)
                loss = loss + loss_reg * self.loss_reg_weight
                log_message += f', loss_reg: {_get_tensor_value(loss_reg):.3f}'

            if self.loss_clip_weight:
                loss_clip = self.clip_loss(x_rec, self.text_inputs)
                loss = loss + loss_clip[0][0] * self.loss_clip_weight
                log_message += f', loss_clip: {_get_tensor_value(loss_clip[0][0]):.3f}'

            log_message += f', loss: {_get_tensor_value(loss):.3f}'
            pbar.set_description_str(log_message)
            if self.logger:
                self.logger.debug(f'Step: {step:05d}, '
                                f'lr: {self.learning_rate:.2e}, '
                                f'{log_message}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if num_viz > 0 and step % (self.iteration // num_viz) == 0:
                viz_results.append(self.G.postprocess(_get_tensor_value(x_rec))[0])

        return _get_tensor_value(z), viz_results

  
    