import torch


@torch.no_grad()
def encode_image(img,vae,args,ratio):
    zpad = args.zpad
    if zpad > 0:
        ppad = zpad*ratio
        img = torch.concat([img[:,:,:,-ppad:],img,img[:,:,:,:ppad]],dim=-1)

    x = vae.scale_(vae.encode(img).latent_dist.sample())

    if zpad > 0:
        x = x[:,:,:,zpad:-zpad]
    return x