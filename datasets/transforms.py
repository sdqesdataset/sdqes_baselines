from torchvision import transforms
from torchvision.transforms._transforms_video import RandomCropVideo, RandomResizedCropVideo,CenterCropVideo, NormalizeVideo,ToTensorVideo,RandomHorizontalFlipVideo
from PIL import Image

def init_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ])
    }
    return tsfm_dict

def init_video_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),

                        ## CLIP params
                        norm_mean=(0.48145466, 0.4578275, 0.40821073), 
                        norm_std=(0.26862954, 0.26130258, 0.27577711)):

                        # # EGOVLP params
                        # norm_mean=(0.485, 0.456, 0.406),
                        # norm_std=(0.229, 0.224, 0.225)):
    print('Video Transform is used!')
    normalize = NormalizeVideo(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'train': transforms.Compose([
            RandomResizedCropVideo(input_res, scale=randcrop_scale),
            RandomHorizontalFlipVideo(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(center_crop),
            # transforms.CenterCrop(center_crop),
            # transforms.Resize(input_res),
            # normalize,
        ]),
        # 'test': transforms.Compose([
        #     ,transforms.Resize(input_res)
        # ])
    }
    return tsfm_dict

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC),
        transforms.CenterCrop(n_px),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])