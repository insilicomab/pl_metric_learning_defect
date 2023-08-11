from omegaconf import DictConfig
from torchvision import transforms


class NoneTransform:
    def __call__(self, image):
        return image


class Transforms():

    def __init__(self, config: DictConfig) -> None:
        
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(
                    (config.train_transform.resize.image_size, config.train_transform.resize.image_size)
                )
                if config.train_transform.resize.enable
                else NoneTransform(),
                transforms.RandomHorizontalFlip(p=config.train_transform.random_horizontal_flip.p)
                if config.train_transform.random_horizontal_flip.enable
                else NoneTransform(),
                transforms.RandomVerticalFlip(p=config.train_transform.random_vertical_flip.p)
                if config.train_transform.random_vertical_flip.enable
                else NoneTransform(),
                transforms.RandomRotation(degrees=config.train_transform.random_rotation.degrees)
                if config.train_transform.random_rotation.enable
                else NoneTransform(),
                transforms.RandomAffine(
                    degrees=config.train_transform.random_affine.degrees,
                    translate=config.train_transform.random_affine.translate,
                    scale=config.train_transform.random_affine.scale,
                    shear=config.train_transform.random_affine.shear,
                )
                if config.train_transform.random_affine.enable
                else NoneTransform(),
                transforms.ColorJitter(
                    brightness=config.train_transform.color_jitter.brightness,
                    contrast=config.train_transform.color_jitter.contrast,
                    saturation=config.train_transform.color_jitter.saturation,
                    hue=config.train_transform.color_jitter.hue
                )
                if config.train_transform.color_jitter.enable
                else NoneTransform(),
                transforms.ToTensor(),
                transforms.Normalize(
                    config.train_transform.normalize.mean,
                    config.train_transform.normalize.std
                ),
                ]),
            'val': transforms.Compose([
                transforms.Resize(
                    (config.test_transform.resize.image_size, config.test_transform.resize.image_size)
                )
                if config.test_transform.resize.enable
                else NoneTransform(),
                transforms.ToTensor(),
                transforms.Normalize(
                    config.test_transform.normalize.mean,
                    config.test_transform.normalize.std
                ),
                ]),
            'test': transforms.Compose([
                transforms.Resize(
                    (config.test_transform.resize.image_size, config.test_transform.resize.image_size)
                )
                if config.test_transform.resize.enable
                else NoneTransform(),
                transforms.ToTensor(),
                transforms.Normalize(
                    config.test_transform.normalize.mean,
                    config.test_transform.normalize.std
                ),
                ]),
        }
    
    def __call__(self, phase, img):
        return self.data_transform[phase](img)


class TestTransforms():

    def __init__(self, image_size):
        
        self.data_transform = {
            'test': transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
                ]),
        }
    
    def __call__(self, phase, img):
        return self.data_transform[phase](img)