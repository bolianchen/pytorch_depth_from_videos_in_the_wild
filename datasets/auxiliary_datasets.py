import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision

def make_image_checker(img_ext):
    def is_processed_image(filename):
        if not filename.lower().endswith(img_ext):
            return False
        if 'fseg' in filename:
            return False
        return True
    return is_processed_image

class ProcessedImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, img_ext):
        super().__init__(
            root, 
            torchvision.transforms.ToTensor(),
            is_valid_file=make_image_checker(img_ext)
        )
    
    def __getitem__(self, index):
        # Load processed image
        path, _ = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)

        return image, path

class ImageDataset(Dataset):
    """Collect all the images in the specified folder"""

    def __init__(self, image_dir, transform=None):
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')
        img_files = [f for f in os.listdir(image_dir) if f.lower().endswith(IMG_EXTENSIONS)]
        self.image_paths = [os.path.join(image_dir, f) for f in img_files]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        im = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im
