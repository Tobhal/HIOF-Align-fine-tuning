from PIL import Image
from data.dataset_bengali import ImageLoader

import os

class ImageCache:
    def __init__(self, root: str):
        self.cache = {}
        self.image_loader = ImageLoader(root)

    def load_image(self, img_path):
        # Check if the image is already in the cache
        if img_path not in self.cache:
            # If not, load it, ensuring to close the file handle after loading
            self.cache[img_path] = self.image_loader(img_path)
                
        return self.cache[img_path]

    def preload_images(self, data_loader, image_path_func):
        """
        Preload images from a DataLoader into an ImageCache.
        """
        for batch in data_loader:
            image_paths = image_path_func(batch)
            for img_path in image_paths:
                self.load_image(img_path)  # Load and cache the image

        print("Preloading complete: Cached", len(self.cache), "images.")

    @staticmethod
    def extract_image_paths(batch):
        *_, image_names, _, _ = batch
        return [img_name for img_name in image_names]
