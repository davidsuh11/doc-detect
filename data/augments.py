import albumentations as A
import albumentations.pytorch as AP
import cv2 as cv

# Define transforms for training and validation sets
preprocess_train = A.Compose([
    A.Resize(256, 256),
    A.Normalize(max_pixel_value=1.),
    AP.ToTensorV2()
])

augmentations_train = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Transpose(),
    A.ShiftScaleRotate(shift_limit = 0.3, scale_limit = 1., border_mode=cv.BORDER_REPLICATE),
    A.RandomShadow(),
    A.RandomSunFlare(num_flare_circles_lower=1, num_flare_circles_upper=2, src_radius=50),
    A.MotionBlur(),
    A.RandomBrightnessContrast(p=0.8),
    A.Normalize(max_pixel_value=1.),
    preprocess_train
])

