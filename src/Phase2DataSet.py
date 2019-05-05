import cv2
import glob
import re

kernel = 64
stride = int(kernel/2)
mask_ratio = 0.35
patch_size = (kernel*kernel)
fake_patches_output_dir = 'training/fake_patches'
pristine_patches_output_dir = 'training/pristine_patches'
fake_images_dir = 'training/fake/'

def num_black_pixels(mask):
    return len([p for p in mask.flatten() if p >= 200])

def image_to_patch_tuples(image, mask):
    '''
    Takes an image as input.
    Sample 64x64 patches along the image.
    '''
    patches = []

    img_width = image.shape[0]
    img_height = image.shape[1]

    for x in range(0, img_width - kernel - 1, stride):
        for y in range(0, img_height - kernel - 1, stride):
            n_black_pixels = num_black_pixels(mask[x:x+kernel, y:y+kernel, 0])
            if n_black_pixels > patch_size*mask_ratio and\
                n_black_pixels < patch_size*(1-mask_ratio):
                patches.append(image[x:x+kernel, y:y+kernel])
    return patches

def generate_and_store_patches(image_name):
    image_path = f'{fake_images_dir}/{image_name}.png'
    mask_path =  f'{fake_images_dir}/{image_name}.mask.png'
    patches = image_to_patch_tuples(cv2.imread(image_path), cv2.imread(mask_path))
    for index, patch in enumerate(patches):
        cv2.imwrite(f'{fake_patches_output_dir}/{image_name}-patch-{index}.png', patch)

def get_fake_image_names():
    examples = glob.glob(F"{fake_images_dir}/*")
    examples = [path for path in examples if 'mask' not in path]
    image_names = [re.search('(\w*).png', example).group(1) for example in examples]
    return image_names

def generate_fake_patches():
    fake_image_names = get_fake_image_names()
    for index, image_name in enumerate(fake_image_names):
        print(f'Progress: {index / len(fake_image_names) * 100}%')
        generate_and_store_patches(image_name)

if __name__ == '__main__':
    generate_fake_patches()
