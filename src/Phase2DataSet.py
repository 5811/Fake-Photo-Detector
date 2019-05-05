import cv2
import glob
import re

kernel = 64
stride = int(kernel/2)
mask_ratio = 0.35
patch_size = (kernel*kernel)

fake_images_dir = 'training/fake/'
pristine_images_dir = 'training/pristine'

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

def generate_and_store_patches(image_name, dir, generate_mask=False):
    image_path = f'{dir}/{image_name}.png'
    mask_path =  f'{dir}/{image_name}.mask.png'

    image = cv2.imread(image_path)
    if generate_mask:
        mask = np.zeros(image)
    else:
        mask = cv2.imread(mask_path)

    patches = image_to_patch_tuples(image, mask)
    for index, patch in enumerate(patches):
        cv2.imwrite(f'{dir}_patches/{image_name}-patch-{index}.png', patch)

def get_image_names(dir):
    examples = glob.glob(F"{dir}/*")
    examples = [path for path in examples if 'mask' not in path]
    image_names = [re.search('(\w*).png', example).group(1) for example in examples]
    return image_names

def generate_fake_patches():
    fake_image_names = get_image_names(fake_images_dir)
    for index, image_name in enumerate(fake_image_names):
        print(f'Progress: {index / len(fake_image_names) * 100}%')
        generate_and_store_patches(image_name, pristine_images_dir)

def generate_pristine_images():
    pristine_image_names = get_image_names(pristine_images_dir)
    for index, image_name in enumerate(pristine_image_names):
        print(f'Progress: {index / len(pristine_image_names) * 100}%')
        generate_and_store_patches(image_name, pristine_images_dir, True)
    import pdb; pdb.set_trace();

if __name__ == '__main__':
    # generate_fake_patches()
    generate_pristine_images()