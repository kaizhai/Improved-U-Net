import os

import cv2
import matplotlib.pyplot as plt 
import numpy as np
import UNET

def predict(image,save_path=None):
    image = np.expand_dims(image, axis=0)
    result = model.predict(image)

    result = result > 0.5

    # image
    plt.imshow(image[0])
    plt.axis('off')
    plt.title("")
    if save_path:
        plt.savefig(os.path.join(save_path, 'original_image.png'), bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    plt.imshow(result[0], cmap="gray")
    plt.axis('off')  # Remove axis
    plt.title("")  # Remove title
    if save_path:
        plt.savefig(os.path.join(save_path, 'result_image.jpg'), bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    return result

image_size = 128
current_dir = os.getcwd()
# Test path having test images
test_path = os.path.join(current_dir, "Test")
# Test image ids
test_ids = next(os.walk(test_path))[2]
# Directory to load trained weights from
weights_dir = os.path.join(current_dir, "NewWeights")
#weights_filename = "MyModel.h5"
weights_filename = "NewMyModel.h5" # TCDD Dataset
weights_path = os.path.join(weights_dir, weights_filename)

UNET.image_size = image_size
UNET.input_dropout_rate = 0
UNET.dropout_rate = 0
model = UNET.UNet()
# Loading weights
weights = model.load_weights(weights_path)

for image_id in test_ids:
    image_path = os.path.join(test_path, image_id)
    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred = predict(image)

