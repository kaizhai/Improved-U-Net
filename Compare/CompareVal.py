import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import TUNET
import Cloudseg
import FCN
import Seg_Net


def predict(image,save_path=None):
    image = np.expand_dims(image, axis=0)
    result = model.predict(image)

    result = result > 0.5

    # image
    plt.imshow(image[0])
    plt.axis('off')
    plt.title("")
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
test_path = current_dir + "/Test/"
# Test image ids
test_ids = next(os.walk(test_path))[2]
# Directory to load trained weights from
weights_dir = current_dir + "/NewWeights/"
#weights_filename = "TUNET.h5"
#weights_filename = "Cloudseg.h5"
#weights_filename = "FCN.h5"
#weights_filename = "Seg.h5"

## TCDD weights
#weights_filename = "NewTUNET.h5"
weights_filename = "NewCloudseg.h5"
#weights_filename = "NewFCN.h5"
#weights_filename = "NewSeg.h5"

"""TUNET.image_size = image_size
TUNET.input_dropout_rate = 0
TUNET.dropout_rate = 0
model = TUNET.UNet()"""

#model = FCN.FCN_model(128, dropout_rate=0.2)

#model = Seg_Net.SegNet(128)

model = Cloudseg.Cloudseg(128)


# Loading weights
weights = model.load_weights(weights_dir + weights_filename)

# Create a directory to save results
results_dir = os.path.join(current_dir, "Results/")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

for image_id in test_ids:
    image_path = os.path.join(test_path, image_id)
    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred = predict(image, save_path=results_dir)

