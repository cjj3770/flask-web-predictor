from keras.applications.resnet50 import ResNet50
#from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
#import numpy as np

Resnet50_model = ResNet50(weights='imagenet')
Resnet50_model.save('Resnet50.h5', include_optimizer=False)

