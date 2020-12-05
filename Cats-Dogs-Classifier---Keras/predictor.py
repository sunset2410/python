#The trained model is saved as the classifier.h5
#The predictor.py shows the way to use that seperately. 

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

#Make sure the version is correct
classifier = load_model('classifier_chieu.h5') #classifier_chieu.h5


test_image = image.load_img("F:/python_project/python/Cats-Dogs-Classifier---Keras/dataset/single_prediction/dog.7.jpg", target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
#print(np.argmax(result))
#print(result)
{'dogs': 1, 'cats': 0}
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print("Yes! It's a cute "+prediction)
