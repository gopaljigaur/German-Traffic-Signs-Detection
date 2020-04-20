
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import load_model
import cv2

model=load_model('traffic-signs.h5')

y_test=pd.read_csv("Test.csv")
paths=y_test['Path'].values
y_test=y_test['ClassId'].values
l=len(y_test)
data=[]

for f in paths:
    imge=cv2.imread(f)
    image_from_array = Image.fromarray(imge, 'RGB')
    size_image = image_from_array.resize((32, 32))
    data.append(np.array(size_image))

X_test=np.array(data)
X_test = X_test.astype('float32')/255 
pred = model.predict_classes(X_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))


plt.figure(0)
test_img_path='D:/NN/datasets/traffic-signs/Test/00025.png'
test_img_plot=plt.imread(test_img_path,0)
plt.imshow(test_img_plot)
test_img = image.load_img(test_img_path, target_size=(32,32))
x = image.img_to_array(test_img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print(classes)

plt.figure(1)
test_img1=plt.imread('D:/NN/datasets/traffic-signs/Meta/{0}.png'.format(classes[0]))
new=plt.imshow(test_img1)
