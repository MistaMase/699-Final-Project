# 699-Final-Project

For testing the emotion model, the code would look like this:

from skimage import io

img = image.load_img('testimages/wallpaper.jpg', grayscale=True, target_size=(48, 48))

show_img=image.load_img('testimages/wallpaper2you_443897.jpg', grayscale=False, target_size=(200, 200))

x = image.img_to_array(img)

x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)

print('Expression Prediction:',objects[ind])
