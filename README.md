# Eye-power
Controlling a cursor with your eye.
1. [eye] : Regression model
2. [eye_classification]: 9-class classifier model

## Collecting training images

1. Run 1-create_training_images.py
- Watch out for the pop up screen
- Maximize the screen
- Fix the position of your eye wrt to the camera
- Focus both eyes on the dot
- Press 'c' to capture the image
- Upon pressing 'c', images of both eyes will be randomly saved in the trian and test folders (3:1)
- Upon pressing 'c', another pop up screen showing the camera view will appear for your reference
- Keep pressing 'c' to capture more images
- Nubmer of training images is shown in the top-left corner.

2. Run 2-train_cnn.py
- First, a pop up scatter plot showing the distribution of trainin points will appear. 
- Ensure that you have a good spread of points (~300-600 is good, or more)
- Save the scatter plot if necessary, then close the screen
- Training will commence. Some model params (e.g. epochs) can be changed in the __init__ of Utils() class.
- Model architecture can be changed in the init_model() method in the Utils() class.
- Upon conclusion of training, the (1) MSE-epoch plot will pop up, and (2) the error plot will pop up.

3. Run 3-demo.py
- Run the file
- Watch out for the pop up screen, and maximize it.
- Fix the position of your eye wrt to the camera
- Use your eye to move the cursor across the screen.
