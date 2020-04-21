# U-Net, R2U-Net and IterNet for Retinal image segmentation

## requirements

use
~~~
pip install -r requirements.txt
~~~
to install all required packages

## test

Test mode will run the model on test set of DRIVE dataset.

Before get started, download pre-trained model from: https://drive.google.com/drive/folders/1AxaeHDFN-X8EDN38znEysthh2tSdD_0u?usp=sharing 

command:
~~~
python3 main.py --model MODEL_NAME --show T_OR_F
~~~

'--model' refers to the model's name which includes 'U-Net', 'R2U-Net' and 'IterNet'.

'--show' refers to if visualize the result

example:
~~~
python3 main.py --model IterNet --show True
~~~

This will run IterNet on test set and show the predicted result for each image.