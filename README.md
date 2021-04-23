# U-Net, R2U-Net and IterNet for Retinal image segmentation

## Introduction

PyTorch implementation of [U-Net](https://arxiv.org/abs/1505.04597), [R2U-Net](https://arxiv.org/pdf/1802.06955.pdf) and [IterNet](https://arxiv.org/abs/1912.05763). Running on DRIVE dataset.


## Train
~~~
python3 main.py --model MODEL_NAME --epoch NUM_EPOCHS
~~~


## Test


You can also download pre-trained model from: https://drive.google.com/drive/folders/1AxaeHDFN-X8EDN38znEysthh2tSdD_0u?usp=sharing 

command:
~~~
python3 main.py --model MODEL_NAME --mode test --show T_OR_F
~~~

'--model' refers to the model's name which includes 'U-Net', 'R2U-Net' and 'IterNet'.

'--show' refers to if visualize the result

example:
~~~
python3 main.py --model IterNet --show True
~~~

This will run IterNet on test set and show the predicted result for each image.