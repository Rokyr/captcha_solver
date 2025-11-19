# captcha_solver

dataset: https://www.kaggle.com/datasets/fournierp/captcha-version-2-images

This project implements a deep learning model that predicts a fixed number of characters from an input image to solve captchas. Can also be used for multi-character recogntition tasks.

Residual CNN backbone with multiple softmax output heads.

If anyone is planning on re-using this. I'd reccomend adding a save function for model after training, so it may be re-used, or for during training. 
