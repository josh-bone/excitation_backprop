# excitation_backprop
Pytorch implementation of the algorithm described in "Top-Down Neural Attention by Excitation Backprop" by Zhang et. al

This code was written in a hurry and it was written for a specific network architecture. However, most layers used in modern CNN's are implemented in this repo. If you want to implement excitation backprop in your own model, you should use the functions here, and follow my faster_rcnn_ebp function as an example.

If you happen to be using the faster-rcnn model, you're in luck! Just build it as described [here](https://github.com/loolzaaa/faster-rcnn-pytorch), pop ebp.py into your top-level folder, and run the file.
