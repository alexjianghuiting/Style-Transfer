# Style-Transfer
Implementation of the Style-transfer model based on "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"

## Formula used
The stochastic gradient descent function for minimizing the difference between the output image and target image, making the output image (from the transform net) more like the target image
![Formula for gradient descent](https://github.com/alexjianghuiting/Style-Transfer/blob/master/formula/Pasted%20Graphic%202.png)

Feature Reconstruction Loss function, penalizes the difference in content
![Formula for gradient descent](https://github.com/alexjianghuiting/Style-Transfer/blob/master/formula/Pasted%20Graphic%203.png)

Covariance, capturing the information about which features tend to activate together
![Formula for gradient descent](https://github.com/alexjianghuiting/Style-Transfer/blob/master/formula/Pasted%20Graphic%204.png)

Style Reconstruction Loss function, penalizes the difference in style
![Formula for gradient descent](https://github.com/alexjianghuiting/Style-Transfer/blob/master/formula/Pasted%20Graphic%205.png)


Pixel Loss, normalized Euclidean distance between the output image and the target
(However, Feature Reconstruction Loss function does a better job at reconstructing fine details, leading to pleasing visual results)
![Formula for gradient descent](https://github.com/alexjianghuiting/Style-Transfer/blob/master/formula/Pasted%20Graphic%206.png)


Spatial smoothness

![Formula for gradient descent](https://github.com/alexjianghuiting/Style-Transfer/blob/master/formula/Pasted%20Graphic%207.png)

# Run
python style.py —-checkpoint-dir ./model/ —-style ./style/wave.jpg

python evaluate.py —-checkpoint ./model/[YOUR_CKPT_NAME].ckpt —-in-path ./examples/content/chicago.jpg —-out-path ./
