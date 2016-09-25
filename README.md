# neural-nets-thesis
This is a repository for the code work to be performed as part of a senior thesis at Lafayette College under the supervision of Professor Amir Sadovnik.

The folder basic-mnist contains implementations of the neural network that will be trained to recognize digits from the MNIST data set. These follow the TensorFlow tutorials [here](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html) and [here](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html#deep-mnist-for-experts).

The folder measure-accuracy contains a script that runs the VGG face networkand then computes its accuracy.
The intro comment in the script contains details but it can be run with this command:

'''bash
python measure_accuracy.py lfw/ pairsDevTest.txt
'''

Also adding the vggface code which is not implemented by me. It came from [here](https://github.com/AKSHAYUBHAT/TensorFace).
