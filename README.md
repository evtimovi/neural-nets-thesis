# neural-nets-thesis
This is a repository for the code work to be performed as part of a senior thesis at Lafayette College under the supervision of Professor Amir Sadovnik.

The folder basic-mnist contains implementations of the neural network that will be trained to recognize digits from the MNIST data set. These follow the TensorFlow tutorials [here](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html) and [here](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html#deep-mnist-for-experts).

The folder measure-accuracy contains a script that runs the VGG face networkand then computes its accuracy.
The intro comment in the script contains details but it can be run with this command:

'''
python measure_accuracy.py lfw/ pairsDevTest.txt
'''

The folder vggface contains an implementation of the VGG neural network used for facial recognition.
The researchers who developed the network are O.M. Parkhi, A. Vedaldi, and A. Zisserman and their paper was called Deep Face Recognition (published in 2015 in the British Machine Vision Conference). Their work carries a [Creative Commons Attribution License](https://creativecommons.org/licenses/by-nc/4.0/legalcode).  
The code for implementing the network in tensorflow is available on GitHub [here](https://github.com/AKSHAYUBHAT/TensorFace). Additional modifications were done by Wassim Gharbi'19 at Lafayette College.
