# neural-nets-thesis
This is a repository for the code work to be performed as part of a senior thesis at Lafayette College under the supervision of Professor Amir Sadovnik.

The folder basic-mnist contains implementations of the neural network that will be trained to recognize digits from the MNIST data set. These follow the TensorFlow tutorials [here](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html) and [here](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html#deep-mnist-for-experts).

The folder 'foo' contains various scripts that are used to run and train the networks and/or perform various experiments on them. For example, a script that runs the VGG face networkand then computes its accuracy is contained in 'foo/measure_accuracy.py'.
The intro comment in the script contains details but it can be run with this command: ''python measure_accuracy.py lfw/ pairsDevTest.txt''

Generally, these scripts are meant to be copied into the root folder of the repo and run from there (so that they can find the various Python packages described below).

The folder 'vggface' contains an implementation of the VGG neural network used for facial recognition.
This folder is organized into a Python package structure and further the different networks are objects. So to invoke any particular implementation (or rather generate an instance of that object), one needs to 'from vggface import networks as vggn' and then call, for example, 'vggn.VGGFaceVanilla()' for a plain vanilla implementation of VGGFace. 
The researchers who developed the network are O.M. Parkhi, A. Vedaldi, and A. Zisserman and their paper was called Deep Face Recognition (published in 2015 in the British Machine Vision Conference). Their work carries a [Creative Commons Attribution License](https://creativecommons.org/licenses/by-nc/4.0/legalcode).  
The code for implementing the network in tensorflow is available on GitHub [here](https://github.com/AKSHAYUBHAT/TensorFace). Additional modifications were done by Wassim Gharbi'19 at Lafayette College.

The folder 'util' contains various utility modules which are NOT written in an object oriented fashion. Rather, they are split into various functions. So, in order to use the different performance measures, one needs to invoke 'from util import performance p' and then call them by writing 'p.fnmr(...)'
