# Code of "Malware Traffic Classification using Domain Adaptation and Ladder Network for Secure Industrial Internet of Things"

### Data Preparation

Download the dataset at https://github.com/yungshenglu/USTC-TK2016 and follow the instructions to generate the mnist format 

Decompress the data compression package in the *mnist/* directory to ensure that its file organization is:

```
mnist/
	pre_train/
		t10k-images-idx3-ubyte
		t10k-labels-idx1-ubyte
		train-images-idx3-ubyte
		train-labels-idx1-ubyte
	trans/
		t10k-images-idx3-ubyte
		t10k-labels-idx1-ubyte
		train-images-idx3-ubyte
		train-labels-idx1-ubyte
```

### Environments

Ubuntu 18.04 <br>
Python 3.6.2 <br>TensorFlow >=1.12.0 <br>keras >=3.6.2 

### Train Model

Use python to run the file directly, take KT-CNN as an example: 

```python
python KT-CNN.py
```

