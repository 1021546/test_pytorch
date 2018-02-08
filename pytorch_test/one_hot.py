# first method
import numpy as np

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

y_test=np.zeros(3, dtype=np.int)
for i in range(0,3):
	y_test[i]=i

nb_classes = 6

print(to_categorical(y_test, nb_classes))

# second method
# import numpy as np

# y_test=np.zeros(3, dtype=np.int)
# for i in range(0,3):
# 	y_test[i]=i

# nb_classes = 6

# one_hot_targets = np.eye(nb_classes)[y_test]

# print(one_hot_targets)