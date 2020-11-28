# sphinx_gallery_thumbnail_number = 3
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

print(reg)
print('yo')

def modelsSuper(h, w):
    print(h, w)
    return h, w


x = modelsSuper(179, 60)
print(x)
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.
plt.show()
