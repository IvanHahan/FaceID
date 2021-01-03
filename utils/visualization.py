import matplotlib.pyplot as plt


def visualize_series(images):
    for im in images:
        plt.imshow(im)
        plt.show()