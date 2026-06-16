import cv2
import matplotlib.pyplot as plt


def show(img, title="", figsize=(14, 10)):
    plt.figure(figsize=figsize)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show2(img1, title1, img2, title2):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, img, title in zip(axes, [img1, img2], [title1, title2]):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
