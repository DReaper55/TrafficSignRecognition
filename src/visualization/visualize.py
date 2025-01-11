import matplotlib.pyplot as plt


def plot_images(images, labels, class_names):
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f"Class: {labels[i]}")
        plt.axis('off')
    plt.show()
