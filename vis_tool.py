import numpy as np
import matplotlib
import torch as t

from matplotlib import pyplot as plot


def vis_image(img, ax=None):
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax


def vis_bbox(img, bbox, label=None, score=None, ax=None):
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()

        if label is not None:
            lb = label[i]
            caption.append(str(lb))
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax