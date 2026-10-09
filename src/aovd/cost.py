"""Annotation / compute cost model.

Annotation time for N fully-labeled images:  N * (t_img + b * t_box)  seconds,
with b the mean number of boxes per image of the target dataset.
Defaults (t_img = 5 s of viewing/class selection, t_box = 10 s per drawn box) are
*assumptions*; the paper reports sensitivity over t_box in {5, 10, 35} s.
"""


def annotation_hours(n_images, boxes_per_image, t_img=5.0, t_box=10.0):
    return n_images * (t_img + boxes_per_image * t_box) / 3600.0


def break_even_images(extra_gain_hours, boxes_per_image, t_img=5.0, t_box=10.0):
    """Number of images whose annotation time equals `extra_gain_hours`."""
    return extra_gain_hours * 3600.0 / (t_img + boxes_per_image * t_box)
