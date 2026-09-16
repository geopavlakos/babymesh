import torch


def get_seq_figure_skip(seq_name=None):
    """
    Frame stride used when accumulating a static multi-exposure figure
    (`accumulate=True` in vis/output.py's build_pyrender_scene).
    """
    return 10


def get_seq_static_lookat_points(seq_name=None, bounds=None):
    """
    Returns ((top_source, top_target), (side_source, side_target)) camera
    placements for the "above"/"side" render views, fit to the scene's bounding
    box so the subject is framed regardless of the capture volume/coordinate
    convention.
    """
    if bounds is not None:
        bb_min, bb_max, center = bounds
        length = torch.abs(bb_max - bb_min).max()
        top_source = center + torch.tensor([0.0, -2.0, -0.9 * length])
        side_source = center + torch.tensor([0.5 * length, -0.5, -0.7 * length])
        return (top_source, center), (side_source, center)

    top_source = torch.tensor([0.0, -2.0, -3.0])
    top_target = torch.tensor([0.0, 0.0, 1.0])

    side_source = torch.tensor([3.0, -1.0, -1.0])
    side_target = torch.tensor([0.0, 0.0, 1.0])
    return (top_source, top_target), (side_source, side_target)
