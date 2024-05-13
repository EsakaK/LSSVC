import torch


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def get_coord_cell(shape):
    coord_hr = make_coord(shape)
    cell = torch.ones_like(coord_hr)
    cell[:, 0] *= 2 / (coord_hr.shape[-2])
    cell[:, 1] *= 2 / (coord_hr.shape[-1])
    return coord_hr, cell
