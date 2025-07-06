import numpy as np
import discretize
from discretize import utils
import matplotlib.pyplot as plt

def get_theta_ind_mirror(mesh, theta_ind):
    return (
        theta_ind+int(mesh.vnC[1]/2)
        if theta_ind < int(mesh.vnC[1]/2)
        else theta_ind-int(mesh.vnC[1]/2)
    )

def mesh2d_from_3d(mesh):
        """
        create cylindrically symmetric mesh generator
        """
        mesh2D = discretize.CylindricalMesh(
            [mesh.h[0], 1., mesh.h[2]], x0=mesh.x0
        )
        return mesh2D
    
def plot_slice(
    mesh, v, ax=None, clim=None, pcolor_opts=None, theta_ind=0,
    cb_extend=None, show_cb=True
):
    """
    Plot a cell centered property

    :param numpy.array prop: cell centered property to plot
    :param matplotlib.axes ax: axis
    :param numpy.array clim: colorbar limits
    :param dict pcolor_opts: dictionary of pcolor options
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if pcolor_opts is None:
        pcolor_opts = {}
    if clim is not None:
        norm = Normalize(vmin=clim.min(), vmax=clim.max())
        pcolor_opts["norm"] = norm

    # generate a 2D mesh for plotting slices
    mesh2D = mesh2d_from_3d(mesh)

    vplt = v.reshape(mesh.vnC, order="F")
    plotme = discretize.utils.mkvc(vplt[:, theta_ind, :])
    if not mesh.is_symmetric:
        theta_ind_mirror = get_theta_ind_mirror(mesh, theta_ind)
        mirror_data = discretize.utils.mkvc(vplt[:, theta_ind_mirror, :])
    else:
        mirror_data = plotme

    out = mesh2D.plot_image(
        plotme, ax=ax,
        mirror=True, mirror_data=mirror_data,
        pcolor_opts=pcolor_opts,
    )

    out += (ax, )

    if show_cb:
        cb = plt.colorbar(
            out[0], ax=ax,
            extend=cb_extend if cb_extend is not None else "neither"
        )
        out += (cb, )

        # if clim is not None:
        #     cb.set_clim(clim)
        #     cb.update_ticks()

    return out


def pad_for_casing_and_data(
    casing_outer_radius,
    csx1=2.5e-3,
    csx2=25,
    pfx1=1.3,
    pfx2=1.5,
    domain_x=1000,
    npadx=10
):

    ncx1 = np.ceil(casing_outer_radius/csx1+2)
    npadx1 = np.floor(np.log(csx2/csx1) / np.log(pfx1))

    # finest uniform region
    hx1a = utils.unpack_widths([(csx1, ncx1)])

    # pad to second uniform region
    hx1b = utils.unpack_widths([(csx1, npadx1, pfx1)])

    # scale padding so it matches cell size properly
    dx1 = np.sum(hx1a)+np.sum(hx1b)
    dx1 = 3 #np.floor(dx1/self.csx2)
    hx1b *= (dx1*csx2 - np.sum(hx1a))/np.sum(hx1b)

    # second uniform chunk of mesh
    ncx2 = np.ceil((domain_x - dx1)/csx2)
    hx2a = utils.unpack_widths([(csx2, ncx2)])

    # pad to infinity
    hx2b = utils.unpack_widths([(csx2, npadx, pfx2)])

    return np.hstack([hx1a, hx1b, hx2a, hx2b])
