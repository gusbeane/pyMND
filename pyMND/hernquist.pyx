from math import sqrt

cpdef _hernquist_potential_derivative_z(double R, double z, double M, double a, double G):
    """
    The value of the partial derivative in z direction of the Hernquist potential.
    Parameters
    ----------
    R : `float`
        Cylindrical radius.
    z : `float`
        Height.
    M : `float`
        Total mass of the dark matter halo.
    a : `float`
        Scale length of the dark matter halo.
    G : `float`
        Newton's gravitational constant (sets units).
    Returns
    -------
    pot_z : `float`
        The value of the partial derivative in z direction of the halo potential at pos.
    """
    cdef double pot_z, r

    r = sqrt(R*R + z*z)
    pot_z = G * M
    pot_z /= (r+a) * (r+a)
    pot_z *= z/r

    return pot_z

cpdef _hernquist_potential_derivative_R(double R, double z, double M, double a, double G):
    """
    The value of the partial derivative in R direction of the halo potential.
    Parameters
    ----------
    R : `float`
        Cylindrical radius.
    z : `float`
        Height.
    M : `float`
        Total mass of the dark matter halo.
    a : float
        Scale length of the dark matter halo.
    G : `float`
        Newton's gravitational constant (sets units).
    Returns
    -------
    pot_R : `float`
        The value of the partial derivative in R direction of the halo potential at pos.
    """
    cdef double pot_z, r

    r = sqrt(R*R + z*z)
    pot_R = G * M
    pot_R /= (r + a)*(r + a)
    pot_R *= R/r

    return pot_R
