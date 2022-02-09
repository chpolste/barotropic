from numbers import Number
import numpy as np
import scipy.interpolate
import scipy.signal
import scipy.ndimage
import spharm
from . import formatting
from .constants import EARTH_RADIUS, EARTH_OMEGA, MERIDIONAL


class Grid:
    """Regular lon-lat grid and operations for a spherical planet.

    Note:
        Latitudes start at the North Pole, as is convention in
        :py:mod:`spharm`. This results in horizontally flipped images when
        plotting fields in matplotlib without explicitly specifying the
        latitude grid.

    Parameters:
        resolution (number): Grid resolution (uniform) in degrees.
        rsphere (number): Radius of the planet in m.
        omega (number): Angular velocity of the planet in 1/s.
        ntrunc (int): Threshold for triangular truncation.
        legfunc (str): Parameter given to :py:class:`spharm.Spharmt`.

    The default values of **rsphere** and **omega** correspond to those of
    planet Earth. Consult the `spharm <https://github.com/jswhit/pyspharm>`_
    documentation for further information on the **ntrunc** and **legfunc**
    parameters.

    Attributes:
        nlon, nlat: Number of longitudes/latitudes.
        dlon, dlat: Longitude/latitude spacing in degrees.
        lon, lat: Longitude/latitude coordinates in degrees (1D).
        lon2, lat2: Longitude/latitude coordinates in degrees (2D).
        dlam, dphi: Longitude/latitude spacing in radians.
        lam, phi: Longitude/latitude coordinates in radians (1D).
        lam2, phi2: Longitude/latitude coordinates in radians (2D).
        fcor: Coriolis parameter in 1/s (1D).
        fcor2: Coriolis parameter in 1/s (2D).
    """

    def __init__(self, resolution=2.5, rsphere=EARTH_RADIUS, omega=EARTH_OMEGA,
            ntrunc=None, legfunc="stored"):
        # Planet parameters (radius, angular velocity)
        self.rsphere = rsphere
        self.omega = omega
        # Setup longitude and latitude grid. Make sure longitude 0° is not
        # repeated and that both latitudes 90° and -90° exist. Latitudes start
        # at the North pole, which is the convention used by pyspharm.
        self.nlon = int(360. / resolution)
        self.nlat = int(180. / resolution) + 1
        if self.nlat % 2 == 0:
            raise ValueError("Number of latitudes must be odd but is {}".format(self.nlat))
        self.lon = np.linspace( 0., 360., self.nlon, endpoint=False)
        self.lat = np.linspace(90., -90., self.nlat, endpoint=True)
        self.lon2, self.lat2 = np.meshgrid(self.lon, self.lat)
        # Grid spacing in degrees
        self.dlon = resolution
        self.dlat = -resolution
        # Spherical coordinate grid (for use with trigonometric functions)
        self.lam = np.deg2rad(self.lon)
        self.phi = np.deg2rad(self.lat)
        self.lam2, self.phi2 = np.meshgrid(self.lam, self.phi)
        # Grid spacing in radians
        self.dlam = np.deg2rad(self.dlon)
        self.dphi = np.deg2rad(self.dlat)
        # Spherical harmonic transform object
        self._spharm = spharm.Spharmt(self.nlon, self.nlat, rsphere=rsphere, gridtype="regular", legfunc=legfunc)
        self._ntrunc = (self.nlat - 1) if ntrunc is None else ntrunc
        # Eigenvalues of the horizontal Laplacian for each spherical harmonic.
        # Force use of complex64 datatype (= 2 * float32) because spharm will
        # cast to float32 components anyway but the multiplication with the
        # python scalars results in float64 values.
        _, specindxn = spharm.getspecindx(self._ntrunc)
        self._laplacian_eigenvalues = (
                specindxn * (1. + specindxn) / rsphere / rsphere
                ).astype(np.complex64, casting="same_kind")
        # Precompute Coriolis field
        self.fcor = self.coriolis(self.lat)
        self.fcor2 = self.coriolis(self.lat2)
        self.fcor2_spectral = self.to_spectral(self.fcor2)

    def __repr__(self):
        return formatting.grid_repr(self)

    def __eq__(self, other):
        return (
            self.shape == other.shape
            and np.isclose(self.rsphere, other.rsphere)
            and np.isclose(self.omega, other.omega)
            and self._ntrunc == other._ntrunc
        )

    def __hash__(self):
        return hash((self.shape, self.rsphere, self.omega, self._ntrunc))

    @property
    def shape(self):
        """Tuple of grid dimensions (:py:attr:`nlat`, :py:attr:`nlon`)."""
        return self.nlat, self.nlon

    def circumference(self, lat):
        """Circumference (in m) of the sphere at (a) given latitude(s).

        Parameters:
            lat (number | array): Input latitude(s) in degrees.

        Returns:
            Circumference or array of circumferences.
        """
        return 2 * np.pi * self.rsphere * np.cos(np.deg2rad(lat))

    def coriolis(self, lat):
        """Coriolis parameter (in m) for a given latitude (in degrees).

        Parameters:
            lat (number | array): Input latitude(s) in degrees.

        Returns:
            Coriolis parameter or array of Coriolis parameters.
        """
        return 2. * self.omega * np.sin(np.deg2rad(lat))

    # Region extraction

    @property
    def region(self):
        """Region extractor with indexing syntax.

        .. note::
            Coordinate order is [lon,lat]. This is inverted compared to indexing into the arrays.
        """
        return GridRegionIndexer(self)

    # Spectral-grid transforms

    def to_spectral(self, field_grid):
        """Transform a gridded field into spectral space.

        Parameters:
            field_grid (array): Gridded representation of input field.

        Returns:
            Spectral representation of input field.
        """
        return self._spharm.grdtospec(field_grid, self._ntrunc)

    def to_grid(self, field_spectral):
        """Transform a spectral field into grid space.

        Parameters:
            field_spectral (array): Spectral representation of input field.

        Returns:
            Gridded representation of input field.
        """
        return self._spharm.spectogrd(field_spectral)

    # Wind computations

    def wind(self, vorticity, divergence):
        """Wind components from vorticity and divergence fields.

        Parameters:
            vorticity (array): Vorticity field (gridded).
            divergence (array): Divergence field (gridded).

        Return:
            Tuple of (zonal, meridional) wind compontents (gridded).
        """
        if vorticity.shape == self.shape:
            vorticity = self.to_spectral(vorticity)
        if divergence.shape == self.shape:
            divergence = self.to_spectral(divergence)
        return self._spharm.getuv(vorticity, divergence)

    def vorticity(self, u, v):
        """Vorticity from vector components.

        Parameters:
            u (array): Zonal component of input field (gridded).
            v (array): Meridional component of input field (gridded).

        Returns:
            Gridded vorticity field.
        """
        return self.to_grid(self.vorticity_spectral(u, v))

    def vorticity_spectral(self, u, v):
        """:py:meth:`vorticity` with spectral output field."""
        return self.vorticity_divergence_spectral(u, v)[0]

    def divergence(self, u, v):
        """Gridded divergence from vector components.

        Parameters:
            u (2D array): Zonal component of input vector field.
            v (2D array): Meridional component of input vector field.

        Returns:
            2D divergence field.
        """
        return self.to_grid(self.divergence_spectral(u, v))

    def divergence_spectral(self, u, v):
        """:py:meth:`divergence` with spectral output field."""
        return self.vorticity_divergence_spectral(u, v)[1]

    def vorticity_divergence(self, u, v):
        """Vorticity and divergence from vector components.

        Parameters:
            u (array): Zonal component of input field (gridded).
            v (array): Meridional component of input field (gridded).

        Returns:
            Tuple of gridded (vorticity, divergence) fields.
        """
        vort, div = vorticity_divergence_spectral(u, v)
        return self.to_grid(vort), self.to_grid(div)

    def vorticity_divergence_spectral(self, u, v):
        """:py:meth:`vorticity_divergence` with spectral output fields."""
        return self._spharm.getvrtdivspec(u, v, self._ntrunc)

    # Derivatives and PDE solvers

    def gradient(self, f):
        """Gridded vector gradient of a horizontal (2D) field.

        Parameters:
            f (array): 2D input field.

        Returns:
            A tuple with the two components of the horizontal gradient.

        Horizontal gradient on the sphere::

            ∇f(φ,λ) = (1/r ∂f/∂φ, 1/(r sin(φ)) ∂f/∂λ)

        """
        return self._spharm.getgrad(self.to_spectral(f))

    def derivative_meridional(self, f, order=4):
        """Finite difference first derivative in meridional direction (``∂f/∂φ``).

        Parameters:
            f (array): 1D or 2D input field.
            order (2 | 4): Order of the finite-difference approximation.

        Returns:
            Array containing the derivative.

        Accepts both 2D ``f = f(φ,λ)`` and 1D ``f = f(φ)`` fields. For 1D input, the
        derivatives at the poles are always set to zero, as the input is
        assumed to represent a zonally symmetric profile of some quantity (e.g.
        zonal-mean PV).

        The 2nd order approximation uses a 3-point stencil, the 4th order
        approximation a 5-point stencil (centered in inner domain, offset at
        the edges). Stencil coefficients were obtained from
        http://web.media.mit.edu/~crtaylor/calculator.html.
        """
        f = np.asarray(f)
        assert f.ndim <= 2
        # Output has same shape as input
        out = np.zeros_like(f)
        if order == 2:
            assert self.nlat >= 3, "Order 2 approximation requires at least 3 latitudes"
            # Inner domain: centered difference
            out[1:-1,...] = f[2:,...] - f[:-2,...]
            # Only evaluate pole points for higher dimensions (see docstring)
            if f.ndim != 1:
                out[0   ,...] = -3*f[0,...] + 4*f[1,...] - f[2,...]
                out[  -1,...] = f[-3,...] - 4*f[-2,...] + 3*f[-1,...]
            return out / 2 / self.dphi
        if order == 4 or order is None:
            assert self.nlat >= 5, "Order 4 approximation requires at least 5 latitudes"
            # Northern edge of interior domain: offset 5-point stencil
            out[1   ,...] = -3*f[0,...] - 10*f[1,...] + 18*f[2,...] - 6*f[3,...] + f[4,...]
            # Interior domain: centered 5-point stencil
            out[2:-2,...] = f[:-4,...] - 8*f[1:-3,...] + 8*f[3:-1,...] - f[4:,...]
            # Southern edge of interior domain: offset 5-point stencil
            out[  -2,...] = -f[-5,...] + 6*f[-4,...] - 18*f[-3,...] + 10*f[-2,...] + 3*f[-1,...]
            # Only evaluate pole points for higher dimensions (see docstring)
            if f.ndim != 1:
                out[0   ,...] = -25*f[0,...] + 48*f[1,...] - 36*f[2,...] + 16*f[3,...] - 3*f[4,...]
                out[  -1,...] = 3*f[-5,...] - 16*f[-4,...] + 36*f[-3,...] - 48*f[-2,...] + 25*f[-1,...]
            # Common divisor of stencil is 12
            return out / 12 / self.dphi
        raise NotImplementedError("Requested order of approximation not available (choose 2 or 4)")

    def laplace(self, f):
        """Laplacian of an input field.

        Parameters:
            f (array): 2D input field.

        Returns:
            Gridded Laplacian of **f**.
        """
        return self.to_grid(self.laplace_spectral(self.to_spectral(f)))

    def laplace_spectral(self, f):
        """:py:meth:`laplace` with spectral in- and output fields."""
        return -f * self._laplacian_eigenvalues

    def solve_poisson(self, rhs_grid, op_add=0.):
        """Solve Poisson(-like) equations.

        Parameters:
            rhs_grid (array): Input field for right hand side of Poisson
                equation.
            op_add (number): Optional term with constant coefficient.

        Returns:
            Solution to the Poisson(-like) equation.

        Discretized equation::

            (∆ - op_add) f = rhs
        """
        rhs_spec = self.to_spectral(rhs_grid)
        solution = self.solve_poisson_spectral(rhs_spec, op_add)
        return self.to_grid(solution)

    def solve_poisson_spectral(self, rhs_spec, op_add=0.):
        """:py:meth:`solve_poisson` with spectral in- and output fields."""
        solution = np.zeros_like(rhs_spec)
        solution[1:] = -rhs_spec[1:] / (self._laplacian_eigenvalues[1:] + op_add)
        return solution

    def solve_diffusion(self, field_grid, coeff, dt, order=1):
        """Advance diffusion equations of various order with an implicit step.

        Parameters:
            field_grid (array): 2D input field.
            coeff (number): Diffusion coefficient.
            dt (number): Time step in s.
            order (int): Order of diffusion. ``1``: regular diffusion, ``2``:
                hyperdiffusion, ...

        Returns:
            Solution after one ``dt``-sized step.

        Solves::

            ∂f/∂t = κ·∇²f

        (here order = 1) with an implicit Euler step.
        """
        field_spec = self.to_spectral(field_grid)
        solution = self.solve_diffusion_spectral(field_spec, coeff, dt, order)
        return self.to_grid(solution)

    def solve_diffusion_spectral(self, field_spectral, coeff, dt, order=1):
        """:py:meth:`solve_diffusion` with spectral in- and output fields."""
        eigenvalues_op = self._laplacian_eigenvalues ** order
        return field_spectral / (1. + dt * coeff * eigenvalues_op)

    # Area-weighted operators

    @property
    def gridpoint_area(self):
        """Surface area of each gridpoint as a function of latitude.

        The area associated with a gridpoint (λ, φ) in a regular lon-lat grid::

            r² * dλ * ( sin(φ + dφ) - sin(φ - dφ) )
        """
        # Calculate dual phi grid (latitude mid-points)
        mid_phi = 0.5 * (self.phi[1:] + self.phi[:-1])
        # Start with scaling factor
        gridpoint_area = np.full(self.nlat, self.rsphere * self.rsphere * self.dlam, dtype=float)
        # Calculate latitude term of area formula
        gridpoint_area[1:-1] *= np.sin(mid_phi[:-1]) - np.sin(mid_phi[1:])
        # Exceptions for polar gridpoints, which are "triangular"
        gridpoint_area[ 0] *= 1 - np.sin(mid_phi[ 0])
        gridpoint_area[-1] *= 1 + np.sin(mid_phi[-1])
        return gridpoint_area

    def mean(self, field, axis=None, region=None):
        """Area-weighted mean of the input field.

        Parameters:
            field (array): 2D input field.
            axis (None | int | "meridional" | "zonal"): Axis over which to
                compute the mean.
            region (None | :py:class:`GridRegion`): Region to which the mean is
                restricted. Construct regions with :py:attr:`region` or specify
                any object that implements a compatible **extract** method.

        Returns:
            Area-weighted mean.
        """
        # 
        # If a region is given, extract region if shape of field matches that
        # of grid else check that region has already been extracted from field
        if region is None:
            assert field.shape == self.shape
        elif field.shape == self.shape:
            field = region.extract(field)
        else:
            assert field.shape == region.shape
        # Determine area weights for mean calculation
        area = self.gridpoint_area[:,None] if region is None else region.gridpoint_area[:,None]
        # Pick normalization depending on axis over which mean is taken
        if axis is None:
            return (field * area).sum() / area.sum() / field.shape[1]
        elif axis == 0 or axis == -2 or axis == "meridional":
            return ((field * area).sum(axis=0) / area.sum(axis=0))
        elif axis == 1 or axis == -1 or axis == "zonal":
            return field.mean(axis=1)
        else:
            raise ValueError("invalid value for axis parameter: {}".format(axis))

    # Numerical quadrature

    def quad(self, y, z=None, method="sptrapz"):
        """Surface integral with an optional custom domain of integration.

        Parameters:
            y (array): 2D input field.
            z (array): Custom domain of integration given by the region **z** >= 0.
            method ("boxcount" | "sptrapz"): quadrature scheme.

        Returns:
            Value of the integral.

        See also :py:meth:`quad_meridional`.
        """
        return self.rsphere * self.dlam * np.sum(self.quad_meridional(y, z, method))

    def quad_meridional(self, y, z=None, method="sptrapz"):
        """Line integral along meridians.

        Parameters:
            y (array): 2D input field.
            z (array): Custom domain of integration given by the region **z** >= 0.
            method ("boxcount" | "sptrapz"): quadrature scheme.

        Returns:
            nlon-sized array of integral values.

        The sptrapz quadrature scheme is based on the trapezoidal rule adapted
        for sperical surface domains using the antiderivate of ``r * (a*φ + b) * cos(φ)``
        to integrate over the piecewise-linear, interpolated segments between
        gridpoints in the meridional direction. If given, the field that
        defines the custom domain of integration is also linearly interpolated.
        This implementation is accurate but slow compared to the much simpler
        boxcounting. Note that no interpolation is carried out in the zonal
        direction (since lines of constant latitude are not great-circles,
        linear interpolation is non-trivial). The boundaries of the domain of
        integration are therefore not continuous in the zonal direction even
        for the sptrapz scheme.

        See also :py:meth:`quad`.
        """
        # Check that number of meridional gridpoints matches. Allow mismatch in
        # longitude to make sectoral zonalization possible.
        assert y.ndim == 2, "input field must be 2-dimensional"
        assert y.shape[MERIDIONAL] == self.nlat, "shape mismatch in meridional dimension"
        # Method: boxcounting quadrature
        if method == "boxcount":
            # Integrate everywhere or extract domain of integration from z
            domain = True if z is None else (z >= 0)
            # Boxcounting is just area times value in the domain
            area = np.sum(self.gridpoint_area[:,None] * y, where=domain, axis=MERIDIONAL)
            # Convert to line integral
            return area / self.rsphere / self.dlam
        # Method: trapezoidal rule for the sphere
        elif method == "sptrapz":
            # Take only as much of phi as needed for the given data (input might
            # only be a sector of the full globe)
            x = self.phi2[:,:y.shape[1]]
            # If no z-values are given, integrate everywhere
            if z is None:
                z = np.ones_like(y)
            # Construct slicing tuple based on latitude axis (-2)
            js = slice(None,   -1), slice(None, None)
            je = slice(   1, None), slice(None, None)
            # Precompute x- and y-distances along axis
            dx = x[je] - x[js]
            dy = y[je] - y[js]
            # Compute slopes and offsets of piecewise linear y-interpolation
            # y = a*x + b
            aa = dy / dx
            bb = y[js] - aa * x[js]
            # Determine integration domain: where are z-values greater than 0?
            nonneg = z >= 0
            # Case 1: Intervals in which z-values change from negative to positive are
            # trimmed such that z = 0 at the left boundary point.
            interp_start = ~nonneg[js] & nonneg[je]
            zl = z[js][interp_start] # left z-value of interval
            zr = z[je][interp_start] # right z-value of interval
            # Compute position of z-root, this is the new left interval boundary
            xs = x[js].copy()
            xs[interp_start] = xs[interp_start] - zl / (zr - zl) * dx[interp_start]
            # Case 2: Intervals in which z-values change from positive to negative are
            # trimmed such that z = 0 at the right boundary point.
            interp_end = nonneg[js] & ~nonneg[je]
            zl = z[js][interp_end] # left z-value of interval
            zr = z[je][interp_end] # right z-value of interval
            # Compute position of z-root, this is the new right interval boundary
            xe = x[je].copy()
            xe[interp_end] = xe[interp_end] - zr / (zr - zl) * dx[interp_end]
            # Piecewise integration of
            #   (a*x + b) * r * cos(x)
            # along meridian using antiderivative
            #   (a*cos(x) - (a*x + b) * sin(x)) * r
            # Integrate in reverse order to flip sign of result since phi goes from
            # +pi/2 to -pi/2
            trapz = self.rsphere * (
                        aa * (np.cos(xs) - np.cos(xe))
                        + (aa * xs + bb) * np.sin(xs)
                        - (aa * xe + bb) * np.sin(xe))
            # Only intervals where z is positive somewhere are considered
            return np.sum(trapz, where=np.logical_or(nonneg[js], nonneg[je]), axis=MERIDIONAL)
        # Unknown method of quadrature
        else:
            raise ValueError(f"unknown quadrature scheme '{method}'")

    # Equivalent-latitude zonalization

    def equivalent_latitude(self, area):
        """Latitude such that surface up to North Pole is **area**-sized.

        Parameters:
            area (number | array): input area (in m²).

        Returns:
            Latitude(s) in degrees.

        Solve formula for area ``A``, north of latitude ``φ``, for ``φ``::

            A = 2 * pi * (1 - sin(φ)) * r²
            φ = arcsin(1 - A / 2 / pi / r²)
        """
        # Calculate argument of arcsin
        sine = 1. - 0.5 * area / np.pi / self.rsphere**2
        # Make sure argument of arcsin is in [-1, 1]
        sine[sine >=  1] =  1
        sine[sine <= -1] = -1
        # Equivalent latitude in degrees
        return np.rad2deg(np.arcsin(sine))

    def zonalize(self, field, levels=None, interpolate=True, quad="sptrapz"):
        """Zonalize the field with equivalent latitude coordinates.

        Parameters:
            field (array): 2D input field.
            levels (None | array | int): Contours generated for the
                zonalization. If an array of contour-values is given, these are
                used directly. If value is an integer, this number of contours
                is sampled between the maximum and minimum values in the input
                field automatically. By default, the number of levels is set
                equal to a multiple of the number of latitudes resolved by the
                grid (2 for boxcounting quadrature, 1 otherwise).
            interpolate (bool): Interpolate output onto regular latitudes of
                grid. Without interpolation, values are returned on the
                equivalent latitudes that arise in the computation. These may
                be irregular and unordered.
            quad (str): Quadrature rule used in the surface integrals of the
                computation. See :py:meth:`quad`.

        Returns:
            If **interpolate** is `True`, return the contour values
            interpolated onto the regular latitudes as an array. Otherwise,
            return a tuple of contour values and associated equivalent
            latitudes (both arrays).

        Implements the zonalization procedure of Nakamura and Zhu (2010).
        """
        # Select contours for area computations
        q_min = np.min(field)
        q_max = np.max(field)
        # If nothing is specified about the contour levels, use a multiple of
        # nlat many (1 for the trapezoidal scheme, 2 for boxcounting)
        if levels is None:
            levels = self.nlat * (2 if quad == "boxcount" else 1)
        # If contours is specified as the number of contours to use, distribute
        # contours between the min and max found in field, following sine in
        # the interval -90° to 90° and scaled and offset such that the min and
        # max are reached. This should be a decent distribution in the general
        # case as it resembles the distribution of planetary PV. Omit min and
        # max contours as field >= min everywhere and field >= max most likely
        # only in a single point.
        if isinstance(levels, int):
            _ = 0.5 * np.pi * np.linspace(-1, 1, levels+2)[1:-1]
            q = 0.5 * ((q_max - q_min) * np.sin(_) + q_min + q_max)
        # Otherwise use the given contour values
        else:
            q = levels
        # Integrating a field of ones in the regions where each threshold is
        # exceeded yields the area of these regions
        ones = np.ones_like(field)
        area = np.vectorize(lambda thresh: self.quad(ones, z=(field-thresh), method=quad))(q)
        # If the input field is only a sector of the full globe (determined by
        # the width relative to the full globe at the given resolution), adjust
        # the integrated contour areas such that they correspond to the full
        # globe and compute equivalent latitude for the globe
        area_factor = self.nlon / field.shape[1]
        # Calculate equivalent latitude associated with contour areas
        y = self.equivalent_latitude(area * area_factor)
        # No interpolation: return contour values on internal equivalent
        # latitudes and return the latitudes as well for reference
        if not interpolate:
            return q, y
        # When interpolated, only return the contour values
        return self.interpolate_meridional(q, y, pole_values=(q_max, q_min))

    # Interpolation

    def interp1d_meridional(self, field, lat, pole_values, kind="linear"):
        """Meridional interpolation function.

        Parameters:
            field (array): 1D meridional profile or 2D field on the input
                latitudes.
            lat (array): Input latitudes in degrees.
            pole_values ((number, number)): Values at the poles (North, South),
                required to complete the interpolation.
            kind (string): Given to :py:func:`scipy.interpolate.interp1d`.

        Returns:
            Configured :py:func:`scipy.interpolate.interp1d` interpolation
            function.

        See also: :py:meth:`Grid.interpolate_meridional`.
        """
        assert 1 <= field.ndim <= 2
        assert len(pole_values) == 2
        pad = (1, 1) if field.ndim == 1 else ((1, 1), (0, 0))
        axis = -1    if field.ndim == 1 else MERIDIONAL
        x = np.pad(lat, pad_width=(1, 1), mode="constant", constant_values=(90., -90.))
        y = np.pad(field, pad_width=pad, mode="constant", constant_values=pole_values)
        return scipy.interpolate.interp1d(x, y, axis=axis, kind=kind, copy=False)

    def interpolate_meridional(self, *interp1d_args, **interp1d_kwargs):
        """Interpolation onto the regular latitude grid.

        Constructs and immediately evaluates the interpolation function from
        :py:meth:`Grid.interp1d_meridional` for :py:attr:`Grid.lat`.
        """
        return self.interp1d_meridional(*interp1d_args, **interp1d_kwargs)(self.lat)

    # Filtering

    def get_filter_window(self, window, width):
        """Wraps :py:func:`scipy.signal.get_window` for degree input.

        Parameters:
            window (str): The type of window to create.
            width (number): Width of the window in degrees.

        Returns:
            Gridded window function.

        Window widths are restricted to odd numbers of gridpoints so windows
        can properly be centered on a gridpoint during convolution. The
        returned window array is normalized such that it sums to 1.
        """
        # Convert width to gridpoints
        width = round(width / self.dlon)
        width = width if width % 2 == 1 else width + 1
        window = scipy.signal.get_window(window, width, fftbins=False)
        return window / np.sum(window)

    def filter_meridional(self, field, window, width=None):
        """Filter the input in meridional direction with the given window.

        Parameters:
            field (array): 1 or 2-dimensional input signal.
            window (str | array): Window used in the convolution.
            width (number): Width of the window.

        Returns:
            Filtered field.

        If **width** is ``None``, **window** must be a gridded window given as
        a 1D array.  Otherwise, **window** and **width** are given to
        :py:meth:`get_filter_window` to obtain a window function.
        """
        if width is not None:
            window = self.get_filter_window(window, width)
        # Use symmetrical boundary condition
        if field.ndim == 1:
            assert field.size == self.nlat
            return scipy.ndimage.convolve(field, window, mode="reflect")
        elif field.ndim == 2:
            assert field.shape[0] == self.nlat
            return scipy.ndimage.convolve(field, window[:,None], mode="reflect")
        else:
            raise ValueError("input field must be 1- or 2-dimensional")


    def filter_zonal(self, field, window, width=None):
        """Filter the input in zonal direction with the given window.

        Parameters:
            field (array): 1 or 2-dimensional input signal.
            window (str | array): Window used in the convolution.
            width (number): Width of the window.

        Returns:
            Filtered field.

        If **width** is ``None``, **window** must be a gridded window given as
        a 1D array.  Otherwise, **window** and **width** are given to
        :py:meth:`get_filter_window` to obtain a window function.
        """
        if width is not None:
            window = self.get_filter_window(window, width)
        # Use periodic boundary condition (only makes sense if 
        if field.ndim == 1:
            assert field.size == self.nlon
            return scipy.ndimage.convolve(field, window, mode="wrap")
        elif field.ndim == 2:
            assert field.shape[1] == self.nlon
            return scipy.ndimage.convolve(field, window[None,:], mode="wrap")
        else:
            raise ValueError("input field must be 1- or 2-dimensional")



class GridRegionIndexer:

    def __init__(self, grid):
        self._grid = grid

    def __getitem__(self, selection):
        # Non-tuple selections apply to longitude only
        if not isinstance(selection, tuple):
            selection = selection, slice(None, None)
        # Selecting a 2-dimensional region
        if len(selection) != 2:
            raise IndexError("too many dimensions in region selection")
        # Compute the indices that extract the selected region in each
        # dimension (only rectangular, axis-aligned regions possible)
        lon_indices = self._get_lon_indices(selection[0])
        lat_indices = self._get_lat_indices(selection[1])
        # Now in index world, axes order is flipped there (lat first)
        return GridRegion(self._grid, lat_indices, lon_indices)

    def _get_lat_indices(self, slc):
        # Allow special values N and S to select hemispheres
        if slc == "N":
            slc = slice(0, 90)
        if slc == "S":
            slc = slice(-90, 0)
        # Selection must be a numeric slice without a step parameter
        if not isinstance(slc, slice):
            raise IndexError("latitude selection must be given as a slice")
        if not (isinstance(slc.start, Number) or slc.start is None):
            raise IndexError("start value of latitude selection must be numeric or None")
        if not (isinstance(slc.stop, Number) or slc.stop is None):
            raise IndexError("stop value of latitude selection must be numeric or None")
        if slc.step is not None:
            raise IndexError("step parameter not supported for latitude selection")
        # Compute the indices that achive the the latitude range selection.
        # Both ends of the selection are inclusive. Treat stop < start in the
        # same way as stop < start.
        indices = np.arange(self._grid.nlat)
        lo, hi = slc.start, slc.stop
        if lo is None and hi is None:
            return indices
        elif lo is None:
            return indices[self._grid.lat <= hi]
        elif hi is None:
            return indices[lo <= self._grid.lat]
        else:
            return indices[(min(lo, hi) <= self._grid.lat) & (self._grid.lat <= max(lo, hi))]

    def _get_lon_indices(self, slc):
        # Selection must be a numeric slice without a step parameter
        if not isinstance(slc, slice):
            raise IndexError("longitude selection must given as a slice")
        if not (isinstance(slc.start, Number) or slc.start is None):
            raise IndexError("start value of longitude selection must be numeric or None")
        if not (isinstance(slc.stop, Number) or slc.stop is None):
            raise IndexError("stop value of longitude selection must be numeric or None")
        if slc.step is not None:
            raise IndexError("step parameter not supported for longitude selection")
        # Compute the indices that achive the the longitude range selection.
        # Both ends of the selection are inclusive. If stop < start, select
        # across the 0°-meridian and assemble into a contiguous region.
        indices = np.arange(self._grid.nlon)
        lo, hi = slc.start, slc.stop
        if lo is None and hi is None:
            return indices
        elif lo is None:
            return indices[self._grid.lon <= hi]
        elif hi is None:
            return indices[lo <= self._grid.lon]
        elif hi < lo:
            lo_mask = lo <= self._grid.lon
            return np.roll(indices[(self._grid.lon <= hi) | lo_mask], np.count_nonzero(lo_mask))
        else:
            return indices[(lo <= self._grid.lon) & (self._grid.lon <= hi)]



class GridRegion:

    def __init__(self, grid, lat_indices, lon_indices):
        self._grid = grid
        self._lon_indices = np.require(lon_indices, dtype=int)
        self._lat_indices = np.require(lat_indices, dtype=int)
        assert self._lat_indices.ndim == 1 and self._lat_indices.size <= grid.nlat
        assert self._lon_indices.ndim == 1 and self._lon_indices.size <= grid.nlon

    def __repr__(self):
        return formatting.grid_region_repr(self)

    def _extract_one(self, field):
        # Meridional profile
        if field.ndim == 1 and field.size == self._grid.nlat:
            return field[self._lat_indices]
        # Zonal profile
        elif field.ndim == 1 and field.size == self._grid.nlon:
            return field[self._lon_indices]
        # 2-dimensional field
        elif field.shape == self._grid.shape:
            # https://stackoverflow.com/questions/42309460
            return field[np.ix_(self._lat_indices, self._lon_indices)]
        else:
            raise ValueError("unable to extract from field '{}'".format(field))

    def extract(self, *fields):
        """Extract the region from the given fields."""
        if len(fields) == 0:
            raise ValueError("no field(s) given for extraction")
        elif len(fields) == 1:
            return self._extract_one(fields[0])
        else:
            return tuple(map(self._extract_one, fields))

    @property
    def mask(self):
        """Boolean array that is true in the region."""
        mask = np.full(self._grid.shape, False)
        mask[np.ix_(self._lat_indices, self._lon_indices)] = True
        return mask

    @property
    def shape(self):
        """Shape of extracted 2D fields."""
        return self._lat_indices.size, self._lon_indices.size

    @property
    def lat(self):
        """Latitudes of the region."""
        return self._grid.lat[self._lat_indices]

    @property
    def lon(self):
        """Longitudes of the region.

        If the region crosses the 0° meridian, these will not be monotonic. If
        you need a monotonic longitude coordinate, e.g. for plotting, use
        `lon_mono`, where longitudes left of 0° are reduced by 360°.
        """
        return self._grid.lon[self._lon_indices]

    @property
    def lon_mono(self):
        """Longitudes of the region, monotonic even for regions crossing 0°"""
        lon = self.lon
        jump = np.argwhere(np.diff(lon) < 0)
        assert 0 <= jump.size <= 1
        if jump.size == 1:
            lon[:jump[0,0]+1] -= 360.
        return lon

    @property
    def gridpoint_area(self):
        return self.extract(self._grid.gridpoint_area)

    def mean(self, field):
        return self._grid.mean(field, region=self)

