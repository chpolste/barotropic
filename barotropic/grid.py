import numpy as np
import spharm
from .constants import EARTH_RADIUS, EARTH_OMEGA


class Grid:
    """Regular lat-lon grid and operations for a spherical planet.

    Latitudes start at the north pole, as is convention in `spharm`. This
    results in horizontally flipped images when plotting fields in matplotlib
    without explicitly specifying the latitude grid.

    Attributes set during initialization:

    - `nlon`, `nlat`: Number of longitudes/latitudes.
    - `dlon`, `dlat`: Longitude/latitude spacing in degrees.
    - `lons`, `lats`: Longitude/latitude coordinates in degrees (1D).
    - `lon`, `lat`: Longitude/latitude coordinates in degrees (2D).
    - `dlam`, `dphi`: Longitude/latitude spacing in radians.
    - `lams`, `phis`: Longitude/latitude coordinates in radians (1D).
    - `lam`, `phi`: Longitude/latitude coordinates in radians (2D).
    - `fcor`: Coriolis parameter in 1/s (2D).
    - `laplacian_eigenvalues`: Eigenvalues of the horizontal Laplace
      Operator for each spherical harmonic.

    Consider using the `ZONAL` and `MERIDIONAL` constants as convenient and
    readable accessors for the grid dimensions.
    """


    def __init__(self, rsphere=EARTH_RADIUS, omega=EARTH_OMEGA, resolution=2.5,
            ntrunc=None, legfunc="stored"):
        """Grid constructor.

        - The radius and angular velocity of the planet are given by `rsphere` in
          m and `omega` in 1/s, respectively (the default values correspond to
          those of the Earth).
        - The grid resolution is uniform and specified with the `resolution`
          parameter in degrees.
        - Uses spherical harmonics for some operations utilizing the
          `spharm`/`pyspharm` package. By default (`ntrunc=None`,
          `legfunc="stored"`), the spherical harmonics are truncated after (number
          of latitudes - 1) functions and precomputed. Consult the documentation of
          `spharm.Spharmt` for more information on these parameters.
        """
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
        self.lons = np.linspace( 0., 360., self.nlon, endpoint=False)
        self.lats = np.linspace(90., -90., self.nlat, endpoint=True)
        self.lon, self.lat = np.meshgrid(self.lons, self.lats)
        # Grid spacing in degrees
        self.dlon = resolution
        self.dlat = -resolution
        # Spherical coordinate grid (for use with trigonometric functions)
        self.lams = np.deg2rad(self.lons)
        self.phis = np.deg2rad(self.lats)
        self.lam, self.phi = np.meshgrid(self.lams, self.phis)
        # Grid spacing in radians
        self.dlam = np.deg2rad(self.dlon)
        self.dphi = np.deg2rad(self.dlat)
        # Precompute Coriolis field
        self.fcor = self.coriolis(self.lat)
        # Spherical harmonic transform object
        self._spharm = spharm.Spharmt(self.nlon, self.nlat, rsphere=rsphere, gridtype="regular", legfunc=legfunc)
        self._ntrunc = (self.nlat - 1) if ntrunc is None else ntrunc
        _, self.specindxn = spharm.getspecindx(self._ntrunc)
        # Eigenvalues of the horizontal Laplacian for each spherical harmonic.
        # Force use of complex64 datatype (= 2 * float32) because spharm will
        # cast to float32 components anyway but the multiplication with the
        # python scalars results in float64 values.
        self.laplacian_eigenvalues = (
                self.specindxn * (1. + self.specindxn) / rsphere / rsphere
                ).astype(np.complex64, casting="same_kind")

    @property
    def shape(self):
        """Tuple of grid dimensions (`nlat`, `nlon`)."""
        return self.phi.shape

    def circumference(self, lat):
        """Circumference (in m) of the sphere at constant latitude."""
        return 2 * np.pi * self.rsphere * np.cos(np.deg2rad(lat))

    def coriolis(self, lat):
        """Coriolis parameter (in m) for a given latitude (in degrees)."""
        return 2. * self.omega * np.sin(np.deg2rad(lat))

    # Spectral-grid transforms

    def to_spectral(self, field_grid):
        """Transform a gridded field into spectral space."""
        return self._spharm.grdtospec(field_grid, self._ntrunc)

    def to_grid(self, field_spectral):
        """Transform a spectral field into grid space."""
        return self._spharm.spectogrd(field_spectral)

    # Wind computations

    def wind(self, vorticity, divergence):
        """Gridded wind components from vorticity and divergence fields."""
        vorticity_spectral = self.to_spectral(vorticity)
        divergence_spectral = self.to_spectral(divergence)
        return self._spharm.getuv(vorticity_spectral, divergence_spectral)

    def vorticity(self, u, v):
        """Gridded vorticity from wind components."""
        return self.to_grid(self.vorticity_spectral(u, v))

    def vorticity_spectral(self, u, v):
        """Spectral vorticity from wind components."""
        return self.vorticity_divergence_spectral(u, v)[0]

    def divergence(self, u, v):
        """Gridded divergence from vector components."""
        return self.to_grid(self.divergence_spectral(u, v))

    def divergence_spectral(self, u, v):
        """Spectral divergence from vector components."""
        return self.vorticity_divergence_spectral(u, v)[1]

    def vorticity_divergence(self, u, v):
        """Gridded vorticity and divergence from vector components."""
        vort, div = vorticity_divergence_spectral(u, v)
        return self.to_grid(vort), self.to_grid(div)

    def vorticity_divergence_spectral(self, u, v):
        """Spectral vorticity and divergence from vector components."""
        return self._spharm.getvrtdivspec(u, v, self._ntrunc)

    # General derivatives

    def gradient(self, f):
        """Gridded vector gradient of the 2D field f(φ,λ).

        Returns a tuple containing the two components `1/r df/dφ` and
        `1/(r sin(φ)) df/dλ`.
        """
        return self._spharm.getgrad(self.to_spectral(f))

    def derivative_meridional(self, f, order=None):
        """Finite difference first derivative in meridional direction: `df/dφ`.

        Accepts both 2D `f(φ,λ) = f(lat,lon)` and 1D `f(φ) = f(lat)` fields.
        For 1D input, the derivatives at the poles are always set to zero, as
        the input is assumed to represent a zonally symmetric profile of some
        quantity (e.g. zonal-mean PV).

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

    # Numerical quadrature

    @property
    def gridpoint_area(self):
        """Surface area associated with each gridpoint based on a dual grid.
        
        The associated area of a gridpoint (lon, lat) in a regular grid is
        given by: `r² * dlon * [ sin(lat + dlat) - sin(lat - dlat) ]`

        Used for box-counting quadrature.
        """
        # Compute area associated with each gridpoint for box counting
        # integration:
        # Calculate dual phi grid (latitude mid-points)
        mid_phi = 0.5 * (self.phi[1:,:] + self.phi[:-1,:])
        # Calculate latitude term of area formula
        dlat_mid = np.sin(mid_phi[:-1,:]) - np.sin(mid_phi[1:,:])
        # Exceptions for polar gridpoints, which are "triangular"
        dlat_np = 1 - np.sin(mid_phi[ 0,:])
        dlat_sp = 1 + np.sin(mid_phi[-1,:])
        # Evaluate formula
        gridpoint_area = np.full_like(self.lam, self.rsphere * self.rsphere * self.dlam)
        gridpoint_area[   0,:] *= dlat_np
        gridpoint_area[1:-1,:] *= dlat_mid
        gridpoint_area[  -1,:] *= dlat_sp
        return gridpoint_area

    def quad_boxcount(self, y, where=True):
        """Surface integral summing (`area * value` of `y`) at every gridpoint.
        
        The domain of integration can optionally be specified by a boolean
        array in the `where` parameter.
        """
        return np.sum(self.gridpoint_area * y, where=where)

    def quad_sptrapz(self, y, z=None):
        """Surface integral based on meridional linear interpolation.
        
        See `Grid.quad_sptrapz_meridional`.
        """
        return self.rsphere * self.dlam * np.sum(self.quad_sptrapz_meridional(y, z))

    def quad_sptrapz_meridional(self, y, z=None):
        """Line integral of `y` along meridians using linear interpolation.

        A custom domain of integration can be specified with parameter `z`. The
        domain is then determined by the condition `z >= 0` using linear
        interpolation. If `z` is not given, the integration domain is the
        entire surface of the sphere.

        The quadrature rule is based on the trapezoidal rule adapted for
        sperical surface domains using the antiderivate of `r * (a*lat + b) * cos(lat)`
        to integrate over the piecewise-linear, interpolated segments between
        gridpoints in the meridional direction. This implementation is accurate
        but rather slow.

        No interpolation is carried out in the zonal direction (since lines of
        constant latitude are not great-circles, linear interpolation is
        non-trivial). The boundaries of the domain of integration are therefore
        not continuous in the zonal direction.
        """
        x = self.phi
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
        trapz[~(nonneg[js] | nonneg[je])] = 0
        return np.sum(trapz, axis=-2)
        # TODO: the where argument of np.sum was added only in 1.17, which is still very new
        #return np.sum(trapz, where=np.logical_or(nonneg[js], nonneg[je]), axis=-2)

    # Equivalent-latitude zonalization

    def equivalent_latitude(self, area):
        """Latitude such that surface up to North Pole is `area`-sized.

        - Area north of latitude `lat`: `area = 2 * pi * (1 - sin(lat)) * r²`.
        - Solve for latitude: `lat = arcsin(1 - area / 2 / pi / r²)`.
        """
        # Calculate argument of arcsin
        sine = 1. - 0.5 * area / np.pi / self.rsphere**2
        # Make sure argument of arcsin is in [-1, 1]
        sine[sine >=  1] =  1
        sine[sine <= -1] = -1
        # Equivalent latitude in degrees
        return np.rad2deg(np.arcsin(sine))

    def zonalize(self, field, levels=None, interpolate=None, quad="sptrapz"):
        """Zonalize the field with equivalent latitude coordinates.

        Implements the zonalization procedure of Nakamura and Zhu (2010).
        
        - The number of contours generated from the field for the zonalization
          is determined by the `levels` parameter. If `levels` is an integer,
          contours are sampled between the highest and lowest occuring value in
          the field. If `levels` is a list of contour-values, these are used
          directly. By default (`levels=None`), the number of levels is set
          equal to the number of latitudes resolved by the grid.
        - An output latitude grid can be specified with the `interpolate`
          argument. The zonalized contour values are then interpolated to this
          grid using linear interpolation. By default (`interpolate=None`), no
          interpolation is carried out.
        - The quadrature rule used in the surface integrals of the zonalization
          computation can be specified with the `quad` argument. Possible
          values are `"sptrapz"` and `"boxcount"`, corresponding to methods
          `Grid.quad_sptrapz` and `Grid.quad_boxcount`, respectively. It is highly
          recommended to use the slower, but much more accurate `"sptrapz"`
          quadrature to avoid the "jumpiness" of the boxcounting scheme.

        Returns a tuple containing the contour values and associated equivalent
        latitudes.
        """
        # Select contours for area computations
        q_min = np.min(field)
        q_max = np.max(field)
        # If nothing is specified about the contour levels, use as many as
        # there are gridpoints in meridional direction
        if levels is None:
            levels = self.nlat
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
        # Determine area where each threshold is exceeded
        if quad == "sptrapz":
            area_int = lambda thresh: self.quad_sptrapz(np.ones_like(field), z=(field - thresh))
        elif quad == "boxcount":
            area_int = lambda thresh: self.quad_boxcount(1., field >= thresh)
        else:
            raise ValueError("unknown quadrature method '{}'".format(quad))
        area = np.vectorize(area_int)(q)
        # Calculate equivalent latitude associated with contour areas
        y = self.equivalent_latitude(area)
        # If desired, interpolate values onto the given latitudes
        if interpolate is not None:
            q = np.interp(interpolate, y, q, left=q_min, right=q_max)
            y = interpolate
        return q, y

