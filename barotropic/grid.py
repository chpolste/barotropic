import numpy as np
import spharm
from .constants import EARTH_RADIUS, EARTH_OMEGA


class Grid:
    """Regular lat-lon grid and operations for a spherical planet

    Uses (truncated) spherical harmonics for some operations utilizing the
    pyspharm package.

    The latitudes starts at the north pole, as is convention in pyspharm. This
    results in horizontally flipped images when plotting fields in matplotlib
    without explicitly specifying the latitude grid. Apply numpy.flipud to
    obtain fields that start at the south-pole.
    """

    def __init__(self, rsphere=EARTH_RADIUS, omega=EARTH_OMEGA, latlon_resolution=2.5,
            ntrunc=None, legfunc="stored"):
        """Regular lat-lon grid for a spherical planet
        
        The radius and angular velocity of the planet are given by `rsphere` in
        m and `omega` in 1/s, respectively (the default values correspond to
        those of the Earth).

        The grid resolution is uniform and specified with the
        `latlon_resolution` parameter in degrees.

        By default (`ntrunc=None`, `legfunc="stored"`), the spherical harmonics are
        truncated after (number of latitudes - 1) functions and precomputed.
        See spharm.Spharmt for more information on these parameters.
        """
        # Planet parameters (radius, angular velocity)
        self.rsphere = rsphere
        self.omega = omega
        # Setup longitude and latitude grid. Make sure longitude 0° is not
        # repeated and that both latitudes 90° and -90° exist. Latitudes start
        # at the North pole, which is the convention used by pyspharm.
        n_lon = int(360. / latlon_resolution)
        n_lat = int(180. / latlon_resolution) + 1
        if n_lat % 2 == 0:
            raise ValueError("Number of latitudes must be odd but is {n_lat}".format(n_lat=n_lat))
        lons = np.linspace( 0., 360., n_lon, endpoint=False)
        lats = np.linspace(90., -90., n_lat, endpoint=True)
        self.lon, self.lat = np.meshgrid(lons, lats)
        # Spherical coordinate grid (for use with trigonometric functions)
        lam = np.deg2rad(lons)
        phi = np.deg2rad(lats)
        self.lam, self.phi = np.meshgrid(lam, phi)
        # Grid spacing in radians
        self.dlam = 2 * np.pi / self.lam.shape[1]
        self.dphi = np.pi / self.phi.shape[0]
        # Precompute Coriolis field
        self.fcor = self.coriolis(self.lat)
        # Spherical harmonic transform object
        self._spharm = spharm.Spharmt(n_lon, n_lat, rsphere=rsphere, gridtype="regular", legfunc=legfunc)
        self._ntrunc = (n_lat - 1) if ntrunc is None else ntrunc
        _, self.specindxn = spharm.getspecindx(self._ntrunc)
        # Eigenvalues of the horizontal Laplacian for each spherical harmonic.
        # Force use of complex64 datatype (= 2 * float32) because spharm will
        # cast to float32 components anyway but the multiplication with the
        # python scalars results in float64 values.
        self.laplacian_eigenvalues = (
                self.specindxn * (1. + self.specindxn) / rsphere / rsphere
                ).astype(np.complex64, casting="same_kind")
        # Keep some additional grid properties for convenience
        self.longitudes = self.lon[0,:]
        self.latitudes = self.lat[:,0]

    @property
    def shape(self):
        """Grid dimensions"""
        return self.phi.shape

    def circumference(self, lat):
        """Circumference of the sphere at constant latitude"""
        return 2 * np.pi * self.rsphere * np.cos(np.deg2rad(lat))

    def coriolis(self, lat):
        """Coriolis parameter for a given latitude"""
        return 2. * self.omega * np.sin(np.deg2rad(lat))

    # Spectral-grid transforms

    def to_spectral(self, field_grid):
        """Transform gridded field into spectral space"""
        return self._spharm.grdtospec(field_grid, self._ntrunc)

    def to_grid(self, field_spectral):
        """Transform spectral field into gridded space"""
        return self._spharm.spectogrd(field_spectral)

    # Wind computations

    def wind(self, vorticity, divergence):
        """Compute gridded wind components from vorticity and divergence fields"""
        vorticity_spectral = self.to_spectral(vorticity)
        divergence_spectral = self.to_spectral(divergence)
        return self._spharm.getuv(vorticity_spectral, divergence_spectral)

    def vorticity(self, u, v):
        """Compute gridded vorticity from wind components"""
        return self.to_grid(self.vorticity_spectral(u, v))

    def vorticity_spectral(self, u, v):
        """Compute spectral vorticity from wind components"""
        return self.vorticity_divergence_spectral(u, v)[0]

    def divergence(self, u, v):
        """Compute gridded divergence from vector components"""
        return self.to_grid(self.divergence_spectral(u, v))

    def divergence_spectral(self, u, v):
        """Compute spectral divergence from vector components"""
        return self.vorticity_divergence_spectral(u, v)[1]

    def vorticity_divergence(self, u, v):
        """Compute gridded vorticity and divergence from vector components"""
        vort, div = vorticity_divergence_spectral(u, v)
        return self.to_grid(vort), self.to_grid(div)

    def vorticity_divergence_spectral(self, u, v):
        """Compute spectral vorticity and divergence from vector components"""
        return self._spharm.getvrtdivspec(u, v, self._ntrunc)

    # Numerical quadrature

    @property
    def gridpoint_area(self):
        """Surface area associated with each gridpoint based on a dual grid
        
        Associated area of gridpoint (lon, lat) in regular grid:
            r² * dlon * [ sin(lat + dlat) - sin(lat - dlat) ]

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
        """Surface integral summing (area * value of `y`) at every gridpoint
        
        Domain of integration (optionally) specified by boolean array `where`.
        """
        return np.sum(self.gridpoint_area * y, where=where)

    def quad_sptrapz(self, y, z=None):
        """Surface integral based on meridional linear interpolation
        
        See `Grid.quad_sptrapz_meridional`.
        """
        return self.rsphere * self.dlam * np.sum(self.quad_sptrapz_meridional(y, z))

    def quad_sptrapz_meridional(self, y, z=None):
        """Line integral of `y` along meridians using linear interpolation

        A custom domain of integration can be specified with array argument
        `z`. The domain is determined by the condition z >= 0 using linear
        interpolation. If `z` is not given, the entire integration domain is
        the entire surface of the sphere.

        The quadrature rule is based on the trapezoidal rule adapted for
        sperical surface domains using the antiderivate of
            r * (a*lat + b) * cos(lat)
        to integrate over the piecewise-linear, interpolated segments between
        gridpoints in the meridional direction.

        No interpolation is carried out in zonal direction (since lines of
        constant latitude are not great-circles, linear interpolation is
        non-trivial. The boundaries of the domain of integration are therefore
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
        """Latitude such that surface up to North Pole is `area`-sized

        Area north of latitude lat:
          area = 2 * pi * (1 - sin(lat)) * r²
        Solve for latitude:
           lat = arcsin(1 - area / 2 / pi / r²)
        """
        # Calculate argument of arcsin
        sine = 1. - 0.5 * area / np.pi / self.rsphere**2
        # Make sure argument of arcsin is in [-1, 1]
        sine[sine >=  1] =  1
        sine[sine <= -1] = -1
        # Equivalent latitude in degrees
        return np.rad2deg(np.arcsin(sine))

    def zonalize_eqlat(self, field, levels=None, interpolate=None, quad="sptrapz"):
        """Zonalize the field with equivalent latitude coordinates

        Implements zonalization procedure of Nakamura and Zhu (2010).
        
        The number of contours generated from field for the zonalization is
        determined by the `levels` parameter. If `levels` is an integer, the
        contours are sampled equidistantly between the highest and lowest
        occuring value in the field. If `levels` is a list of contour-values,
        these are used directly. By default (`levels=None`), the number of
        levels is set equal to the number of latitudes resolved by the grid.

        An output latitude grid can be specified with the `interpolate`
        argument.  The zonalized contour values are then interpolated to this
        grid using linear interpolation. By default (`interpolate=None`), no
        interpolation is carried out.

        The quadrature rule used in the surface integrals of the zonalization
        computation can be specified with the `quad` argument. Possible values
        are "sptrapz" and "boxcount", corresponding to methods quad_sptrapz and
        quad_boxcount, respectively. It is highly recommended to use the
        slower, but much more accurate sptrapz quadrature to avoid the
        "jumpiness" of the boxcounting scheme.
        """
        # Select contours for area computations
        q_min = np.min(field)
        q_max = np.max(field)
        # If nothing is specified about the contour levels, use as many as
        # there are gridpoints in meridional direction
        if levels is None:
            levels = 2 * self.latitudes.size
        # If contours is specified as the number of contours to use, distribute
        # contours linearly between the min and max found in field. Omit min
        # and max as field >= min everywhere and field >= max most likely only
        # in a single point.
        if isinstance(levels, int):
            q = np.linspace(q_min, q_max, levels + 2)[1:-1]
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

