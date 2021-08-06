from numbers import Number
import numpy as np
import spharm
from . import formatting
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

    Consider using the `ZONAL` and `MERIDIONAL` constants as convenient and
    readable accessors for the grid dimensions.
    """

    def __init__(self, resolution=2.5, rsphere=EARTH_RADIUS, omega=EARTH_OMEGA,
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
        self.fcor_spectral = self.to_spectral(self.fcor)

    def __repr__(self):
        return formatting.grid_repr(self)

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

    # Region extraction

    @property
    def region(self):
        """Create a region extractor with indexing syntax."""
        return GridRegionIndexer(self)

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
        if vorticity.shape == self.shape:
            vorticity = self.to_spectral(vorticity)
        if divergence.shape == self.shape:
            divergence = self.to_spectral(divergence)
        return self._spharm.getuv(vorticity, divergence)

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

    # Derivatives and PDE solvers

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

    def laplace(self, f):
        """TODO"""
        return self.to_grid(self.laplace_spectral(self.to_spectral(f)))

    def laplace_spectral(self, f):
        """TODO"""
        return -f * self._laplacian_eigenvalues

    def solve_poisson(self, rhs_grid, op_add=0.):
        """TODO"""
        rhs_spec = self.to_spectral(rhs_grid)
        solution = self.solve_poisson_spectral(rhs_spec, op_add)
        return self.to_grid(solution)

    def solve_poisson_spectral(self, rhs_spec, op_add=0.):
        """Solve `(∆ - op_add) f = rhs`

        TODO
        """
        solution = np.zeros_like(rhs_spec)
        solution[1:] = -rhs_spec[1:] / (self._laplacian_eigenvalues[1:] + op_add)
        return solution

    def solve_diffusion(self, field_grid, coeff, dt, order=1):
        """Advance diffusion equations of various order with an implicit step

        Wraps Grid.solve_diffusion_spectral. If you intend to integrate
        multiple steps in direct sequence, convert to spectral representation
        once, take steps with solve_diffusion_spectral, then transform back
        instead.
        """
        field_spec = self.to_spectral(field_grid)
        solution = self.solve_diffusion_spectral(field_spec, coeff, dt, order)
        return self.to_grid(solution)

    def solve_diffusion_spectral(self, field_spectral, coeff, dt, order=1):
        """Advance diffusion equations of various order with an implicit step

        Takes an implicit Euler step.
        order=1 → diffusion
        order=2 → hyperdiffusion
        ...

        Solves ∂f/∂t = κ·∇²f etc.
        """
        eigenvalues_op = self._laplacian_eigenvalues ** order
        return field_spectral / (1. + dt * coeff * eigenvalues_op)

    # Area-weighted operators

    @property
    def gridpoint_area(self):
        """Surface area of each gridpoint as a function of latitude.
        
        The associated area of a gridpoint (lon, lat) in a regular grid is
        given by: `r² * dlon * [ sin(lat + dlat) - sin(lat - dlat) ]`
        """
        # Calculate dual phi grid (latitude mid-points)
        mid_phis = 0.5 * (self.phis[1:] + self.phis[:-1])
        # Start with scaling factor
        gridpoint_area = np.full(self.nlat, self.rsphere * self.rsphere * self.dlam, dtype=float)
        # Calculate latitude term of area formula
        gridpoint_area[1:-1] *= np.sin(mid_phis[:-1]) - np.sin(mid_phis[1:])
        # Exceptions for polar gridpoints, which are "triangular"
        gridpoint_area[ 0] *= 1 - np.sin(mid_phis[ 0])
        gridpoint_area[-1] *= 1 + np.sin(mid_phis[-1])
        return gridpoint_area

    def mean(self, field, axis=None, region=None):
        """Area-weighted mean of `field`.

        - The mean over the entire region is calculated by default. By
          specifying the `axis` argument, a zonal or meridional mean can be
          calculated.
        - A region to which the mean should be restricted can be given with the
          `region` parameter.
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
        return np.sum(trapz, where=np.logical_or(nonneg[js], nonneg[je]), axis=-2)

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

    # Filtering

    def get_filter_window(self, window, width):
        """Wraps `scipy.signal.get_window` with the window width given in °.

        Window widths are restricted to odd numbers of gridpoints so windows
        can properly be centered on a gridpoint during convolution. The
        returned window array is normalized such that it sums to 1.

        Requires `scipy`.
        """
        from scipy.signal import get_window
        # Convert width to gridpoints
        width = round(width / self.dlon)
        width = width if width % 2 == 1 else width + 1
        window = get_window(window, width, fftbins=False)
        return window / np.sum(window)

    def filter_meridional(self, field, window, width=None):
        """Filter the input in meridional direction with the given window.

        - `field` is the input signal and can be 1- or 2-dimensional.
        - If `width` is None, `window` must be gridded window (1D array) used
          for the convolution operation. Otherwise `window` and `width` are
          given to `Grid.get_filter_window` to obtain a window function.

        Requires `scipy`.
        """
        from scipy.ndimage import convolve
        if width is not None:
            window = self.get_filter_window(window, width)
        # Use symmetrical boundary condition
        if field.ndim == 1:
            assert field.size == self.nlat
            return convolve(field, window, mode="reflect")
        elif field.ndim == 2:
            assert field.shape[0] == self.nlat
            return convolve(field, window[:,None], mode="reflect")
        else:
            raise ValueError("input field must be 1- or 2-dimensional")
            

    def filter_zonal(self, field, window, width=None):
        """Filter the input in zonal direction with the given window.

        - `field` is the input signal and can be 1- or 2-dimensional.
        - If `width` is None, `window` must be gridded window (1D array) used
          for the convolution operation. Otherwise `window` and `width` are
          given to `Grid.get_filter_window` to obtain a window function.

        Requires `scipy`.
        """
        from scipy.ndimage import convolve
        if width is not None:
            window = self.get_filter_window(window, width)
        # Use periodic boundary condition (only makes sense if 
        if field.ndim == 1:
            assert field.size == self.nlon
            return convolve(field, window, mode="wrap")
        elif field.ndim == 2:
            assert field.shape[1] == self.nlon
            return convolve(field, window[None,:], mode="wrap")
        else:
            raise ValueError("input field must be 1- or 2-dimensional")



class GridRegionIndexer:

    def __init__(self, grid):
        self._grid = grid

    def __getitem__(self, selection):
        # Non-tuple selections apply to latitude only
        if not isinstance(selection, tuple):
            selection = selection, slice(None, None)
        # Selecting a 2-dimensional region
        if len(selection) != 2:
            raise IndexError("too many dimensions in region selection")
        # Compute the indices that extract the selected region in each
        # dimension (only rectangular, axis-aligned regions possible)
        lat_indices = self._get_lat_indices(selection[0])
        lon_indices = self._get_lon_indices(selection[1])
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
            return indices[self._grid.lats <= hi]
        elif hi is None:
            return indices[lo <= self._grid.lats]
        else:
            return indices[(min(lo, hi) <= self._grid.lats) & (self._grid.lats <= max(lo, hi))]

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
            return indices[self._grid.lons <= hi]
        elif hi is None:
            return indices[lo <= self._grid.lons]
        elif hi < lo:
            lo_mask = lo <= self._grid.lons
            return np.roll(indices[(self._grid.lons <= hi) | lo_mask], np.count_nonzero(lo_mask))
        else:
            return indices[(lo <= self._grid.lons) & (self._grid.lons <= hi)]



class GridRegion:
    
    def __init__(self, grid, lat_indices, lon_indices):
        self._grid = grid
        self._lon_indices = np.require(lon_indices, dtype=int)
        self._lat_indices = np.require(lat_indices, dtype=int)
        assert self._lat_indices.ndim == 1 and self._lat_indices.size <= grid.shape[0]
        assert self._lon_indices.ndim == 1 and self._lon_indices.size <= grid.shape[1]

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
    def lats(self):
        """Latitudes of the region."""
        return self._grid.lats[self._lat_indices]

    @property
    def lons(self):
        """Longitudes of the region.
        
        If the region crosses the 0° meridian, these will not be monotonic. If
        you need a monotonic longitude coordinate, e.g. for plotting, use
        `lons_mono`, where lontitudes left of the 0° are reduced by 360°.
        """
        return self._grid.lons[self._lon_indices]

    @property
    def lons_mono(self):
        """Longitudes of the region, monotonic even for regions crossing 0°"""
        lons = self.lons
        jump = np.argwhere(np.diff(lons) < 0)
        assert 0 <= jump.size <= 1
        if jump.size == 1:
            lons[:jump[0,0]+1] -= 360.
        return lons

    @property
    def gridpoint_area(self):
        return self.extract(self._grid.gridpoint_area)

    def mean(self, field):
        return self._grid.mean(field, region=self)

