"""
Tabulated radiative cooling curves.

Builds piecewise power-law cooling-curve parameters, converted to code units,
from

* the published Schure et al. (2009) high-temperature cooling table (which also
  includes the Dalgarno & McCray 1972 low-temperature curve), and
* AthenaK's ISM cooling curve (Koyama & Inutsuka 2002 below 10^4.2 K, the
  Schure et al. SPEX table, and the CGOLS power-law fit above 10^8.15 K), which
  extends far below 10^4 K for thermal-instability setups.

NOTE: The rate tables feed the explicit and implicit cooling updates of
``_cooling.py``. The Townsend (2009) coefficients ``Y_table`` are filled in for
completeness; the exact-integration scheme that would use them is not wired in.
"""

# jax
import jax.numpy as jnp

# numerics
import numpy as np

# units and constants
from astropy import units as u
import astropy.constants as c

# astronomix containers
from astronomix._modules._cooling.cooling_options import PiecewisePowerLawParams
from astronomix.units.unit_helpers import CodeUnits

# astronomix functions
from astronomix._modules._cooling._cooling import get_effective_molecular_weights


def _townsend_y_coefficients(
    temperature_table: jnp.ndarray,
    cooling_rate_table: jnp.ndarray,
    slopes: jnp.ndarray,
) -> jnp.ndarray:
    """
    Townsend (2009) temporal-evolution coefficients Y_k of a piecewise power law
    (Eq. A6), accumulated downwards from Y_N = 0 at the top of the table.

    Args:
        temperature_table: The bin-edge temperatures T_k (ascending).
        cooling_rate_table: The cooling rates Lambda_k at the bin edges.
        slopes: The power-law slope alpha_k of each bin.

    Returns:
        The coefficients Y_k at the bin edges.
    """
    Y_table = jnp.zeros_like(temperature_table)
    Lambda_N = cooling_rate_table[-1]
    T_N = temperature_table[-1]
    for k in range(len(slopes) - 1, -1, -1):
        T_k = temperature_table[k]
        T_k1 = temperature_table[k + 1]
        Lambda_k = cooling_rate_table[k]
        alpha_k = slopes[k]

        Y_table = Y_table.at[k].set(
            Y_table[k + 1] - jnp.where(
                alpha_k != 1.0,
                1 / (1 - alpha_k) * Lambda_N / Lambda_k * T_k / T_N
                * (1 - (T_k / T_k1) ** (alpha_k - 1)),
                Lambda_N / Lambda_k * T_k / T_N * jnp.log(T_k / T_k1),
            )
        )
    return Y_table


def schure_cooling(
    code_units: CodeUnits,
) -> PiecewisePowerLawParams:
    """
    Build piecewise power-law cooling parameters from the Schure 2009 table.

    Args:
        code_units: The code-unit system used to convert the tabulated
            temperatures and cooling rates from physical to code units.

    Returns:
        A :class:`PiecewisePowerLawParams` holding the log10 temperature and
        cooling-rate tables, the per-bin power-law slopes, the Townsend
        temporal-evolution coefficients and the reference temperature, all in
        code units.
    """

    # High-temperature cooling table from Schure et al. (2009),
    # https://arxiv.org/pdf/0909.5204. That paper also includes the
    # low-temperature cooling curve from Dalgarno & McCray (1972).

    # Tabulated temperatures in Kelvin.
    log10_T = np.array([
        3.80, 3.84, 3.88, 3.92, 3.96, 4.00, 4.04, 4.08, 4.12, 4.16,
        4.20, 4.24, 4.28, 4.32, 4.36, 4.40, 4.44, 4.48, 4.52, 4.56,
        4.60, 4.64, 4.68, 4.72, 4.76, 4.80, 4.84, 4.88, 4.92, 4.96,
        5.00, 5.04, 5.08, 5.12, 5.16, 5.20, 5.24, 5.28, 5.32, 5.36,
        5.40, 5.44, 5.48, 5.52, 5.56, 5.60, 5.64, 5.68, 5.72, 5.76,
        5.80, 5.84, 5.88, 5.92, 5.96, 6.00, 6.04, 6.08, 6.12, 6.16,
        6.20, 6.24, 6.28, 6.32, 6.36, 6.40, 6.44, 6.48, 6.52, 6.56,
        6.60, 6.64, 6.68, 6.72, 6.76, 6.80, 6.84, 6.88, 6.92, 6.96,
        7.00, 7.04, 7.08, 7.12, 7.16, 7.20, 7.24, 7.28, 7.32, 7.36,
        7.40, 7.44, 7.48, 7.52, 7.56, 7.60, 7.64, 7.68, 7.72, 7.76,
        7.80, 7.84, 7.88, 7.92, 7.96, 8.00, 8.04, 8.08, 8.12, 8.16
    ])

    T = 10**log10_T

    # convert T to code units
    temperature_in_code_units = (T * u.K * c.k_B / c.m_p).to(
        code_units.code_energy / code_units.code_mass
    )
    T = jnp.array(temperature_in_code_units.value)

    log10_T = jnp.log10(T)

    reference_temperature = T[-1]

    # Lambda in erg cm^3 / s
    log10_Lambda = np.array([
        -25.7331, -25.0383, -24.4059, -23.8288, -23.3027,
        -22.8242, -22.3917, -22.0067, -21.6818, -21.4529,
        -21.3246, -21.3459, -21.4305, -21.5293, -21.6138,
        -21.6615, -21.6551, -21.5919, -21.5092, -21.4124,
        -21.3085, -21.2047, -21.1067, -21.0194, -20.9413,
        -20.8735, -20.8205, -20.7805, -20.7547, -20.7455,
        -20.7565, -20.7820, -20.8008, -20.7994, -20.7847,
        -20.7687, -20.7590, -20.7544, -20.7505, -20.7545,
        -20.7888, -20.8832, -21.0450, -21.2286, -21.3737,
        -21.4573, -21.4935, -21.5098, -21.5345, -21.5863,
        -21.6548, -21.7108, -21.7424, -21.7576, -21.7696,
        -21.7883, -21.8115, -21.8303, -21.8419, -21.8514,
        -21.8690, -21.9057, -21.9690, -22.0554, -22.1488,
        -22.2355, -22.3084, -22.3641, -22.4033, -22.4282,
        -22.4408, -22.4443, -22.4411, -22.4334, -22.4242,
        -22.4164, -22.4134, -22.4168, -22.4267, -22.4418,
        -22.4603, -22.4830, -22.5112, -22.5449, -22.5819,
        -22.6177, -22.6483, -22.6719, -22.6883, -22.6985,
        -22.7032, -22.7037, -22.7008, -22.6950, -22.6869,
        -22.6769, -22.6655, -22.6531, -22.6397, -22.6258,
        -22.6111, -22.5964, -22.5816, -22.5668, -22.5519,
        -22.5367, -22.5216, -22.5062, -22.4912, -22.4753,
    ])

    Lambda = 10**log10_Lambda

    # convert Lambda to code units
    code_cooling_rate_unit = (
        code_units.code_energy * code_units.code_length ** 3
        / (code_units.code_time * code_units.code_mass ** 2)
    )
    Lambda = jnp.array(
        (Lambda * u.erg * u.cm ** 3 / u.s / c.m_p ** 2).to(code_cooling_rate_unit).value
    )

    log10_Lambda = jnp.log10(Lambda)

    # piecewise power law fits in the form
    # Lambda(T) = Lambda_k * (T / T_k)^alpha_k for T_k <= T < T_{k+1}
    alpha = (log10_Lambda[1:] - log10_Lambda[:-1]) / (log10_T[1:] - log10_T[:-1])

    Y_table = _townsend_y_coefficients(T, Lambda, alpha)

    return PiecewisePowerLawParams(
        log10_T_table=log10_T,
        log10_Lambda_table=log10_Lambda,
        alpha_table=alpha,
        Y_table=Y_table,
        reference_temperature=reference_temperature,
    )


def athenak_ism_cooling(
    code_units: CodeUnits,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    mu_athena: float = 0.618,
    log10_T_min: float = 1.0,
    log10_T_max: float = 9.0,
) -> PiecewisePowerLawParams:
    """
    AthenaK's ISM cooling curve as piecewise power-law parameters.

    Reproduces ``src/srcterms/ismcooling.hpp`` of mainline AthenaK exactly:
    Koyama & Inutsuka (2002) analytic cooling for log10 T <= 4.2, the Schure
    et al. (2009) SPEX table on 4.2 < log10 T <= 8.15 (0.04 dex rows starting
    at 4.12, linearly interpolated in log-log, i.e. a piecewise power law), and
    the CGOLS power-law fit above. The curve extends down to ``log10_T_min``
    (the cold branch of the thermal instability lives near 180 K) by sampling
    the Koyama-Inutsuka formula.

    Normalisation: AthenaK applies this as de/dt = -n_p^2 Lambda + n_p Gamma
    with the total particle density n_p = rho / (mu_athena * m_u). The
    astronomix kernel multiplies the tabulated curve by n_e * n_H instead, so
    the table is pre-scaled by (mu_e * mu_H / mu_athena^2); the kernel then
    produces AthenaK's volumetric rate.

    Args:
        code_units: The code-unit system.
        hydrogen_mass_fraction: X used by the astronomix kernel prefactor.
        metal_mass_fraction: Z used by the astronomix kernel prefactor.
        mu_athena: The constant mean molecular weight of the AthenaK setup
            (its ``<units> mu``; the Guo-Kim-Stone setup uses 0.618).
        log10_T_min: The lower end of the table, log10 of the temperature in
            Kelvin.
        log10_T_max: The upper end of the table, log10 of the temperature in
            Kelvin.

    Returns:
        ``PiecewisePowerLawParams`` in code units.
    """
    # AthenaK's tabulated Schure SPEX rates (log10 of erg cm^3 / s), 0.04 dex
    # rows from log10 T = 4.12.
    log10_lambda_spex_cgs = np.array([
        -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
        -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
        -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
        -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
        -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
        -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
        -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
        -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
        -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
        -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
        -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
        -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
        -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928])
    spex_log10_temperature = 4.12 + 0.04 * np.arange(len(log10_lambda_spex_cgs))

    # Sample at 0.01 dex below the Koyama-Inutsuka / SPEX switch, where the
    # Koyama-Inutsuka exponential is steep (0.04 dex sampling would leave an
    # interpolation error of ~14% near 8000 K), and at 0.04 dex on the
    # tabulated SPEX range, where the nodes are exact.
    log10_T = np.concatenate([
        np.arange(log10_T_min, 4.2, 0.01),
        np.arange(4.2, log10_T_max + 1e-9, 0.04),
    ])
    temperature_kelvin = 10.0 ** log10_T

    def ism_cooling_rate_cgs(temperature, log10_temperature):
        """AthenaK's ISM cooling rate Lambda(T) in erg cm^3 / s."""
        # Koyama & Inutsuka (2002) below the SPEX range (AthenaK switches at
        # log10 T = 4.2).
        koyama_inutsuka_rate = (
            2.0e-19 * np.exp(-1.184e5 / (temperature + 1.0e3))
            + 2.8e-28 * np.sqrt(temperature) * np.exp(-92.0 / temperature)
        )
        # The CGOLS fit above the table.
        cgols_rate = 10.0 ** (0.45 * log10_temperature - 26.065)
        spex_rate = 10.0 ** np.interp(
            log10_temperature,
            spex_log10_temperature,
            log10_lambda_spex_cgs,
        )
        return np.where(
            log10_temperature <= 4.2,
            koyama_inutsuka_rate,
            np.where(log10_temperature > 8.15, cgols_rate, spex_rate),
        )

    Lambda_cgs = ism_cooling_rate_cgs(temperature_kelvin, log10_T)

    # Conversions for the astronomix kernel, derived from first principles
    # (unlike ``schure_cooling``, which normalises with m_p). The kernel looks
    # temperatures up at T~ = p * mu / rho (code) and computes
    # dT~/dt = -Lambda~ * (gamma - 1) * rho_code * mu / (mu_e * mu_H).
    # AthenaK evaluates its curve at T = mu_athena * m_u * p / (rho * k_B)
    # (cgs) and applies de/dt = -n_p^2 Lambda with n_p = rho / (mu_athena m_u):
    #   (i)  table row T_K sits at T~ = T_K * k_B * mu / (mu_athena * m_u)
    #        in code (pressure / density) units;
    #   (ii) Lambda~ = Lambda_cgs * mu_e * mu_H * unit_rho^2 * unit_t
    #        / ((mu_athena m_u)^2 * unit_p).
    mu, mu_e, mu_H = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    T = jnp.array(
        (temperature_kelvin * u.K * c.k_B * mu / (mu_athena * c.u))
        .to(code_units.code_pressure / code_units.code_density)
        .value
    )
    unit_density_cgs = (1.0 * code_units.code_density).to(u.g / u.cm ** 3).value
    unit_pressure_cgs = (1.0 * code_units.code_pressure).to(u.erg / u.cm ** 3).value
    unit_time_cgs = (1.0 * code_units.code_time).to(u.s).value
    atomic_mass_unit_grams = c.u.to(u.g).value
    Lambda = jnp.array(
        Lambda_cgs * mu_e * mu_H * unit_density_cgs ** 2 * unit_time_cgs
        / ((mu_athena * atomic_mass_unit_grams) ** 2 * unit_pressure_cgs)
    )

    log10_T_code = jnp.log10(T)
    log10_Lambda = jnp.log10(Lambda)
    alpha = (log10_Lambda[1:] - log10_Lambda[:-1]) / (log10_T_code[1:] - log10_T_code[:-1])

    Y_table = _townsend_y_coefficients(T, Lambda, alpha)

    return PiecewisePowerLawParams(
        log10_T_table=log10_T_code,
        log10_Lambda_table=log10_Lambda,
        alpha_table=alpha,
        Y_table=Y_table,
        reference_temperature=T[-1],
    )
