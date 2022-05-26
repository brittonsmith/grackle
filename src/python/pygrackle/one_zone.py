"""
OneZoneModel class


"""

import abc
from collections import defaultdict
import functools
import numpy as np
from scipy.interpolate import interp1d
from unyt import unyt_array, unyt_quantity
from unyt.unit_registry import UnitRegistry

from pygrackle.utilities.physical_constants import \
    gravitational_constant_cgs, \
    sec_per_year, \
    mass_sun_cgs

_fc_calc_fields = (
    "cooling_time",
    "dust_temperature",
    "gamma",
    "mean_molecular_weight",
    "pressure",
    "temperature",
)

class OneZoneModel(abc.ABC):
    """
    Class for running one-zone models with a pygrackle fluid container.
    """

    name = None
    verbose = True
    stopping_criteria = ()
    _ignore_external = ("time")
    _cmb_in_cooling_time = False

    def __init__(self, fc, data=None, external_data=None,
                 unit_registry=None, event_trigger_fields="all"):

        self.fc = fc
        self.data = data

        if unit_registry is None:
            unit_registry = UnitRegistry()
        self.unit_registry = unit_registry

        if external_data is None:
            external_data = []
        self.external_data = external_data

        for field in self.external_data:
            if field not in self.fc.fields:
                self.fc._setup_fluid(field)

        self.check_stopping_criteria()
        self.prepare_event_times(event_trigger_fields)

    def prepare_event_times(self, fields):
        """
        Create a list of times to make sure we hit.

        These fields need to be in the external data.
        """

        if fields == "all":
            self.events = self.external_data["time"]
            return

        self.events = []
        if fields is None:
            return

        for field in fields:
            data = self.external_data[field]
            onoff = np.where(~((data[:-1] > 0) ^ (data[1:] <= 0)))[0] + 1
            for t in self.external_data["time"][onoff]:
                if t not in self.events:
                    self.events.append(t)

        self.events = np.array(self.events)

    def calculate_cooling_time(self):
        """
        Thin wrapper around fluid container method.

        This will optionally remove the cmb floor.
        """

        fc = self.fc
        cmb = fc.chemistry_data.cmb_temperature_floor
        if not self._cmb_in_cooling_time:
            fc.chemistry_data.cmb_temperature_floor = 0
        fc.calculate_cooling_time()
        fc.chemistry_data.cmb_temperature_floor = cmb

    def get_current_field(self, field, copy=False):
        """
        Return current field values, i.e., from the fluid container.
        """

        fc = self.fc
        size = fc.n_vals

        if field in _fc_calc_fields:
            fname = f"calculate_{field}"
            # check for a method associated with the one-zone model
            cfunc = getattr(self, fname, None)
            # if none, then just call the fluid container method
            if cfunc is None:
                getattr(fc, fname)()
            else:
                getattr(self, fname)()

        data = fc[field]
        if size == 1:
            return data[0]
        if copy:
            return data.copy()
        return data

    _arr = None
    @property
    def arr(self):
        """
        Create a unyt_array using the Arbor's unit registry.
        """
        if self._arr is not None:
            return self._arr
        self._arr = functools.partial(unyt_array,
                                      registry=self.unit_registry)
        return self._arr

    _quan = None
    @property
    def quan(self):
        """
        Create a unyt_quantity using the Arbor's unit registry.
        """
        if self._quan is not None:
            return self._quan
        self._quan = functools.partial(unyt_quantity,
                                       registry=self.unit_registry)
        return self._quan

    def check_stopping_criteria(self):
        canstop = False
        for cri in self.stopping_criteria:
            if getattr(self, cri, None) is not None:
                canstop = True
                break
        if not canstop:
            raise RuntimeError(
                f"Must specify at least one stopping criteria: {self.stopping_criteria}.")

    def calculate_timestep(self):
        dt = self.safety_factor * np.abs(self.get_current_field("cooling_time"))
        dt = min(dt, self.remaining_time)
        return dt

    @property
    def external_fields(self):
        return [field for field in self.external_data
                if field not in self._ignore_external]

    @property
    def remaining_time(self):
        """
        Return time to nearest event or final time.
        """
        if len(self.events):
            upcoming = self.events > self.current_time
            if not upcoming.any():
                t_next = np.inf
            else:
                t_next = (self.events[upcoming] - self.current_time).min()
        else:
            t_next = np.inf

        if self.final_time is None:
            final_time = np.inf
        else:
            final_time = self.final_time

        rt = final_time - self.current_time

        return min(t_next, rt)

    @abc.abstractproperty
    def finished(self):
        pass

    @abc.abstractmethod
    def update_quantities(self):
        pass

    def update_external_fields(self):
        if not self.external_data:
            return

        edata = self.external_data
        time = edata["time"]
        itime = np.digitize(self.current_time, time) - 1
        if itime < 0 or itime >= time.size - 1:
            return

        new_fields = {}
        for field in self.external_fields:
            fdata = edata[field]
            if fdata[itime] < 0 or fdata[itime+1] < 0:
                new_fields[field] = fdata[itime]
            else:
                fdata = np.log(np.clip(fdata, a_min=1e-50, a_max=np.inf))
                slope = (fdata[itime+1] - fdata[itime]) / (time[itime+1] - time[itime])
                new_fields[field] = \
                  np.exp(slope * (self.current_time - time[itime]) + fdata[itime])

        for field in new_fields:
            self.fc[field][:] = new_fields[field]

    def add_to_data(self):
        """
        Add current fluid container values to the data structure.
        """

        fc = self.fc
        if self.data is None:
            self.data = defaultdict(list)
        data = self.data

        data["time"].append(self.current_time)

        cfields = [
            "cooling_time",
            "gamma",
            "mean_molecular_weight",
            "pressure",
            "temperature"
        ]
        if fc.chemistry_data.h2_on_dust:
            cfields.append("dust_temperature")

        all_fields = \
          fc.density_fields + \
          self.external_fields + \
          cfields + \
          ["energy"]

        for field in all_fields:
            data[field].append(self.get_current_field(field, copy=True))

    def finalize_data(self):
        """
        Turn lists of values into array with proper cgs units.
        """

        fc = self.fc
        my_chemistry = fc.chemistry_data
        data = self.data

        for field in data:
            if field in fc.density_fields:
                data[field] = my_chemistry.density_units * \
                  self.arr(data[field], "g/cm**3")
            elif field == "energy":
                data[field] = my_chemistry.energy_units * \
                  self.arr(data[field], "erg/g")
            elif "time" in field:
                data[field] = my_chemistry.time_units * \
                  self.arr(data[field], "s")
            elif "temperature" in field:
                data[field] = self.arr(data[field], "K")
            elif "pressure" in field:
                data[field] = my_chemistry.pressure_units * \
                  self.arr(data[field], "dyne/cm**2")
            elif "radius" in field:
                data[field] = my_chemistry.length_units * \
                  self.arr(data[field], "cm")
            elif "mass" in field:
                data[field] = my_chemistry.density_units * my_chemistry.length_units**3 * \
                  self.arr(data[field], "g")
            elif "velocity" in field:
                data[field] = my_chemistry.velocity_units * \
                  self.arr(data[field], "cm/s")
            else:
                data[field] = np.array(data[field])
        return data

    def print_status(self):
        if not self.verbose:
            return

        my_chemistry = self.fc.chemistry_data
        ctime = self.current_time * my_chemistry.time_units / sec_per_year
        cdensity = self.get_current_field("density") * my_chemistry.density_units
        ctemperature = self.get_current_field("temperature")

        status = f"Evolve {self.name} - t: {ctime:e} yr, " + \
          f"rho: {cdensity:e} g/cm^3, T: {ctemperature:e} K."
        print (status, flush=True)

    def scale_density_fields(self, factor, exclude=None):
        fc = self.fc

        if exclude is None:
            exclude = ["dark_matter"]

        for field in fc.density_fields:
            if field in exclude:
                continue
            fc[field][:] *= factor

    def before_solve_chemistry(self):
        pass

    def evolve(self):
        if self.data is None:
            self.current_time = 0
            self.update_external_fields()
            self.add_to_data()
        else:
            self.current_time = self.data["time"][-1]

        self.print_status()

        while not self.finished:
            self.update_external_fields()
            self.before_solve_chemistry()

            dt = self.calculate_timestep()

            self.fc.solve_chemistry(dt)

            self.current_time += dt

            self.update_quantities()
            self.print_status()
            self.add_to_data()

class CoolingModel(OneZoneModel):
    stopping_criteria = ("final_time", "final_temperature")

    def __init__(self, fc, data=None, safety_factor=0.01,
                 final_time=None, final_temperature=None):

        self.safety_factor = safety_factor
        self.final_time = final_time
        self.final_temperature = final_temperature
        super().__init__(fc, data=data)

    @property
    def finished(self):
        if self.final_temperature is not None and \
          np.min(self.get_current_field("temperature")) <= self.final_temperature:
            return True

        if self.final_time is not None and \
          self.current_time >= self.final_time:
            return True

        return False

class ConstantDensityModel(CoolingModel):
    name = "constant density"

    def update_quantities(self):
        pass

class ConstantPressureModel(CoolingModel):
    name = "constant pressure"

    def update_quantities(self):
        fc = self.fc
        my_chemistry = fc.chemistry_data

        T1 = self.data["temperature"][-1]
        mu1 = self.data["mean_molecular_weight"][-1]
        T2 = self.get_current_field("temperature")
        mu2 = self.get_current_field("mean_molecular_weight")
        factor = (T1 * mu2) / (T2 * mu1)

        self.scale_density_fields(factor)

class ConstantEntropyModel(ConstantPressureModel):
    name = "constant entropy"

    def update_quantities(self):
        fc = self.fc
        my_chemistry = fc.chemistry_data

        T1 = self.data["temperature"][-1]
        mu1 = self.data["mean_molecular_weight"][-1]
        rho1 = self.data["density"][-1]
        g1 = self.data["gamma"][-1]

        T2 = self.get_current_field("temperature")
        mu2 = self.get_current_field("mean_molecular_weight")
        g2 = self.get_current_field("gamma")
        factor = (mu2 / mu1) * (rho1 / mu1)**((g1-g2)/(g2-1)) * \
          (T2 / T1)**(1 / (g2 - 1))

        self.scale_density_fields(factor)

class FreeFallModel(OneZoneModel):
    name = "free-fall"
    stopping_criteria = ("final_time", "final_density")

    def __init__(self, fc, data=None,
                 external_data=None, unit_registry=None,
                 safety_factor=0.01, include_pressure=True,
                 final_time=None, final_density=None,
                 event_trigger_fields="all"):

        self.include_pressure = include_pressure
        self.safety_factor = safety_factor
        self.final_time = final_time
        self.final_density = final_density
        super().__init__(fc, data=data, external_data=external_data,
                         unit_registry=unit_registry,
                         event_trigger_fields=event_trigger_fields)

    @property
    def gravitational_constant(self):
        """
        Gravitational constant in internal units.
        """
        my_chemistry = self.fc.chemistry_data
        val = gravitational_constant_cgs * \
          my_chemistry.density_units * my_chemistry.time_units**2
        return val

    @property
    def freefall_constant(self):
        """
        Constant used in analytical free-fall solution.
        """
        return np.sqrt((32 * self.gravitational_constant) / (3. * np.pi))

    @property
    def finished(self):
        if self.final_density is not None and \
          np.max(self.get_current_field("density")) >= self.final_density:
            return True

        if self.final_time is not None and \
          self.current_time >= self.final_time:
            return True

        return False

    def calculate_timestep(self):
        fc = self.fc
        dt_ff = self.safety_factor / self.freefall_constant / \
          np.sqrt(self.get_current_field("density"))

        dt_cool = self.safety_factor * \
          np.abs(self.get_current_field("cooling_time"))

        dt = min(np.min(dt_ff), np.min(dt_cool))
        dt = min(dt, self.remaining_time)
        self.dt = dt
        return dt

    def update_quantities(self):
        fc = self.fc
        my_chemistry = fc.chemistry_data

        self.calculate_collapse_factor()
        force_factor = self.get_current_field("force_factor")

        pressure = self.get_current_field("pressure")
        density = self.get_current_field("density")
        val = self.freefall_constant * np.sqrt(density) * self.dt
        factor = np.sqrt(1 - force_factor) * val + 1

        # update energy assuming un-altered free-fall collapse
        e_factor = val + 1

        self.scale_density_fields(factor)

        de = - pressure * (1 - e_factor) / (e_factor * density)
        fc["energy"][:] += de

    def calculate_collapse_factor(self):
        """
        Compute the new density using the modified
        free-fall collapse as per Omukai et al. (2005).
        """

        data = self.data
        force_factor = np.zeros(self.fc.n_vals)
        self.fc["force_factor"] = force_factor

        if not self.include_pressure:
            return

        # Calculate the effective adiabatic index, dlog(p)/dlog(rho).
        density = data["density"]
        pressure = data["pressure"]

        if len(pressure) < 3:
            return

        density = np.array(density[-3:])
        pressure = np.array(pressure[-3:])

        # compute dlog(p) / dlog(rho) using last two timesteps
        gamma_eff = np.log10(pressure[-1] / pressure[-2]) / \
            np.log10(density[-1] / density[-2])

        # compute a higher order derivative if more than two points available
        if len(pressure) > 2:
            gamma_eff += 0.5 * ((np.log10(pressure[-2] / pressure[-3]) /
                                 np.log10(density[-2] / density[-3])) - gamma_eff)

        if self.fc.n_vals == 1:
            gamma_eff = np.asarray(gamma_eff)
        np.clip(gamma_eff, a_min=-np.inf, a_max=4/3, out=gamma_eff)

        # Equation 9 of Omukai et al. (2005)
        force_factor[gamma_eff < 0.83] = 0

        f1 = (gamma_eff >= 0.83) & (gamma_eff < 1)
        if f1.any():
            X = gamma_eff[f1] - 1
            force_factor[f1] = 0.6 + 2.5 * X - 6.0 * X**2

        f2 = (gamma_eff >= 1)
        if f2.any():
            X = gamma_eff[f2] - 4/3
            force_factor[f2] = 1.0 + 0.2 * X - 2.9 * X**2

        np.clip(force_factor, a_min=0, a_max=0.95, out=force_factor)

class MinihaloModel(FreeFallModel):
    name = "minihalo"
    stopping_criteria = ("final_time", "final_density", "gas_mass")
    use_dark_matter = False
    _ignore_external = ("time", "radius", "radial_bins")

    def __init__(self, fc, data=None,
                 external_data=None, unit_registry=None,
                 safety_factor=0.01, include_pressure=True,
                 final_time=None, final_density=None,
                 initial_radius=None, gas_mass=None,
                 include_turbulence=True,
                 event_trigger_fields="all", cosmology=None,
                 star_creation_time=None):

        self.initial_radius = initial_radius
        self.gas_mass = gas_mass
        self.include_turbulence = include_turbulence

        super().__init__(fc, data=data,
                 external_data=external_data,
                 unit_registry=unit_registry,
                 safety_factor=safety_factor,
                 include_pressure=include_pressure,
                 final_time=final_time,
                 final_density=final_density,
                 event_trigger_fields=event_trigger_fields)

        self.cosmology = cosmology
        self.initialize_cosmology()
        self.star_creation_time = star_creation_time

    def prepare_event_times(self, fields):
        """
        Create a list of times to make sure we hit.

        These fields need to be in the external data.
        """

        if fields == "all":
            self.events = self.external_data["time"]
            return

        self.events = []
        if fields is None:
            return

        for field in fields:
            data = self.external_data[field]
            onoff = (~((data[:-1, :] > 0) ^ (data[1:, :] <= 0))).sum(axis=1)
            ievent = np.where(onoff)[0] + 1
            for t in self.external_data["time"][ievent]:
                if t not in self.events:
                    self.events.append(t)

        self.events = np.array(self.events)

    def update_external_fields(self):
        if not self.external_data:
            return

        edata = self.external_data
        time = edata["time"]
        itime = np.digitize(self.current_time, time) - 1
        if itime < 0 or itime >= time.size - 1:
            return

        u1 = edata["used_bins"][itime]
        x1 = np.log(edata["radius"][u1])
        u2 = edata["used_bins"][itime+1]
        x2 = np.log(edata["radius"][u2])
        xp = np.log(self.current_radius)

        ikwargs = {"kind": "linear", "fill_value": "extrapolate"}

        new_fields = {}
        for field in self.external_fields:
            fdata = edata[field]

            y1 = np.log(np.clip(edata[field][itime, u1],
                                a_min=1e-50, a_max=np.inf))
            f1 = interp1d(x1, y1, **ikwargs)
            v1 = f1(xp)

            y2 = np.log(np.clip(edata[field][itime+1, u2],
                                a_min=1e-50, a_max=np.inf))
            f2 = interp1d(x2, y2, **ikwargs)
            v2 = f2(xp)

            slope = (v2 - v1) / (time[itime+1] - time[itime])
            new_fields[field] = np.exp(slope * (self.current_time - time[itime]) + v1)

        for field in new_fields:
            self.fc[field][:] = new_fields[field]

    def initialize_cosmology(self):
        if self.cosmology is None:
            self.initial_redshift = None
            self.initial_cosmo_time = None
            return

        my_chemistry = self.fc.chemistry_data
        a_initial = my_chemistry.a_value * my_chemistry.a_units
        self.initial_redshift = 1 / a_initial - 1
        self.initial_cosmo_time = \
          self.cosmology.t_from_z(self.initial_redshift).to('s').d / \
          my_chemistry.time_units

    @property
    def current_redshift(self):
        if self.cosmology is None:
            return None
        cosmo_time = self.fc.chemistry_data.time_units * \
          (self.current_time + self.initial_cosmo_time)
        return self.cosmology.z_from_t(cosmo_time)

    @property
    def finished(self):
        if super().finished:
            return True

        if self.gas_mass is not None:
            m_BE = self.calculate_bonnor_ebert_mass()
            if np.max(self.gas_mass / m_BE) >= 1:
                return True

        return False

    def before_solve_chemistry(self):
        if "metallicity" in self.external_fields:
            self.fc["metal"][:] = self.get_current_field("density") * \
              self.get_current_field("metallicity")
        self.fc.chemistry_data.override_redshift = self.current_redshift

    @property
    def current_radius(self):
        if self.data is None:
            factor = 1
        else:
            factor= (self.data["density"][0] /
                     self.get_current_field("density"))**(1/3)
        return self.initial_radius * factor

    def calculate_hydrostatic_pressure_profile_2(self, itime):
        edata = self.external_data
        # subtract 1 to get the bin just inside the current radius
        iradius = np.digitize(self.current_radius, edata["radial_bins"]) - 1
        used = np.where(edata["used_bins"][itime])[0]
        used = used[used >= iradius]

        # keep inner bin separate
        iinner = used[0]
        # save only bins outside current radius
        used = used[1:]

        my_chemistry = self.fc.chemistry_data
        length_units = my_chemistry.length_units
        density_units = my_chemistry.density_units
        mass_units = density_units * length_units**3

        rbins = edata["radial_bins"] * length_units
        dr = np.diff(rbins)[used]
        r = edata["radius"][used] * length_units
        m_dm_all = edata["dark_matter_mass_enclosed"][itime] * mass_units
        m_dm = m_dm_all[used]

        # "core" values for the central component of the
        # hydrostatic pressure.
        rhoc = self.data["density"][-1] * density_units
        rc = self.current_radius * length_units
        mc = self.gas_mass * mass_units

        slope = np.log(m_dm_all[used[0]] / m_dm_all[iinner]) / \
          np.log(rbins[used[0]] / rbins[iinner])
        m_dmc = np.exp(slope * np.log(rc / rbins[iinner]) +
                       np.log(m_dm_all[iinner]))

        if edata["time"][itime] < self.star_creation_time:
            m_gas = edata["gas_mass_enclosed"][itime, used] * mass_units
            rho_gas = edata["gas_density"][itime][used] * density_units

        else:
            rho_dm = edata["dark_matter"][itime][used] * density_units

            # Assume gas density is at cosmic baryon fraction at the virial radius.
            # At late times, this is roughly true.
            f_gas = self.cosmology.omega_baryon / self.cosmology.omega_matter
            g1 = np.log(rhoc)
            g2 = np.log(rho_dm[-1] * f_gas)
            r1 = np.log(rc)
            r2 = np.log(rbins[used[-1]+1])
            lr = np.log(r)
            slope = (g2 - g1) / (r2 - r1)
            rho_gas = np.exp(slope * (lr - r1) + g1)

            volume = (4 * np.pi / 3) * (rbins[used+1]**3 - rbins[used]**3)
            m_gas = (rho_gas * volume).cumsum() + mc

        # Now add the contribution from the parcel of gas we are following,
        # which should be just inside the central bin.
        drc = rbins[used[0]] - rc
        m_totc = mc + m_dmc
        dpc = gravitational_constant_cgs * m_totc * rhoc * drc / rc**2

        m_tot = m_dm + m_gas
        p_cgs = (gravitational_constant_cgs * m_tot * rho_gas * dr / r**2).sum() + dpc
        return p_cgs / my_chemistry.pressure_units

    def calculate_hydrostatic_pressure_profile(self, itime):
        my_chemistry = self.fc.chemistry_data
        length_units = my_chemistry.length_units
        density_units = my_chemistry.density_units
        mass_units = density_units * length_units**3
        edata = self.external_data

        used = np.where(edata["used_bins"][itime])[0]
        iradius = np.digitize(self.current_radius, edata["radial_bins"])
        r_used = used[used >= iradius.max()]

        # Construct an array consisting of the radii we are following
        # explicitly and the profile bins outside that.
        m_dm = edata["dark_matter_mass_enclosed"][itime, r_used] * mass_units
        r = edata["radius"][r_used] * length_units
        rbins = edata["radial_bins"] * length_units
        dr = np.diff(rbins)[r_used]

        if edata["time"][itime] < self.star_creation_time:
            m_gas = edata["gas_mass_enclosed"][itime, r_used] * mass_units
            rho_gas = edata["gas_density"][itime][r_used] * density_units

        else:
            # Assume gas density is at cosmic baryon fraction at the virial radius.
            # At late times, this is roughly true.
            rho_dm = edata["dark_matter"][itime][r_used] * density_units
            f_gas = self.cosmology.omega_baryon / self.cosmology.omega_matter
            g1 = np.log(rhoc[-1])
            g2 = np.log(rho_dm[-1] * f_gas)
            r1 = np.log(rc[-1])
            r2 = np.log(rbins[r_used[-1]+1])
            lr = np.log(r)
            slope = (g2 - g1) / (r2 - r1)
            rho_gas = np.exp(slope * (lr - r1) + g1)

            volume = (4 * np.pi / 3) * (rbins[r_used+1]**3 - rbins[r_used]**3)
            m_gas = (rho_gas * volume).cumsum() + m_gasc[-1]

        dpc = self.calculate_hydrostatic_dp_parcel(itime)
        # add pressure from radial bins outside the gas parcels
        m_tot = m_dm + m_gas
        dpp = gravitational_constant_cgs * m_tot * rho_gas * dr / r**2
        dp = np.concatenate([dpc, dpp])

        p_cgs = np.flip(np.flip(dp).cumsum())[:self.fc.n_vals]
        return p_cgs / my_chemistry.pressure_units

    def calculate_hydrostatic_pressure(self):
        edata = self.external_data
        time = edata["time"]
        itime = np.digitize(self.current_time, time) - 1

        p1 = np.log(self.calculate_hydrostatic_pressure_profile(itime))
        p2 = np.log(self.calculate_hydrostatic_pressure_profile(itime+1))
        t1 = time[itime]
        t2 = time[itime+1]
        slope = (p2 - p1) / (t2 - t1)
        p = np.exp(slope * (self.current_time - t1) + p1)

        return p

    def update_quantities(self):
        fc = self.fc
        my_chemistry = fc.chemistry_data

        tcs = self.calculate_sound_crossing_time()
        tff = self.calculate_freefall_time()

        pressure = self.get_current_field("pressure", copy=True)

        freefall = tff < tcs
        factor = np.ones(self.fc.n_vals)
        e_factor = np.ones(self.fc.n_vals)

        # free-fall
        if freefall.any():
            self.calculate_collapse_factor()
            force_factor = self.get_current_field("force_factor")[freefall]

            total_density = self.get_current_field("density", copy=True)[freefall]
            if self.use_dark_matter:
                total_density += self.get_current_fields("dark_matter")[freefall]

            val = self.freefall_constant * np.sqrt(total_density) * self.dt
            factor[freefall] = np.sqrt(1 - force_factor) * val + 1

            # update energy assuming un-altered free-fall collapse
            e_factor[freefall] = val + 1

        # pressure-dominated
        prdom = ~freefall
        if (prdom).any():
            hydrostatic_pressure = self.calculate_hydrostatic_pressure()

            P1 = self.data["pressure"][-1][prdom]
            T1 = self.data["temperature"][-1][prdom]
            mu1 = self.data["mean_molecular_weight"][-1][prdom]
            P2 = np.max([pressure[prdom], hydrostatic_pressure[prdom]], axis=0)
            P2 = (P1 + P2) / 2
            fc.calculate_temperature()
            T2 = self.get_current_field("temperature")[prdom]
            mu2 = self.get_current_field("mean_molecular_weight")[prdom]
            factor[prdom] = (P2 * T1 * mu2) / (T2 * mu1 * P1)

            e_factor[prdom] = factor[prdom]

        self.scale_density_fields(factor)

        de = - pressure * (1 - e_factor) / \
          (e_factor * self.get_current_field("density"))
        fc["energy"][:] += de

    def calculate_sound_speed(self):
        fc = self.fc
        my_chemistry = fc.chemistry_data

        density = self.get_current_field("density")
        pressure = self.get_current_field("pressure")
        gamma = self.get_current_field("gamma")

        cs = np.sqrt(gamma * pressure / density)
        if self.include_turbulence:
            v_turb = self.data["turbulent_velocity"][-1]
            cs = np.sqrt(cs**2 + v_turb**2)

        return cs

    def calculate_sound_crossing_time(self):
        cs = self.calculate_sound_speed()
        return 2 * self.current_radius * \
          self.fc.chemistry_data.a_value / cs

    def calculate_freefall_time(self):
        density = self.get_current_field("density", copy=True)
        if self.use_dark_matter:
            density += self.get_current_field("dark_matter")
        return 1 / (self.freefall_constant * np.sqrt(density))

    def calculate_bonnor_ebert_mass(self):
        ### Bonnor-Ebert Mass constant
        a = 1.67
        b = (225 / (32 * np.sqrt(5 * np.pi))) * a**-1.5

        fc = self.fc
        my_chemistry = fc.chemistry_data

        pressure = self.get_current_field("pressure")

        # Convert to CGS because I am tired of
        # messing up cosmological units.
        # Ignore turbulent velocity for cs in BE mass.
        include_turbulence = self.include_turbulence
        self.include_turbulence = False
        cs_cgs = self.calculate_sound_speed() * \
          my_chemistry.velocity_units
        self.include_turbulence = include_turbulence

        G = gravitational_constant_cgs
        pressure_cgs = pressure * my_chemistry.pressure_units
        m_BE = (b * (cs_cgs**4 / G**1.5) * pressure_cgs**-0.5)

        mass_units = my_chemistry.density_units * \
          my_chemistry.length_units**3
        return m_BE / mass_units

    def add_to_data(self):
        super().add_to_data()

        fields = \
          ["freefall_time",
           "sound_crossing_time",
           "bonnor_ebert_mass"]

        for field in fields:
            val = getattr(self, f"calculate_{field}")()
            self.data[field].append(val)

class MinihaloModel1D(MinihaloModel):
    name = "minihalo1d"

    def print_status(self):
        if not self.verbose:
            return

        fc = self.fc
        my_chemistry = fc.chemistry_data

        m_BE = self.calculate_bonnor_ebert_mass()
        ratio = self.gas_mass / m_BE
        index = ratio.argmax()
        index = 0

        cmass = self.gas_mass[index] * my_chemistry.density_units * \
          my_chemistry.length_units**3 / mass_sun_cgs
        ctime = self.current_time * my_chemistry.time_units / sec_per_year / 1e6
        cdensity = fc["density"][index] * my_chemistry.density_units
        ctemperature = fc["temperature"][index]

        status = f"Evolve {self.name} - t: {ctime:8g} Myr, " + \
          f"m: {cmass:8g} Msun, " + \
          f"rho: {cdensity:8g} g/cm^3, T: {ctemperature:8g} K, M/M_BE: {ratio[index]:8g}"
        print (status, flush=True)

    def calculate_hydrostatic_dp_parcel(self, itime):
        """
        Calculate dp from hydrostatic pressure for the gas parcel.

        dp/dr = -G * m_tot * rho_gas / r**2
        """

        my_chemistry = self.fc.chemistry_data
        length_units = my_chemistry.length_units
        density_units = my_chemistry.density_units
        mass_units = density_units * length_units**3
        edata = self.external_data

        used = np.where(edata["used_bins"][itime])[0]

        rhoc = self.data["density"][-1] * density_units
        rc = self.current_radius * length_units
        m_gasc = self.gas_mass * mass_units
        drc = np.gradient(m_gasc) / (4 * np.pi * rhoc * rc**2)

        # Calculate dark matter mass for gas parcels by interpolating
        # from profiles.
        r_all = edata["radius"][used] * length_units
        m_dm_all = edata["dark_matter_mass_enclosed"][itime, used] * mass_units
        m_dm_all = np.clip(m_dm_all, a_min=1e-50, a_max=np.inf)
        x1 = np.log(r_all)
        y1 = np.log(m_dm_all)
        ikwargs = {"kind": "linear", "fill_value": "extrapolate"}
        f1 = interp1d(x1, y1, **ikwargs)
        m_dmc = np.exp(f1(np.log(rc)))
        m_totc = m_gasc + m_dmc

        G = gravitational_constant_cgs
        dpc = G * m_totc * rhoc * drc / rc**2

        return dpc
