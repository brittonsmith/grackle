"""
OneZoneModel class


"""

import abc
from collections import defaultdict
import numpy as np
from unyt import unyt_array

from pygrackle.utilities.physical_constants import \
    gravitational_constant_cgs, \
    sec_per_year

class OneZoneModel(abc.ABC):
    """
    Class for running one-zone models with a pygrackle fluid container.
    """

    name = None
    verbose = True

    def __init__(self, fc, data=None, external_data=None):
        self.fc = fc
        self.data = data
        self.external_data = external_data

    def calculate_timestep(self):
        self.fc.calculate_cooling_time()
        dt = self.safety_factor * np.abs(self.fc["cooling_time"][0])
        dt = min(dt, self.remaining_time)
        return dt

    @property
    def remaining_time(self):
        if self.final_time is None:
            return np.inf
        return self.final_time - self.current_time

    @abc.abstractproperty
    def finished(self):
        pass

    @abc.abstractmethod
    def update_quantities(self):
        pass

    def update_external_fields(self):
        if self.external_data is None:
            return

        edata = self.external_data
        time = edata["time"]
        itime = np.digitize(self.current_time, time) - 1
        if itime < 0 or itime >= time.size - 1:
            return

        efields = [field for field in edata if field != "time"]
        new_fields = {}
        for field in efields:
            fdata = edata[field]
            if fdata[itime] <= 0 or fdata[itime+1] <= 0:
                new_fields[field] = fdata[itime]
            else:
                fdata = np.log(fdata)
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

        for field in fc.density_fields:
            data[field].append(fc[field][0])
        data["energy"].append(fc["energy"][0])

        fc.calculate_gamma()
        data["gamma"].append(fc["gamma"][0])

        fc.calculate_temperature()
        data["temperature"].append(fc["temperature"][0])

        fc.calculate_pressure()
        data["pressure"].append(fc["pressure"][0])

        fc.calculate_mean_molecular_weight()
        data["mean_molecular_weight"].append(fc["mean_molecular_weight"][0])

        if fc.chemistry_data.h2_on_dust:
            fc.calculate_dust_temperature()
            data["dust_temperature"].append(fc["dust_temperature"][0])

    def finalize_data(self):
        """
        Turn lists of values into array with proper cgs units.
        """

        fc = self.fc
        data = self.data

        for field in data:
            if field in fc.density_fields:
                data[field] = fc.chemistry_data.density_units * \
                  unyt_array(data[field], "g/cm**3")
            elif field == "energy":
                data[field] = fc.chemistry_data.energy_units * \
                  unyt_array(data[field], "erg/g")
            elif field == "time":
                data[field] = fc.chemistry_data.time_units * \
                  unyt_array(data[field], "s")
            elif "temperature" in field:
                data[field] = unyt_array(data[field], "K")
            elif field == "pressure":
                data[field] = fc.chemistry_data.pressure_units * \
                  unyt_array(data[field], "dyne/cm**2")
            else:
                data[field] = np.array(data[field])
        return data

    def print_status(self):
        if not self.verbose:
            return

        fc = self.fc
        my_chemistry = fc.chemistry_data
        ctime = self.current_time * my_chemistry.time_units / sec_per_year
        cdensity = fc["density"][0] * my_chemistry.density_units
        ctemperature = fc["temperature"][0]

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
        self.update_external_fields()

    def evolve(self):
        if self.data is None:
            self.current_time = 0
            self.add_to_data()
        else:
            self.current_time = self.data["time"][-1]

        self.print_status()

        while not self.finished:
            self.before_solve_chemistry()

            dt = self.calculate_timestep()

            self.fc.solve_chemistry(dt)

            self.current_time += dt

            self.update_quantities()
            self.print_status()
            self.add_to_data()

class CoolingModel(OneZoneModel):
    def __init__(self, fc, data=None, safety_factor=0.01,
                 final_time=None, final_temperature=None):

        if final_time is None and final_temperature is None:
            raise RuntimeError(
                "Must specify either final_time or final_temperature.")

        super().__init__(fc, data=data)
        self.safety_factor = safety_factor
        self.final_time = final_time
        self.final_temperature = final_temperature

    @property
    def finished(self):
        if self.final_temperature is not None and \
          self.fc["temperature"][0] <= self.final_temperature:
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
        fc.calculate_temperature()
        T2 = fc["temperature"][0]
        fc.calculate_mean_molecular_weight()
        mu2 = fc["mean_molecular_weight"][0]
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

        fc.calculate_temperature()
        T2 = fc["temperature"][0]
        fc.calculate_mean_molecular_weight()
        mu2 = fc["mean_molecular_weight"][0]
        fc.calculate_gamma()
        g2 = fc['gamma'][0]
        factor = (mu2 / mu1) * (rho1 / mu1)**((g1-g2)/(g2-1)) * \
          (T2 / T1)**(1 / (g2 - 1))

        self.scale_density_fields(factor)

class FreeFallModel(OneZoneModel):
    name = "free-fall"

    def __init__(self, fc, data=None, external_data=None,
                 safety_factor=0.01, include_pressure=True,
                 final_time=None, final_density=None):

        if final_time is None and final_density is None:
            raise RuntimeError(
                "Must specify either final_time or final_density.")

        super().__init__(fc, data=data, external_data=external_data)
        self.include_pressure = include_pressure
        self.safety_factor = safety_factor
        self.final_time = final_time
        self.final_density = final_density

    @property
    def gravitational_constant(self):
        """
        Gravitational constant in internal units.
        """
        my_chemistry = self.fc.chemistry_data
        val = 4.0 * np.pi * gravitational_constant_cgs * \
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
          self.fc["density"][0] >= self.final_density:
            return True

        if self.final_time is not None and \
          self.current_time >= self.final_time:
            return True

        return False

    def calculate_timestep(self):
        fc = self.fc
        dt_ff = self.safety_factor / self.freefall_constant / \
          np.sqrt(fc["density"][0])

        fc.calculate_cooling_time()
        dt_cool = self.safety_factor * \
          np.abs(fc["cooling_time"][0])

        dt = min(dt_ff, dt_cool)
        dt = min(dt, self.remaining_time)
        self.dt = dt
        return dt

    def update_quantities(self):
        fc = self.fc
        my_chemistry = fc.chemistry_data

        self.calculate_collapse_factor()
        force_factor = fc["force_factor"][0]

        self.fc.calculate_pressure()
        pressure = self.fc["pressure"][0]
        density = self.fc["density"][0]
        factor = np.sqrt(1 - force_factor) * \
          self.freefall_constant * np.sqrt(density) * self.dt + 1

        # update energy assuming un-altered free-fall collapse
        e_factor = self.freefall_constant * np.sqrt(density) * self.dt + 1

        self.scale_density_fields(factor)

        de = - pressure * (1 - e_factor) / (e_factor * density)
        fc["energy"][0] += de

    def calculate_collapse_factor(self):
        """
        Compute the new density using the modified
        free-fall collapse as per Omukai et al. (2005).
        """

        data = self.data

        if not self.include_pressure:
            self.fc["force_factor"] = np.array([0])
            return

        # Calculate the effective adiabatic index, dlog(p)/dlog(rho).
        density = data["density"]
        pressure = data["pressure"]

        if len(pressure) < 3:
            self.fc["force_factor"] = np.array([0])
            return

        # compute dlog(p) / dlog(rho) using last two timesteps
        gamma_eff = np.log10(pressure[-1] / pressure[-2]) / \
            np.log10(density[-1] / density[-2])

        # compute a higher order derivative if more than two points available
        if len(pressure) > 2:
            gamma_eff += 0.5 * ((np.log10(pressure[-2] / pressure[-3]) /
                                 np.log10(density[-2] / density[-3])) - gamma_eff)

        gamma_eff = min(gamma_eff, 4./3.)

        # Equation 9 of Omukai et al. (2005)
        if gamma_eff < 0.83:
            force_factor = 0.0
        elif gamma_eff < 1.0:
            X = gamma_eff - 1
            force_factor = 0.6 + 2.5 * X - 6.0 * np.power(X, 2.)
        else:
            X = gamma_eff - 4/3
            force_factor = 1.0 + 0.2 * X - 2.9 * np.power(X, 2.)
        force_factor = max(force_factor, 0.0)
        force_factor = min(force_factor, 0.95)

        self.fc["force_factor"] = np.array([force_factor])

class MinihaloModel(FreeFallModel):
    name = "minihalo"
    use_dark_matter = False

    def __init__(self, fc, data=None, external_data=None,
                 safety_factor=0.01, include_pressure=True,
                 final_time=None, final_density=None,
                 initial_radius=None, gas_mass=None):

        self.initial_radius = initial_radius
        self.gas_mass = gas_mass

        super().__init__(fc, data=data,
                 external_data=external_data,
                 safety_factor=safety_factor,
                 include_pressure=include_pressure,
                 final_time=final_time,
                 final_density=final_density)

    @property
    def finished(self):
        if self.gas_mass is not None:
            ### Bonnor-Ebert Mass constant
            a = 1.67
            b = (225 / (32 * np.sqrt(5 * np.pi))) * a**-1.5
            fc = self.fc
            fc.calculate_mean_molecular_weight()
            fc.calculate_pressure()
            p = fc["pressure"][0]
            cs = np.sqrt(fc["mean_molecular_weight"][0] * p / fc["density"][0])
            m_BE = (b * (cs**4 / self.gravitational_constant**1.5) * p**-0.5)
            if self.gas_mass >= m_BE:
                return True

        if self.final_density is not None and \
          self.fc["density"][0] >= self.final_density:
            return True

        if self.final_time is not None and \
          self.current_time >= self.final_time:
            return True

        return False

    @property
    def current_radius(self):
        return self.initial_radius * (self.data["density"][0] /
                                      self.fc["density"][0])**(1/3)

    def update_quantities(self):
        fc = self.fc
        my_chemistry = fc.chemistry_data

        fc.calculate_pressure()
        fc.calculate_mean_molecular_weight()
        density = fc["density"][0]
        pressure = fc["pressure"][0]
        mu = fc["mean_molecular_weight"][0]
        tcs = 2 * self.current_radius / np.sqrt(mu * pressure / density)
        tff = 1 / (self.freefall_constant * np.sqrt(density))

        # free-fall
        if (tff < tcs):
            self.calculate_collapse_factor()
            force_factor = fc["force_factor"][0]

            total_density = density
            if self.use_dark_matter:
                total_density += fc["dark_matter"][0]

            factor = np.sqrt(1 - force_factor) * \
              self.freefall_constant * np.sqrt(total_density) * self.dt + 1

            # update energy assuming un-altered free-fall collapse
            e_factor = self.freefall_constant * np.sqrt(total_density) * self.dt + 1

        # pressure-dominated
        else:
            external_pressure = self.data["external_pressure"][-1]
            P1 = self.data["pressure"][-1]
            T1 = self.data["temperature"][-1]
            mu1 = self.data["mean_molecular_weight"][-1]
            P2 = max(pressure, external_pressure)
            fc.calculate_temperature()
            T2 = fc["temperature"][0]
            fc.calculate_mean_molecular_weight()
            mu2 = fc["mean_molecular_weight"][0]
            factor = (P2 * T1 * mu2) / (T2 * mu1 * P1)

            e_factor = factor

        self.scale_density_fields(factor)

        de = - pressure * (1 - e_factor) / (e_factor * density)
        fc["energy"][0] += de
