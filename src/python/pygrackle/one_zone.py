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

    def __init__(self, fc, data=None):
        self.fc = fc
        self.data = data

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

    def evolve(self):
        if self.data is None:
            self.current_time = 0
            self.add_to_data()
        else:
            self.current_time = self.data["time"][-1]

        self.print_status()

        while not self.finished:
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

    def __init__(self, fc, data=None,
                 safety_factor=0.01, include_pressure=True,
                 final_time=None, final_density=None):

        if final_time is None and final_density is None:
            raise RuntimeError(
                "Must specify either final_time or final_density.")

        super().__init__(fc, data=data)
        self.include_pressure = include_pressure
        self.safety_factor = safety_factor
        self.final_time = final_time
        self.final_density = final_density

        # Set units of gravitational constant
        my_chemistry = fc.chemistry_data
        self.gravitational_constant = (
            4.0 * np.pi * gravitational_constant_cgs *
            my_chemistry.density_units * my_chemistry.time_units**2)

        # some constants for the analytical free-fall solution
        self.freefall_time_constant = \
          np.power(((32. * self.gravitational_constant) /
                    (3. * np.pi)), 0.5)

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
        dt_ff = self.safety_factor * \
          np.power(((3. * np.pi) /
                    (32. * self.gravitational_constant *
                     fc["density"][0])), 0.5)

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
        force_factor = self.data["force_factor"][-1]

        # calculate new density from altered free-fall solution
        new_density = np.power(
            (np.power(fc["density"][0], -0.5) -
             (0.5 * self.freefall_time_constant * self.dt *
              np.power((1 - force_factor), 0.5))), -2)
        factor = new_density / fc["density"][0]

        self.scale_density_fields(factor)

        # now update energy for adiabatic heating from collapse
        fc["energy"][0] += (my_chemistry.Gamma - 1.) * fc["energy"][0] * \
            self.freefall_time_constant * \
            np.power(fc["density"][0], 0.5) * self.dt

    def calculate_collapse_factor(self):
        """
        Compute the new density using the modified
        free-fall collapse as per Omukai et al. (2005).
        """

        data = self.data

        if not self.include_pressure:
            data["force_factor"].append(0)

        # Calculate the effective adiabatic index, dlog(p)/dlog(rho).
        density = data["density"]
        pressure = data["pressure"]

        if len(pressure) < 3:
            data["force_factor"].append(0)
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
            force_factor = 0.6 + 2.5 * (gamma_eff - 1) - \
                6.0 * np.power((gamma_eff - 1.0), 2.)
        else:
            force_factor = 1.0 + 0.2 * (gamma_eff - (4./3.)) - \
                2.9 * np.power((gamma_eff - (4./3.)), 2.)
        force_factor = max(force_factor, 0.0)
        force_factor = min(force_factor, 0.95)

        data["force_factor"].append(force_factor)
