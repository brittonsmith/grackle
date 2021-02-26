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

    _name = None
    verbose = True

    def __init__(self, fc):
        self.fc = fc
        self.data = None

    @abc.abstractmethod
    def calculate_timestep(self):
        pass

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
        data = self.data

        data["time"].append(self.current_time * fc.chemistry_data.time_units)

        for field in fc.density_fields:
            data[field].append(fc[field][0] * fc.chemistry_data.density_units)
        data["energy"].append(fc["energy"][0] * fc.chemistry_data.energy_units)

        fc.calculate_temperature()
        data["temperature"].append(fc["temperature"][0])

        fc.calculate_pressure()
        data["pressure"].append(fc["pressure"][0] * fc.chemistry_data.pressure_units)

        fc.calculate_mean_molecular_weight()
        data["mean_molecular_weight"].append(fc["mean_molecular_weight"][0])

        if fc.chemistry_data.h2_on_dust:
            fc.calculate_dust_temperature()
            data["dust_temperature"].append(fc["dust_temperature"][0])

    def finalize_data(self):
        """
        Turn lists of values into array with proper cgs units.
        """

        data = self.data
        for field in data:
            if field in self.fc.density_fields:
                data[field] = unyt_array(data[field], "g/cm**3")
            elif field == "energy":
                data[field] = unyt_array(data[field], "erg/g")
            elif field == "time":
                data[field] = unyt_array(data[field], "s")
            elif "temperature" in field:
                data[field] = unyt_array(data[field], "K")
            elif field == "pressure":
                data[field] = unyt_array(data[field], "dyne/cm**2")
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

        status = f"Evolve {self._name} - t: {ctime:e} yr, " + \
          f"rho: {cdensity:e} g/cm^3, T: {ctemperature:e} K."
        print (status, flush=True)

    def before_solve_chemistry(self):
        pass

    def evolve(self):
        self.data = defaultdict(list)
        self.current_time = 0
        self.print_status()
        self.add_to_data()

        while not self.finished:
            dt = self.calculate_timestep()

            self.before_solve_chemistry()
            self.fc.solve_chemistry(dt)

            self.current_time += dt

            self.update_quantities()
            self.print_status()
            self.add_to_data()

class CoolingModel(OneZoneModel):
    def __init__(self, fc, safety_factor=0.1,
                 final_time=None, final_temperature=None):

        if final_time is None and final_temperature is None:
            raise RuntimeError(
                "Must specify either final_time or final_temperature.")

        super().__init__(fc)
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

    def calculate_timestep(self):
        self.fc.calculate_cooling_time()
        dt = self.safety_factor * np.abs(self.fc["cooling_time"][0])
        dt = min(dt, self.final_time - self.current_time)
        return dt

class ConstantDensityModel(CoolingModel):
    _name = "constant density"

    def update_quantities(self):
        pass

class ConstantPressureModel(CoolingModel):
    _name = "constant pressure"

    def before_solve_chemistry(self):
        fc = self.fc
        fc.calculate_temperature()
        self.last_temperature = fc["temperature"][0]

    def update_quantities(self):
        fc = self.fc
        my_chemistry = fc.chemistry_data

        fc.calculate_temperature()
        t_ratio = self.last_temperature / fc["temperature"][0]

        for field in fc.density_fields:
            if field == "dark_matter":
                continue
            fc[field][:] *= t_ratio
