########################################################################
#
# Tests rates calculated when chemistry is initialized.
#
# Written by Ewan Jones, 19/03/2021
#
########################################################################

from ast import parse
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import os

#* Import necessary functions from grackle.
from pygrackle import chemistry_data, setup_fluid_container

#* Import necessary constants from grackle.
from pygrackle.utilities.physical_constants import mass_hydrogen_cgs

def set_parameters(parSet, my_chemistry):
    #Default parameter set.
    if parSet == 1:
        my_chemistry.CaseBRecombination = 0
        my_chemistry.useSavin2004 = 1
        my_chemistry.three_body_rate = 0
        my_chemistry.useOmukai2000 = 1
        my_chemistry.crg_coolExi = 1
        my_chemistry.crg_collIon = 1
        my_chemistry.crg_recomCool = 1
        my_chemistry.crg_bremCool = 1
        my_chemistry.useLique2015 == 1
        my_chemistry.photoelectric_heating = -1
        return True
    #Alternate parameter set.
    elif parSet == 2:
        my_chemistry.CaseBRecombination = 1
        my_chemistry.useSavin2004 = 0
        my_chemistry.three_body_rate = 1
        my_chemistry.useOmukai2000 = 0
        my_chemistry.crg_coolExi = 0
        my_chemistry.crg_collIon = 0
        my_chemistry.crg_recomCool = 0
        my_chemistry.crg_bremCool = 0
        my_chemistry.useLique2015 == 0
        my_chemistry.photoelectric_heating = 2
        return True
    #Default parameter sets with other three_body_rate values.
    elif parSet == 3:
        my_chemistry.CaseBRecombination = 0
        my_chemistry.useSavin2004 = 1
        my_chemistry.three_body_rate = 2
        my_chemistry.useOmukai2000 = 1
        my_chemistry.crg_coolExi = 1
        my_chemistry.crg_collIon = 1
        my_chemistry.crg_recomCool = 1
        my_chemistry.crg_bremCool = 1
        my_chemistry.useLique2015 == 1
        my_chemistry.photoelectric_heating = -1
        return True
    elif parSet == 4:
        my_chemistry.CaseBRecombination = 0
        my_chemistry.useSavin2004 = 1
        my_chemistry.three_body_rate = 3
        my_chemistry.useOmukai2000 = 1
        my_chemistry.crg_coolExi = 1
        my_chemistry.crg_collIon = 1
        my_chemistry.crg_recomCool = 1
        my_chemistry.crg_bremCool = 1
        my_chemistry.useLique2015 == 1
        my_chemistry.photoelectric_heating = -1
        return True
    elif parSet == 5:
        my_chemistry.CaseBRecombination = 0
        my_chemistry.useSavin2004 = 1
        my_chemistry.three_body_rate = 4
        my_chemistry.useOmukai2000 = 1
        my_chemistry.crg_coolExi = 1
        my_chemistry.crg_collIon = 1
        my_chemistry.crg_recomCool = 1
        my_chemistry.crg_bremCool = 1
        my_chemistry.useLique2015 == 1
        my_chemistry.photoelectric_heating = -1
        return True
    elif parSet == 6:
        my_chemistry.CaseBRecombination = 0
        my_chemistry.useSavin2004 = 1
        my_chemistry.three_body_rate = 5
        my_chemistry.useOmukai2000 = 1
        my_chemistry.crg_coolExi = 1
        my_chemistry.crg_collIon = 1
        my_chemistry.crg_recomCool = 1
        my_chemistry.crg_bremCool = 1
        my_chemistry.useLique2015 == 1
        my_chemistry.photoelectric_heating = -1
        return True
    else:
        return False



def test_rate_initialisation():
    """
    Test that the rate tables are initialized correctly.
    If writeHDF5 is set to true, all rate coefficients calculated in the initialisation routine will 
    be saved into a hdf5 file. This should be used for testing and is not meant for frequent use.
    """

    #* Navigate to the directory where the file is located.
    filePath = os.path.abspath(__file__)
    dirPath = os.path.dirname(filePath)
    os.chdir(dirPath)

    #* Initialise chemistry_data instance
    my_chemistry = chemistry_data()

    #* Set parameters
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 0
    my_chemistry.primordial_chemistry = 1
    my_chemistry.metal_cooling = 0
    my_chemistry.UVbackground = 0
    my_chemistry.comoving_coordinates = 0
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0
    my_chemistry.density_units = mass_hydrogen_cgs
    my_chemistry.length_units = 1.0
    my_chemistry.time_units = 1.0
    my_chemistry.velocity_units = my_chemistry.length_units / my_chemistry.time_units
    
    #* List of all rate variable names which will be checked.
    rateNames = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13",
                 "k14", "k15", "k16","k17", "k18", "k19", "k20", "k21", "k22", "k23", "k24", "k25",
                 "k26", "k27", "k28", "k29", "k30", "k31", "k50", "k51", "k52", "k53", "k54", "k55",
                 "k56", "k57", "k58", "n_cr_n", "n_cr_d1", "n_cr_d2", "ceHI", "ceHeI", "ceHeII",
                 "ciHI", "ciHeI", "ciHeIS", "ciHeII", "reHII", "reHeII1", "reHeII2","reHeIII", "brem",
                 "hyd01k", "h2k01", "vibh", "roth", "rotl", "HDlte", "HDlow","cieco", "GAHI", "GAH2",
                 "GAHe", "GAHp", "GAel", "H2LTE", "k13dd", "h2dust"]
    
    #* List of all parameter set identifies which will be checked.
    parameterSets = ["1", "2", "3", "4", "5", "6"]

    #* Write initialised rates to hdf5 file
    #If file already exists delete it so that the new one can be created.
    if os.path.exists("initialised_rates.h5"):
        os.remove("initialised_rates.h5")
    f = h5py.File("initialised_rates.h5", "w-")

    #Calculate rate coefficients for each parameter set.
    for parSet in parameterSets:
        #Create fluid container for given parameter set. This initialises the rate coefficients.
        set_parameters(int(parSet), my_chemistry)
        my_fluidContainer = setup_fluid_container(my_chemistry, temperature=np.logspace(4.5, 9, 200),
                            converge=True, tolerance=1e-6, max_iterations=np.inf)

        #Save each rate coefficient and parameter set it came from.
        for rate_key in rateNames:
            f.create_dataset(rate_key + "_" + parSet, data=getattr(my_chemistry, rate_key))

    f.close()
    print("\n\n\n", os.getcwd(), __file__, "\n\n\n")

    #* Compare rates with the correct ones which are stored and check they are in agreement.
    correctRates = h5py.File("example_answers/correct_rates.h5", "r")
    initialisedRates = h5py.File("initialised_rates.h5", "r")
    for rateName in rateNames:
        for parSet in parameterSets:
            #Get rate name by compunding with the parameter set.
            rate_name = rateName + "_" + parSet
            #Check these three rates to a different tolerance as their tiny values will have slight
            #difference when compared to the old code.
            if rate_name in ["ciHI", "ciHeI", "ciHeII"]:
                largeValuePresent = 0
                with open(f"{rate_name}_discrepancies.txt", "w+") as f:
                    f.write(f'\n --------{rate_name}--------- \n')
                    for i, correctRate in enumerate(correctRates[rate_name]):
                        initialisedRate = initialisedRates[rate_name][i]
                        if np.isclose(correctRate, initialisedRate, atol=1e-10) == False:
                            dif = abs(correctRate - initialisedRate) / correctRate
                            if rate_name == "ciHI":
                                corrRate = "k1"
                            elif rate_name == "ciHeI":
                                corrRate = "k3"
                            elif rate_name == "ciHeII":
                                corrRate = "k5"
                            if correctRates[corrRate][i] == 1e-20:
                                istiny = 'tiny'
                            else:
                                largeValuePresent = 1
                                istiny = 'large'
                            f.write(f'{i}: {corrRate} is {istiny}    {correctRate}   {initialisedRate}   {dif} \n')
                    if largeValuePresent == 1:
                        modifier = "one or more"
                    else:
                        modifier = "no"
                    f.write(f"\n'\n -----There are {modifier} large values present----- \n\n")
                
            #Check rates agree to what we deem is an acceptable relative tolerance.
            else:
                assert np.allclose(correctRates[rate_name], initialisedRates[rate_name], atol=1e-10),\
                                    f"Rate Coefficient {rate_name} does not agree."


test_rate_initialisation()


