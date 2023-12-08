cdef extern from "grackle_macros.h":
    cdef int GRACKLE_FAIL_VALUE "FAIL"

cdef extern from "grackle_types.h":
    # This does not need to be exactly correct, only of the right basic type
    ctypedef float gr_float

cdef extern from "grackle_chemistry_data.h":
    ctypedef struct c_chemistry_data "chemistry_data":
        # no need to declare the members since there is no cython code that
        # directly accesses the struct members (dynamic api is used instead)
        pass

    ctypedef struct c_chemistry_data_storage "chemistry_data_storage":
        double *k1
        double *k2
        double *k3
        double *k4
        double *k5
        double *k6

        double *k7
        double *k8
        double *k9
        double *k10
        double *k11
        double *k12
        double *k13
        double *k14
        double *k15
        double *k16
        double *k17
        double *k18
        double *k19
        double *k20
        double *k21
        double *k22
        double *k23
        double *k13dd

        double k24
        double k25
        double k26

        double k27
        double k28
        double k29
        double k30
        double k31

        double *k50
        double *k51
        double *k52
        double *k53
        double *k54
        double *k55
        double *k56

        double *k57
        double *k58

        double *h2dust

        double *n_cr_n
        double *n_cr_d1
        double *n_cr_d2

        double *ceHI
        double *ceHeI
        double *ceHeII
        double *ciHI
        double *ciHeI
        double *ciHeIS
        double *ciHeII
        double *reHII
        double *reHeII1
        double *reHeII2
        double *reHeIII
        double *brem
        double comp
        double comp_xray
        double temp_xray

        double piHI
        double piHeI
        double piHeII

        double crsHI
        double crsHeI
        double crsHeII

        double *hyd01k
        double *h2k01
        double *vibh
        double *roth
        double *rotl
        double *GP99LowDensityLimit
        double *GP99HighDensityLimit

        double *GAHI
        double *GAH2
        double *GAHe
        double *GAHp
        double *GAel

        double *H2LTE

        double *HDlte
        double *HDlow

        double *cieco

        double gammah

        double *regr

        double gamma_isrf

        double *gas_grain
                                
        double hi_avg_crs
        double hei_avg_crs
        double heii_avg_crs

cdef extern from "grackle_types.h":
    ctypedef struct c_code_units "code_units":
      int comoving_coordinates
      double density_units
      double length_units
      double velocity_units
      double time_units
      double a_units
      double a_value

    ctypedef struct c_field_data "grackle_field_data":
      int grid_rank;
      int *grid_dimension;
      int *grid_start;
      int *grid_end;
      gr_float grid_dx;
      gr_float *density;
      gr_float *HI_density;
      gr_float *HII_density;
      gr_float *HM_density;
      gr_float *HeI_density;
      gr_float *HeII_density;
      gr_float *HeIII_density;
      gr_float *H2I_density;
      gr_float *H2II_density;
      gr_float *DI_density;
      gr_float *DII_density;
      gr_float *HDI_density;
      gr_float *e_density;
      gr_float *metal_density;
      gr_float *dust_density;
      gr_float *internal_energy;
      gr_float *x_velocity;
      gr_float *y_velocity;
      gr_float *z_velocity;
      gr_float *volumetric_heating_rate;
      gr_float *specific_heating_rate;
      gr_float *temperature_floor;
      gr_float *RT_heating_rate;
      gr_float *RT_HI_ionization_rate;
      gr_float *RT_HeI_ionization_rate;
      gr_float *RT_HeII_ionization_rate;
      gr_float *RT_H2_dissociation_rate;
      gr_float *H2_self_shielding_length;
      gr_float *H2_custom_shielding_factor;
      gr_float *isrf_habing;

    ctypedef struct c_grackle_version "grackle_version":
      const char* version;
      const char* branch;
      const char* revision;

# define a macro to omit legacy grackle function defined in grackle.h
cdef extern from *:
    """
    #define OMIT_LEGACY_INTERNAL_GRACKLE_FUNC
    """

cdef extern from "grackle_rate_functions.h":
    double k1_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k2_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k3_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k4_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k5_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k6_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k7_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k8_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k9_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k10_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k11_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k12_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k13_rate(double T, double units,  c_chemistry_data *my_chemistry);

    void k13dd_rate(double T, double units, double *k13dd_results, c_chemistry_data *my_chemistry);

    double k14_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k15_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k16_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k17_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k18_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k19_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k20_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k21_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k22_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k23_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k50_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k51_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k52_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k53_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k54_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k55_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k56_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k57_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double k58_rate(double T, double units,  c_chemistry_data *my_chemistry);

    double h2dust_rate(double T, double T_dust, double units, c_chemistry_data *my_chemistry);

    double n_cr_n_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double n_cr_d1_rate(double T, double units,  c_chemistry_data *my_chemistry);
    double n_cr_d2_rate(double T, double units,  c_chemistry_data *my_chemistry);

    double ceHI_rate(double T, double units, c_chemistry_data *my_chemistry);
    double ceHeI_rate(double T, double units, c_chemistry_data *my_chemistry);
    double ceHeII_rate(double T, double units, c_chemistry_data *my_chemistry);

    double ciHeIS_rate(double T, double units, c_chemistry_data *my_chemistry);
    double ciHI_rate(double T, double units, c_chemistry_data *my_chemistry);
    double ciHeI_rate(double T, double units, c_chemistry_data *my_chemistry);
    double ciHeII_rate(double T, double units, c_chemistry_data *my_chemistry);

    double reHII_rate(double T, double units, c_chemistry_data *my_chemistry);
    double reHeII1_rate(double T, double units, c_chemistry_data *my_chemistry);
    double reHeII2_rate(double T, double units, c_chemistry_data *my_chemistry);
    double reHeIII_rate(double T, double units, c_chemistry_data *my_chemistry);

    double brem_rate(double T, double units, c_chemistry_data *my_chemistry);

    double vibh_rate(double T, double units, c_chemistry_data *my_chemistry);
    double hyd01k_rate(double T, double units, c_chemistry_data *my_chemistry);
    double h2k01_rate(double T, double units, c_chemistry_data *my_chemistry);
    double rotl_rate(double T, double units, c_chemistry_data *my_chemistry);
    double roth_rate(double T, double units, c_chemistry_data *my_chemistry);

    double GP99LowDensityLimit_rate(double T, double units, c_chemistry_data *my_chemistry);
    double GP99HighDensityLimit_rate(double T, double units, c_chemistry_data *my_chemistry);
    double GAHI_rate(double T, double units, c_chemistry_data *my_chemistry);
    double GAH2_rate(double T, double units, c_chemistry_data *my_chemistry);
    double GAHe_rate(double T, double units, c_chemistry_data *my_chemistry);
    double GAHp_rate(double T, double units, c_chemistry_data *my_chemistry);
    double GAel_rate(double T, double units, c_chemistry_data *my_chemistry);
    double H2LTE_rate(double T, double units, c_chemistry_data *my_chemistry);
    double HDlte_rate(double T, double units, c_chemistry_data *my_chemistry);

    double HDlow_rate(double T, double units, c_chemistry_data *my_chemistry);
    double cie_thin_cooling_rate(double T);
    double cieco_rate(double T, double units, c_chemistry_data *my_chemistry);

    double gasGrain_rate(double T, double units, c_chemistry_data *my_chemistry);
    double regr_rate(double T, double units, c_chemistry_data *my_chemistry);

    double comp_rate(double units, c_chemistry_data *my_chemistry);
    double gammah_rate(double units, c_chemistry_data *my_chemistry);
    double gamma_isrf_rate(double units, c_chemistry_data *my_chemistry);

cdef extern from "grackle.h":
    int local_initialize_chemistry_parameters(c_chemistry_data *my_chemistry)

    void set_velocity_units(c_code_units *my_units)

    double get_velocity_units(c_code_units *my_units)

    double get_temperature_units(c_code_units *my_units)

    int local_initialize_chemistry_data(c_chemistry_data *my_chemistry,
                                        c_chemistry_data_storage *my_rates,
                                        c_code_units *my_units)

    int* local_chemistry_data_access_int(c_chemistry_data *my_chemistry,
                                         const char* param_name)

    double* local_chemistry_data_access_double(c_chemistry_data *my_chemistry,
                                               const char* param_name)

    char** local_chemistry_data_access_string(c_chemistry_data *my_chemistry,
                                              const char* param_name)

    const char* param_name_int(unsigned int i)

    const char* param_name_double(unsigned int i)

    const char* param_name_string(unsigned int i)

    int c_local_solve_chemistry "local_solve_chemistry"(
                c_chemistry_data *my_chemistry,
                c_chemistry_data_storage *my_rates,
                c_code_units *my_units,
                c_field_data *my_fields,
                double dt_value)

    int c_local_calculate_cooling_time "local_calculate_cooling_time"(
                c_chemistry_data *my_chemistry,
                c_chemistry_data_storage *my_rates,
                c_code_units *my_units,
                c_field_data *my_fields,
                gr_float *cooling_time)

    int c_local_calculate_gamma "local_calculate_gamma"(
                c_chemistry_data *my_chemistry,
                c_chemistry_data_storage *my_rates,
                c_code_units *my_units,
                c_field_data *my_fields,
                gr_float *gamma)

    int c_local_calculate_pressure "local_calculate_pressure"(
                c_chemistry_data *my_chemistry,
                c_chemistry_data_storage *my_rates,
                c_code_units *my_units,
                c_field_data *my_fields,
                gr_float *pressure)

    int c_local_calculate_temperature "local_calculate_temperature"(
                c_chemistry_data *my_chemistry,
                c_chemistry_data_storage *my_rates,
                c_code_units *my_units,
                c_field_data *my_fields,
                gr_float *temperature)

    int c_local_calculate_dust_temperature "local_calculate_dust_temperature"(
                c_chemistry_data *my_chemistry,
                c_chemistry_data_storage *my_rates,
                c_code_units *my_units,
                c_field_data *my_fields,
                gr_float *dust_temperature)

    int c_local_free_chemistry_data "local_free_chemistry_data" (
        c_chemistry_data *my_chemistry,
        c_chemistry_data_storage *my_rates)

    c_grackle_version c_get_grackle_version "get_grackle_version"()
