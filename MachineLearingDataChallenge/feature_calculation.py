import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import find_peaks
import tsfresh

feature_names = [
    'total_acc__abs_energy_frontside',
    'total_acc__absolute_maximum_frontside',
    'total_acc__absolute_sum_of_changes_frontside',
    'total_acc__benford_correlation_frontside',
    'total_acc__count_above_mean_frontside',
    'total_acc__count_below_mean_frontside',
    'total_acc__fft_aggregated__aggtype_"centroid"_frontside',
    'total_acc__fft_aggregated__aggtype_"variance"_frontside',
    'total_acc__fft_aggregated__aggtype_"skew"_frontside',
    'total_acc__fft_aggregated__aggtype_"kurtosis"_frontside',
    'total_acc__first_location_of_maximum_frontside',
    'total_acc__first_location_of_minimum_frontside',
    'total_acc__has_duplicate_frontside',
    'total_acc__has_duplicate_max_frontside',
    'total_acc__has_duplicate_min_frontside',
    'total_acc__kurtosis_frontside',
    'total_acc__last_location_of_maximum_frontside',
    'total_acc__last_location_of_minimum_frontside',
    'total_acc__length_frontside',
    'total_acc__linear_trend__attr_"pvalue"_frontside',
    'total_acc__linear_trend__attr_"rvalue"_frontside',
    'total_acc__linear_trend__attr_"intercept"_frontside',
    'total_acc__linear_trend__attr_"slope"_frontside',
    'total_acc__linear_trend__attr_"stderr"_frontside',
    'total_acc__longest_strike_above_mean_frontside',
    'total_acc__longest_strike_below_mean_frontside',
    'total_acc__maximum_frontside',
    'total_acc__mean_frontside',
    'total_acc__mean_abs_change_frontside',
    'total_acc__mean_change_frontside',
    'total_acc__mean_second_derivative_central_frontside',
    'total_acc__median_frontside',
    'total_acc__minimum_frontside',
    'total_acc__percentage_of_reoccurring_datapoints_to_all_datapoints_frontside',
    'total_acc__percentage_of_reoccurring_values_to_all_values_frontside',
    'total_acc__ratio_value_number_to_time_series_length_frontside',
    'total_acc__root_mean_square_frontside',
    'total_acc__skewness_frontside',
    'total_acc__standard_deviation_frontside',
    'total_acc__sum_of_reoccurring_data_points_frontside',
    'total_acc__sum_of_reoccurring_values_frontside',
    'total_acc__sum_values_frontside',
    'total_acc__variance_frontside',
    'total_acc__variance_larger_than_standard_deviation_frontside',
    'total_acc__variation_coefficient_frontside',
    'acc_x__abs_energy_frontside',
    'acc_x__absolute_maximum_frontside',
    'acc_x__absolute_sum_of_changes_frontside',
    'acc_x__benford_correlation_frontside',
    'acc_x__count_above_mean_frontside',
    'acc_x__count_below_mean_frontside',
    'acc_x__fft_aggregated__aggtype_"centroid"_frontside',
    'acc_x__fft_aggregated__aggtype_"variance"_frontside',
    'acc_x__fft_aggregated__aggtype_"skew"_frontside',
    'acc_x__fft_aggregated__aggtype_"kurtosis"_frontside',
    'acc_x__first_location_of_maximum_frontside',
    'acc_x__first_location_of_minimum_frontside',
    'acc_x__has_duplicate_frontside',
    'acc_x__has_duplicate_max_frontside',
    'acc_x__has_duplicate_min_frontside',
    'acc_x__kurtosis_frontside',
    'acc_x__last_location_of_maximum_frontside',
    'acc_x__last_location_of_minimum_frontside',
    'acc_x__length_frontside',
    'acc_x__linear_trend__attr_"pvalue"_frontside',
    'acc_x__linear_trend__attr_"rvalue"_frontside',
    'acc_x__linear_trend__attr_"intercept"_frontside',
    'acc_x__linear_trend__attr_"slope"_frontside',
    'acc_x__linear_trend__attr_"stderr"_frontside',
    'acc_x__longest_strike_above_mean_frontside',
    'acc_x__longest_strike_below_mean_frontside',
    'acc_x__maximum_frontside',
    'acc_x__mean_frontside',
    'acc_x__mean_abs_change_frontside',
    'acc_x__mean_change_frontside',
    'acc_x__mean_second_derivative_central_frontside',
    'acc_x__median_frontside',
    'acc_x__minimum_frontside',
    'acc_x__percentage_of_reoccurring_datapoints_to_all_datapoints_frontside',
    'acc_x__percentage_of_reoccurring_values_to_all_values_frontside',
    'acc_x__ratio_value_number_to_time_series_length_frontside',
    'acc_x__root_mean_square_frontside',
    'acc_x__skewness_frontside',
    'acc_x__standard_deviation_frontside',
    'acc_x__sum_of_reoccurring_data_points_frontside',
    'acc_x__sum_of_reoccurring_values_frontside',
    'acc_x__sum_values_frontside',
    'acc_x__variance_frontside',
    'acc_x__variance_larger_than_standard_deviation_frontside',
    'acc_x__variation_coefficient_frontside',
    'acc_y__abs_energy_frontside',
    'acc_y__absolute_maximum_frontside',
    'acc_y__absolute_sum_of_changes_frontside',
    'acc_y__benford_correlation_frontside',
    'acc_y__count_above_mean_frontside',
    'acc_y__count_below_mean_frontside',
    'acc_y__fft_aggregated__aggtype_"centroid"_frontside',
    'acc_y__fft_aggregated__aggtype_"variance"_frontside',
    'acc_y__fft_aggregated__aggtype_"skew"_frontside',
    'acc_y__fft_aggregated__aggtype_"kurtosis"_frontside',
    'acc_y__first_location_of_maximum_frontside',
    'acc_y__first_location_of_minimum_frontside',
    'acc_y__has_duplicate_frontside',
    'acc_y__has_duplicate_max_frontside',
    'acc_y__has_duplicate_min_frontside',
    'acc_y__kurtosis_frontside',
    'acc_y__last_location_of_maximum_frontside',
    'acc_y__last_location_of_minimum_frontside',
    'acc_y__length_frontside',
    'acc_y__linear_trend__attr_"pvalue"_frontside',
    'acc_y__linear_trend__attr_"rvalue"_frontside',
    'acc_y__linear_trend__attr_"intercept"_frontside',
    'acc_y__linear_trend__attr_"slope"_frontside',
    'acc_y__linear_trend__attr_"stderr"_frontside',
    'acc_y__longest_strike_above_mean_frontside',
    'acc_y__longest_strike_below_mean_frontside',
    'acc_y__maximum_frontside',
    'acc_y__mean_frontside',
    'acc_y__mean_abs_change_frontside',
    'acc_y__mean_change_frontside',
    'acc_y__mean_second_derivative_central_frontside',
    'acc_y__median_frontside',
    'acc_y__minimum_frontside',
    'acc_y__percentage_of_reoccurring_datapoints_to_all_datapoints_frontside',
    'acc_y__percentage_of_reoccurring_values_to_all_values_frontside',
    'acc_y__ratio_value_number_to_time_series_length_frontside',
    'acc_y__root_mean_square_frontside',
    'acc_y__skewness_frontside',
    'acc_y__standard_deviation_frontside',
    'acc_y__sum_of_reoccurring_data_points_frontside',
    'acc_y__sum_of_reoccurring_values_frontside',
    'acc_y__sum_values_frontside',
    'acc_y__variance_frontside',
    'acc_y__variance_larger_than_standard_deviation_frontside',
    'acc_y__variation_coefficient_frontside',
    'ae__abs_energy_frontside',
    'ae__absolute_maximum_frontside',
    'ae__absolute_sum_of_changes_frontside',
    'ae__benford_correlation_frontside',
    'ae__count_above_mean_frontside',
    'ae__count_below_mean_frontside',
    'ae__fft_aggregated__aggtype_"centroid"_frontside',
    'ae__fft_aggregated__aggtype_"variance"_frontside',
    'ae__fft_aggregated__aggtype_"skew"_frontside',
    'ae__fft_aggregated__aggtype_"kurtosis"_frontside',
    'ae__first_location_of_maximum_frontside',
    'ae__first_location_of_minimum_frontside',
    'ae__has_duplicate_frontside',
    'ae__has_duplicate_max_frontside',
    'ae__has_duplicate_min_frontside',
    'ae__kurtosis_frontside',
    'ae__last_location_of_maximum_frontside',
    'ae__last_location_of_minimum_frontside',
    'ae__length_frontside',
    'ae__linear_trend__attr_"pvalue"_frontside',
    'ae__linear_trend__attr_"rvalue"_frontside',
    'ae__linear_trend__attr_"intercept"_frontside',
    'ae__linear_trend__attr_"slope"_frontside',
    'ae__linear_trend__attr_"stderr"_frontside',
    'ae__longest_strike_above_mean_frontside',
    'ae__longest_strike_below_mean_frontside',
    'ae__maximum_frontside',
    'ae__mean_frontside',
    'ae__mean_abs_change_frontside',
    'ae__mean_change_frontside',
    'ae__mean_second_derivative_central_frontside',
    'ae__median_frontside',
    'ae__minimum_frontside',
    'ae__percentage_of_reoccurring_datapoints_to_all_datapoints_frontside',
    'ae__percentage_of_reoccurring_values_to_all_values_frontside',
    'ae__ratio_value_number_to_time_series_length_frontside',
    'ae__root_mean_square_frontside',
    'ae__skewness_frontside',
    'ae__standard_deviation_frontside',
    'ae__sum_of_reoccurring_data_points_frontside',
    'ae__sum_of_reoccurring_values_frontside',
    'ae__sum_values_frontside',
    'ae__variance_frontside',
    'ae__variance_larger_than_standard_deviation_frontside',
    'ae__variation_coefficient_frontside',
    'acc_z__abs_energy_frontside',
    'acc_z__absolute_maximum_frontside',
    'acc_z__absolute_sum_of_changes_frontside',
    'acc_z__benford_correlation_frontside',
    'acc_z__count_above_mean_frontside',
    'acc_z__count_below_mean_frontside',
    'acc_z__fft_aggregated__aggtype_"centroid"_frontside',
    'acc_z__fft_aggregated__aggtype_"variance"_frontside',
    'acc_z__fft_aggregated__aggtype_"skew"_frontside',
    'acc_z__fft_aggregated__aggtype_"kurtosis"_frontside',
    'acc_z__first_location_of_maximum_frontside',
    'acc_z__first_location_of_minimum_frontside',
    'acc_z__has_duplicate_frontside',
    'acc_z__has_duplicate_max_frontside',
    'acc_z__has_duplicate_min_frontside',
    'acc_z__kurtosis_frontside',
    'acc_z__last_location_of_maximum_frontside',
    'acc_z__last_location_of_minimum_frontside',
    'acc_z__length_frontside',
    'acc_z__linear_trend__attr_"pvalue"_frontside',
    'acc_z__linear_trend__attr_"rvalue"_frontside',
    'acc_z__linear_trend__attr_"intercept"_frontside',
    'acc_z__linear_trend__attr_"slope"_frontside',
    'acc_z__linear_trend__attr_"stderr"_frontside',
    'acc_z__longest_strike_above_mean_frontside',
    'acc_z__longest_strike_below_mean_frontside',
    'acc_z__maximum_frontside',
    'acc_z__mean_frontside',
    'acc_z__mean_abs_change_frontside',
    'acc_z__mean_change_frontside',
    'acc_z__mean_second_derivative_central_frontside',
    'acc_z__median_frontside',
    'acc_z__minimum_frontside',
    'acc_z__percentage_of_reoccurring_datapoints_to_all_datapoints_frontside',
    'acc_z__percentage_of_reoccurring_values_to_all_values_frontside',
    'acc_z__ratio_value_number_to_time_series_length_frontside',
    'acc_z__root_mean_square_frontside',
    'acc_z__skewness_frontside',
    'acc_z__standard_deviation_frontside',
    'acc_z__sum_of_reoccurring_data_points_frontside',
    'acc_z__sum_of_reoccurring_values_frontside',
    'acc_z__sum_values_frontside',
    'acc_z__variance_frontside',
    'acc_z__variance_larger_than_standard_deviation_frontside',
    'acc_z__variation_coefficient_frontside',
    'part_id',
    'total_acc__abs_energy_backside',
    'total_acc__absolute_maximum_backside',
    'total_acc__absolute_sum_of_changes_backside',
    'total_acc__benford_correlation_backside',
    'total_acc__count_above_mean_backside',
    'total_acc__count_below_mean_backside',
    'total_acc__fft_aggregated__aggtype_"centroid"_backside',
    'total_acc__fft_aggregated__aggtype_"variance"_backside',
    'total_acc__fft_aggregated__aggtype_"skew"_backside',
    'total_acc__fft_aggregated__aggtype_"kurtosis"_backside',
    'total_acc__first_location_of_maximum_backside',
    'total_acc__first_location_of_minimum_backside',
    'total_acc__has_duplicate_backside',
    'total_acc__has_duplicate_max_backside',
    'total_acc__has_duplicate_min_backside',
    'total_acc__kurtosis_backside',
    'total_acc__last_location_of_maximum_backside',
    'total_acc__last_location_of_minimum_backside',
    'total_acc__length_backside',
    'total_acc__linear_trend__attr_"pvalue"_backside',
    'total_acc__linear_trend__attr_"rvalue"_backside',
    'total_acc__linear_trend__attr_"intercept"_backside',
    'total_acc__linear_trend__attr_"slope"_backside',
    'total_acc__linear_trend__attr_"stderr"_backside',
    'total_acc__longest_strike_above_mean_backside',
    'total_acc__longest_strike_below_mean_backside',
    'total_acc__maximum_backside',
    'total_acc__mean_backside',
    'total_acc__mean_abs_change_backside',
    'total_acc__mean_change_backside',
    'total_acc__mean_second_derivative_central_backside',
    'total_acc__median_backside',
    'total_acc__minimum_backside',
    'total_acc__percentage_of_reoccurring_datapoints_to_all_datapoints_backside',
    'total_acc__percentage_of_reoccurring_values_to_all_values_backside',
    'total_acc__ratio_value_number_to_time_series_length_backside',
    'total_acc__root_mean_square_backside',
    'total_acc__skewness_backside',
    'total_acc__standard_deviation_backside',
    'total_acc__sum_of_reoccurring_data_points_backside',
    'total_acc__sum_of_reoccurring_values_backside',
    'total_acc__sum_values_backside',
    'total_acc__variance_backside',
    'total_acc__variance_larger_than_standard_deviation_backside',
    'total_acc__variation_coefficient_backside',
    'acc_x__abs_energy_backside',
    'acc_x__absolute_maximum_backside',
    'acc_x__absolute_sum_of_changes_backside',
    'acc_x__benford_correlation_backside',
    'acc_x__count_above_mean_backside',
    'acc_x__count_below_mean_backside',
    'acc_x__fft_aggregated__aggtype_"centroid"_backside',
    'acc_x__fft_aggregated__aggtype_"variance"_backside',
    'acc_x__fft_aggregated__aggtype_"skew"_backside',
    'acc_x__fft_aggregated__aggtype_"kurtosis"_backside',
    'acc_x__first_location_of_maximum_backside',
    'acc_x__first_location_of_minimum_backside',
    'acc_x__has_duplicate_backside',
    'acc_x__has_duplicate_max_backside',
    'acc_x__has_duplicate_min_backside',
    'acc_x__kurtosis_backside',
    'acc_x__last_location_of_maximum_backside',
    'acc_x__last_location_of_minimum_backside',
    'acc_x__length_backside',
    'acc_x__linear_trend__attr_"pvalue"_backside',
    'acc_x__linear_trend__attr_"rvalue"_backside',
    'acc_x__linear_trend__attr_"intercept"_backside',
    'acc_x__linear_trend__attr_"slope"_backside',
    'acc_x__linear_trend__attr_"stderr"_backside',
    'acc_x__longest_strike_above_mean_backside',
    'acc_x__longest_strike_below_mean_backside',
    'acc_x__maximum_backside',
    'acc_x__mean_backside',
    'acc_x__mean_abs_change_backside',
    'acc_x__mean_change_backside',
    'acc_x__mean_second_derivative_central_backside',
    'acc_x__median_backside',
    'acc_x__minimum_backside',
    'acc_x__percentage_of_reoccurring_datapoints_to_all_datapoints_backside',
    'acc_x__percentage_of_reoccurring_values_to_all_values_backside',
    'acc_x__ratio_value_number_to_time_series_length_backside',
    'acc_x__root_mean_square_backside',
    'acc_x__skewness_backside',
    'acc_x__standard_deviation_backside',
    'acc_x__sum_of_reoccurring_data_points_backside',
    'acc_x__sum_of_reoccurring_values_backside',
    'acc_x__sum_values_backside',
    'acc_x__variance_backside',
    'acc_x__variance_larger_than_standard_deviation_backside',
    'acc_x__variation_coefficient_backside',
    'ae__abs_energy_backside',
    'ae__absolute_maximum_backside',
    'ae__absolute_sum_of_changes_backside',
    'ae__benford_correlation_backside',
    'ae__count_above_mean_backside',
    'ae__count_below_mean_backside',
    'ae__fft_aggregated__aggtype_"centroid"_backside',
    'ae__fft_aggregated__aggtype_"variance"_backside',
    'ae__fft_aggregated__aggtype_"skew"_backside',
    'ae__fft_aggregated__aggtype_"kurtosis"_backside',
    'ae__first_location_of_maximum_backside',
    'ae__first_location_of_minimum_backside',
    'ae__has_duplicate_backside',
    'ae__has_duplicate_max_backside',
    'ae__has_duplicate_min_backside',
    'ae__kurtosis_backside',
    'ae__last_location_of_maximum_backside',
    'ae__last_location_of_minimum_backside',
    'ae__length_backside',
    'ae__linear_trend__attr_"pvalue"_backside',
    'ae__linear_trend__attr_"rvalue"_backside',
    'ae__linear_trend__attr_"intercept"_backside',
    'ae__linear_trend__attr_"slope"_backside',
    'ae__linear_trend__attr_"stderr"_backside',
    'ae__longest_strike_above_mean_backside',
    'ae__longest_strike_below_mean_backside',
    'ae__maximum_backside',
    'ae__mean_backside',
    'ae__mean_abs_change_backside',
    'ae__mean_change_backside',
    'ae__mean_second_derivative_central_backside',
    'ae__median_backside',
    'ae__minimum_backside',
    'ae__percentage_of_reoccurring_datapoints_to_all_datapoints_backside',
    'ae__percentage_of_reoccurring_values_to_all_values_backside',
    'ae__ratio_value_number_to_time_series_length_backside',
    'ae__root_mean_square_backside',
    'ae__skewness_backside',
    'ae__standard_deviation_backside',
    'ae__sum_of_reoccurring_data_points_backside',
    'ae__sum_of_reoccurring_values_backside',
    'ae__sum_values_backside',
    'ae__variance_backside',
    'ae__variance_larger_than_standard_deviation_backside',
    'ae__variation_coefficient_backside',
    'acc_y__abs_energy_backside',
    'acc_y__absolute_maximum_backside',
    'acc_y__absolute_sum_of_changes_backside',
    'acc_y__benford_correlation_backside',
    'acc_y__count_above_mean_backside',
    'acc_y__count_below_mean_backside',
    'acc_y__fft_aggregated__aggtype_"centroid"_backside',
    'acc_y__fft_aggregated__aggtype_"variance"_backside',
    'acc_y__fft_aggregated__aggtype_"skew"_backside',
    'acc_y__fft_aggregated__aggtype_"kurtosis"_backside',
    'acc_y__first_location_of_maximum_backside',
    'acc_y__first_location_of_minimum_backside',
    'acc_y__has_duplicate_backside',
    'acc_y__has_duplicate_max_backside',
    'acc_y__has_duplicate_min_backside',
    'acc_y__kurtosis_backside',
    'acc_y__last_location_of_maximum_backside',
    'acc_y__last_location_of_minimum_backside',
    'acc_y__length_backside',
    'acc_y__linear_trend__attr_"pvalue"_backside',
    'acc_y__linear_trend__attr_"rvalue"_backside',
    'acc_y__linear_trend__attr_"intercept"_backside',
    'acc_y__linear_trend__attr_"slope"_backside',
    'acc_y__linear_trend__attr_"stderr"_backside',
    'acc_y__longest_strike_above_mean_backside',
    'acc_y__longest_strike_below_mean_backside',
    'acc_y__maximum_backside',
    'acc_y__mean_backside',
    'acc_y__mean_abs_change_backside',
    'acc_y__mean_change_backside',
    'acc_y__mean_second_derivative_central_backside',
    'acc_y__median_backside',
    'acc_y__minimum_backside',
    'acc_y__percentage_of_reoccurring_datapoints_to_all_datapoints_backside',
    'acc_y__percentage_of_reoccurring_values_to_all_values_backside',
    'acc_y__ratio_value_number_to_time_series_length_backside',
    'acc_y__root_mean_square_backside',
    'acc_y__skewness_backside',
    'acc_y__standard_deviation_backside',
    'acc_y__sum_of_reoccurring_data_points_backside',
    'acc_y__sum_of_reoccurring_values_backside',
    'acc_y__sum_values_backside',
    'acc_y__variance_backside',
    'acc_y__variance_larger_than_standard_deviation_backside',
    'acc_y__variation_coefficient_backside',
    'acc_z__abs_energy_backside',
    'acc_z__absolute_maximum_backside',
    'acc_z__absolute_sum_of_changes_backside',
    'acc_z__benford_correlation_backside',
    'acc_z__count_above_mean_backside',
    'acc_z__count_below_mean_backside',
    'acc_z__fft_aggregated__aggtype_"centroid"_backside',
    'acc_z__fft_aggregated__aggtype_"variance"_backside',
    'acc_z__fft_aggregated__aggtype_"skew"_backside',
    'acc_z__fft_aggregated__aggtype_"kurtosis"_backside',
    'acc_z__first_location_of_maximum_backside',
    'acc_z__first_location_of_minimum_backside',
    'acc_z__has_duplicate_backside',
    'acc_z__has_duplicate_max_backside',
    'acc_z__has_duplicate_min_backside',
    'acc_z__kurtosis_backside',
    'acc_z__last_location_of_maximum_backside',
    'acc_z__last_location_of_minimum_backside',
    'acc_z__length_backside',
    'acc_z__linear_trend__attr_"pvalue"_backside',
    'acc_z__linear_trend__attr_"rvalue"_backside',
    'acc_z__linear_trend__attr_"intercept"_backside',
    'acc_z__linear_trend__attr_"slope"_backside',
    'acc_z__linear_trend__attr_"stderr"_backside',
    'acc_z__longest_strike_above_mean_backside',
    'acc_z__longest_strike_below_mean_backside',
    'acc_z__maximum_backside',
    'acc_z__mean_backside',
    'acc_z__mean_abs_change_backside',
    'acc_z__mean_change_backside',
    'acc_z__mean_second_derivative_central_backside',
    'acc_z__median_backside',
    'acc_z__minimum_backside',
    'acc_z__percentage_of_reoccurring_datapoints_to_all_datapoints_backside',
    'acc_z__percentage_of_reoccurring_values_to_all_values_backside',
    'acc_z__ratio_value_number_to_time_series_length_backside',
    'acc_z__root_mean_square_backside',
    'acc_z__skewness_backside',
    'acc_z__standard_deviation_backside',
    'acc_z__sum_of_reoccurring_data_points_backside',
    'acc_z__sum_of_reoccurring_values_backside',
    'acc_z__sum_values_backside',
    'acc_z__variance_backside',
    'acc_z__variance_larger_than_standard_deviation_backside',
    'acc_z__variation_coefficient_backside',
    'spectral_spread_x_frontside',
    'spectral_spread_y_frontside',
    'spectral_spread_z_frontside',
    'spectral_spread_total_frontside',
    'spectral_spread_ae_frontside',
    'peak_frequency_x_frontside',
    'peak_frequency_y_frontside',
    'peak_frequency_z_frontside',
    'peak_frequency_total_frontside',
    'peak_frequency_ae_frontside',
    'num_peaks_x_frontside',
    'mean_peak_height_x_frontside',
    'std_peak_height_x_frontside',
    'max_peak_height_x_frontside',
    'min_peak_height_x_frontside',
    'mean_peak_frequency_x_frontside',
    'std_peak_frequency_x_frontside',
    'max_peak_frequency_x_frontside',
    'min_peak_frequency_x_frontside',
    'mean_peak_energy_x_frontside',
    'std_peak_energy_x_frontside',
    'max_peak_energy_x_frontside',
    'min_peak_energy_x_frontside',
    'mean_peak_energy_ratio_x_frontside',
    'std_peak_energy_ratio_x_frontside',
    'max_peak_energy_ratio_x_frontside',
    'min_peak_energy_ratio_x_frontside',
    'num_significant_peaks_x_frontside',
    'mean_peak_distance_x_frontside',
    'std_peak_distance_x_frontside',
    'max_peak_distance_x_frontside',
    'min_peak_distance_x_frontside',
    'num_peaks_y_frontside',
    'mean_peak_height_y_frontside',
    'std_peak_height_y_frontside',
    'max_peak_height_y_frontside',
    'min_peak_height_y_frontside',
    'mean_peak_frequency_y_frontside',
    'std_peak_frequency_y_frontside',
    'max_peak_frequency_y_frontside',
    'min_peak_frequency_y_frontside',
    'mean_peak_energy_y_frontside',
    'std_peak_energy_y_frontside',
    'max_peak_energy_y_frontside',
    'min_peak_energy_y_frontside',
    'mean_peak_energy_ratio_y_frontside',
    'std_peak_energy_ratio_y_frontside',
    'max_peak_energy_ratio_y_frontside',
    'min_peak_energy_ratio_y_frontside',
    'num_significant_peaks_y_frontside',
    'mean_peak_distance_y_frontside',
    'std_peak_distance_y_frontside',
    'max_peak_distance_y_frontside',
    'min_peak_distance_y_frontside',
    'num_peaks_z_frontside',
    'mean_peak_height_z_frontside',
    'std_peak_height_z_frontside',
    'max_peak_height_z_frontside',
    'min_peak_height_z_frontside',
    'mean_peak_frequency_z_frontside',
    'std_peak_frequency_z_frontside',
    'max_peak_frequency_z_frontside',
    'min_peak_frequency_z_frontside',
    'mean_peak_energy_z_frontside',
    'std_peak_energy_z_frontside',
    'max_peak_energy_z_frontside',
    'min_peak_energy_z_frontside',
    'mean_peak_energy_ratio_z_frontside',
    'std_peak_energy_ratio_z_frontside',
    'max_peak_energy_ratio_z_frontside',
    'min_peak_energy_ratio_z_frontside',
    'num_significant_peaks_z_frontside',
    'mean_peak_distance_z_frontside',
    'std_peak_distance_z_frontside',
    'max_peak_distance_z_frontside',
    'min_peak_distance_z_frontside',
    'num_peaks_total_frontside',
    'mean_peak_height_total_frontside',
    'std_peak_height_total_frontside',
    'max_peak_height_total_frontside',
    'min_peak_height_total_frontside',
    'mean_peak_frequency_total_frontside',
    'std_peak_frequency_total_frontside',
    'max_peak_frequency_total_frontside',
    'min_peak_frequency_total_frontside',
    'mean_peak_energy_total_frontside',
    'std_peak_energy_total_frontside',
    'max_peak_energy_total_frontside',
    'min_peak_energy_total_frontside',
    'mean_peak_energy_ratio_total_frontside',
    'std_peak_energy_ratio_total_frontside',
    'max_peak_energy_ratio_total_frontside',
    'min_peak_energy_ratio_total_frontside',
    'num_significant_peaks_total_frontside',
    'mean_peak_distance_total_frontside',
    'std_peak_distance_total_frontside',
    'max_peak_distance_total_frontside',
    'min_peak_distance_total_frontside',
    'num_peaks_ae_frontside',
    'mean_peak_height_ae_frontside',
    'std_peak_height_ae_frontside',
    'max_peak_height_ae_frontside',
    'min_peak_height_ae_frontside',
    'mean_peak_frequency_ae_frontside',
    'std_peak_frequency_ae_frontside',
    'max_peak_frequency_ae_frontside',
    'min_peak_frequency_ae_frontside',
    'mean_peak_energy_ae_frontside',
    'std_peak_energy_ae_frontside',
    'max_peak_energy_ae_frontside',
    'min_peak_energy_ae_frontside',
    'mean_peak_energy_ratio_ae_frontside',
    'std_peak_energy_ratio_ae_frontside',
    'max_peak_energy_ratio_ae_frontside',
    'min_peak_energy_ratio_ae_frontside',
    'num_significant_peaks_ae_frontside',
    'mean_peak_distance_ae_frontside',
    'std_peak_distance_ae_frontside',
    'max_peak_distance_ae_frontside',
    'min_peak_distance_ae_frontside',
    'spectral_energy_x_frontside',
    'spectral_energy_y_frontside',
    'spectral_energy_z_frontside',
    'spectral_energy_total_frontside',
    'spectral_energy_ae_frontside',
    'magnitude_mean_x_frontside',
    'magnitude_std_x_frontside',
    'magnitude_1_max_x_frontside',
    'magnitude_2_max_x_frontside',
    'magnitude_3_max_x_frontside',
    'magnitude_min_x_frontside',
    'magnitude_mean_y_frontside',
    'magnitude_std_y_frontside',
    'magnitude_1_max_y_frontside',
    'magnitude_2_max_y_frontside',
    'magnitude_3_max_y_frontside',
    'magnitude_min_y_frontside',
    'magnitude_mean_z_frontside',
    'magnitude_std_z_frontside',
    'magnitude_1_max_z_frontside',
    'magnitude_2_max_z_frontside',
    'magnitude_3_max_z_frontside',
    'magnitude_min_z_frontside',
    'magnitude_mean_total_frontside',
    'magnitude_std_total_frontside',
    'magnitude_1_max_total_frontside',
    'magnitude_2_max_total_frontside',
    'magnitude_3_max_total_frontside',
    'magnitude_min_total_frontside',
    'magnitude_mean_ae_frontside',
    'magnitude_std_ae_frontside',
    'magnitude_1_max_ae_frontside',
    'magnitude_2_max_ae_frontside',
    'magnitude_3_max_ae_frontside',
    'magnitude_min_ae_frontside',
    'spectral_kurtosis_x_frontside',
    'spectral_kurtosis_y_frontside',
    'spectral_kurtosis_z_frontside',
    'spectral_kurtosis_total_frontside',
    'spectral_kurtosis_ae_frontside',
    'spectral_entropy_x_frontside',
    'spectral_entropy_y_frontside',
    'spectral_entropy_z_frontside',
    'spectral_entropy_total_frontside',
    'spectral_entropy_ae_frontside',
    'psd_num_peaks_x_frontside',
    'psd_mean_peak_value_x_frontside',
    'psd_std_peak_value_x_frontside',
    'psd_max_peak_value_x_frontside',
    'psd_min_peak_value_x_frontside',
    'psd_mean_peak_frequency_x_frontside',
    'psd_std_peak_frequency_x_frontside',
    'psd_max_peak_frequency_x_frontside',
    'psd_min_peak_frequency_x_frontside',
    'psd_mean_peak_distance_x_frontside',
    'psd_std_peak_distance_x_frontside',
    'psd_max_peak_distance_x_frontside',
    'psd_min_peak_distance_x_frontside',
    'mean_psd_x_frontside',
    'std_psd_x_frontside',
    'max_psd_x_frontside',
    'min_psd_x_frontside',
    'skewness_psd_x_frontside',
    'kurtosis_psd_x_frontside',
    'power_centroid_psd_x_frontside',
    'psd_num_peaks_y_frontside',
    'psd_mean_peak_value_y_frontside',
    'psd_std_peak_value_y_frontside',
    'psd_max_peak_value_y_frontside',
    'psd_min_peak_value_y_frontside',
    'psd_mean_peak_frequency_y_frontside',
    'psd_std_peak_frequency_y_frontside',
    'psd_max_peak_frequency_y_frontside',
    'psd_min_peak_frequency_y_frontside',
    'psd_mean_peak_distance_y_frontside',
    'psd_std_peak_distance_y_frontside',
    'psd_max_peak_distance_y_frontside',
    'psd_min_peak_distance_y_frontside',
    'mean_psd_y_frontside',
    'std_psd_y_frontside',
    'max_psd_y_frontside',
    'min_psd_y_frontside',
    'skewness_psd_y_frontside',
    'kurtosis_psd_y_frontside',
    'power_centroid_psd_y_frontside',
    'psd_num_peaks_z_frontside',
    'psd_mean_peak_value_z_frontside',
    'psd_std_peak_value_z_frontside',
    'psd_max_peak_value_z_frontside',
    'psd_min_peak_value_z_frontside',
    'psd_mean_peak_frequency_z_frontside',
    'psd_std_peak_frequency_z_frontside',
    'psd_max_peak_frequency_z_frontside',
    'psd_min_peak_frequency_z_frontside',
    'psd_mean_peak_distance_z_frontside',
    'psd_std_peak_distance_z_frontside',
    'psd_max_peak_distance_z_frontside',
    'psd_min_peak_distance_z_frontside',
    'mean_psd_z_frontside',
    'std_psd_z_frontside',
    'max_psd_z_frontside',
    'min_psd_z_frontside',
    'skewness_psd_z_frontside',
    'kurtosis_psd_z_frontside',
    'power_centroid_psd_z_frontside',
    'psd_num_peaks_total_frontside',
    'psd_mean_peak_value_total_frontside',
    'psd_std_peak_value_total_frontside',
    'psd_max_peak_value_total_frontside',
    'psd_min_peak_value_total_frontside',
    'psd_mean_peak_frequency_total_frontside',
    'psd_std_peak_frequency_total_frontside',
    'psd_max_peak_frequency_total_frontside',
    'psd_min_peak_frequency_total_frontside',
    'psd_mean_peak_distance_total_frontside',
    'psd_std_peak_distance_total_frontside',
    'psd_max_peak_distance_total_frontside',
    'psd_min_peak_distance_total_frontside',
    'mean_psd_total_frontside',
    'std_psd_total_frontside',
    'max_psd_total_frontside',
    'min_psd_total_frontside',
    'skewness_psd_total_frontside',
    'kurtosis_psd_total_frontside',
    'power_centroid_psd_total_frontside',
    'psd_num_peaks_ae_frontside',
    'psd_mean_peak_value_ae_frontside',
    'psd_std_peak_value_ae_frontside',
    'psd_max_peak_value_ae_frontside',
    'psd_min_peak_value_ae_frontside',
    'psd_mean_peak_frequency_ae_frontside',
    'psd_std_peak_frequency_ae_frontside',
    'psd_max_peak_frequency_ae_frontside',
    'psd_min_peak_frequency_ae_frontside',
    'psd_mean_peak_distance_ae_frontside',
    'psd_std_peak_distance_ae_frontside',
    'psd_max_peak_distance_ae_frontside',
    'psd_min_peak_distance_ae_frontside',
    'mean_psd_ae_frontside',
    'std_psd_ae_frontside',
    'max_psd_frontside',
    'min_psd_ae_frontside',
    'skewness_psd_ae_frontside',
    'kurtosis_psd_ae_frontside',
    'power_centroid_psd_ae_frontside',
    'spectral_roll_off_point_x_frontside',
    'spectral_roll_off_point_y_frontside',
    'spectral_roll_off_point_z_frontside',
    'spectral_roll_off_point_total_frontside',
    'spectral_roll_off_point_ae_frontside',
    'fundamental_frequency_x_frontside',
    'fundamental_frequency_y_frontside',
    'fundamental_frequency_z_frontside',
    'fundamental_frequency_total_frontside',
    'fundamental_frequency_ae_frontside',
    'spectral_centroid_x_frontside',
    'spectral_centroid_y_frontside',
    'spectral_centroid_z_frontside',
    'spectral_centroid_total_frontside',
    'spectral_centroid_ae_frontside',
    'spectral_skewness_x_frontside',
    'spectral_skewness_y_frontside',
    'spectral_skewness_z_frontside',
    'spectral_skewness_total_frontside',
    'spectral_skewness_ae_frontside',
    'spectral_coefficient_of_variation_x_frontside',
    'spectral_coefficient_of_variation_y_frontside',
    'spectral_coefficient_of_variation_z_frontside',
    'spectral_coefficient_of_variazion_total_frontside',
    'spectral_coefficient_of_variation_ae_frontside',
    'spectral_flatness_x_frontside',
    'spectral_flatness_y_frontside',
    'spectral_flatness_z_frontside',
    'spectral_flatness_total_frontside',
    'spectral_flatness_ae_frontside',
    'weighted_mean_frequency_x_frontside',
    'weighted_mean_frequency_y_frontside',
    'weighted_mean_frequency_z_frontside',
    'weighted_mean_frequency_total_frontside',
    'weighted_mean_frequency_ae_frontside',
    'spectral_coefficient_of_variation_x_backside',
    'spectral_coefficient_of_variation_y_backside',
    'spectral_coefficient_of_variation_z_backside',
    'spectral_coefficient_of_variazion_total_backside',
    'spectral_coefficient_of_variation_ae_backside',
    'spectral_centroid_x_backside',
    'spectral_centroid_y_backside',
    'spectral_centroid_z_backside',
    'spectral_centroid_total_backside',
    'spectral_centroid_ae_backside',
    'spectral_roll_off_point_x_backside',
    'spectral_roll_off_point_y_backside',
    'spectral_roll_off_point_z_backside',
    'spectral_roll_off_point_total_backside',
    'spectral_roll_off_point_ae_backside',
    'spectral_entropy_x_backside',
    'spectral_entropy_y_backside',
    'spectral_entropy_z_backside',
    'spectral_entropy_total_backside',
    'spectral_entropy_ae_backside',
    'psd_num_peaks_x_backside',
    'psd_mean_peak_value_x_backside',
    'psd_std_peak_value_x_backside',
    'psd_max_peak_value_x_backside',
    'psd_min_peak_value_x_backside',
    'psd_mean_peak_frequency_x_backside',
    'psd_std_peak_frequency_x_backside',
    'psd_max_peak_frequency_x_backside',
    'psd_min_peak_frequency_x_backside',
    'psd_mean_peak_distance_x_backside',
    'psd_std_peak_distance_x_backside',
    'psd_max_peak_distance_x_backside',
    'psd_min_peak_distance_x_backside',
    'mean_psd_x_backside',
    'std_psd_x_backside',
    'max_psd_x_backside',
    'min_psd_x_backside',
    'skewness_psd_x_backside',
    'kurtosis_psd_x_backside',
    'power_centroid_psd_x_backside',
    'psd_num_peaks_y_backside',
    'psd_mean_peak_value_y_backside',
    'psd_std_peak_value_y_backside',
    'psd_max_peak_value_y_backside',
    'psd_min_peak_value_y_backside',
    'psd_mean_peak_frequency_y_backside',
    'psd_std_peak_frequency_y_backside',
    'psd_max_peak_frequency_y_backside',
    'psd_min_peak_frequency_y_backside',
    'psd_mean_peak_distance_y_backside',
    'psd_std_peak_distance_y_backside',
    'psd_max_peak_distance_y_backside',
    'psd_min_peak_distance_y_backside',
    'mean_psd_y_backside',
    'std_psd_y_backside',
    'max_psd_y_backside',
    'min_psd_y_backside',
    'skewness_psd_y_backside',
    'kurtosis_psd_y_backside',
    'power_centroid_psd_y_backside',
    'psd_num_peaks_z_backside',
    'psd_mean_peak_value_z_backside',
    'psd_std_peak_value_z_backside',
    'psd_max_peak_value_z_backside',
    'psd_min_peak_value_z_backside',
    'psd_mean_peak_frequency_z_backside',
    'psd_std_peak_frequency_z_backside',
    'psd_max_peak_frequency_z_backside',
    'psd_min_peak_frequency_z_backside',
    'psd_mean_peak_distance_z_backside',
    'psd_std_peak_distance_z_backside',
    'psd_max_peak_distance_z_backside',
    'psd_min_peak_distance_z_backside',
    'mean_psd_z_backside',
    'std_psd_z_backside',
    'max_psd_z_backside',
    'min_psd_z_backside',
    'skewness_psd_z_backside',
    'kurtosis_psd_z_backside',
    'power_centroid_psd_z_backside',
    'psd_num_peaks_total_backside',
    'psd_mean_peak_value_total_backside',
    'psd_std_peak_value_total_backside',
    'psd_max_peak_value_total_backside',
    'psd_min_peak_value_total_backside',
    'psd_mean_peak_frequency_total_backside',
    'psd_std_peak_frequency_total_backside',
    'psd_max_peak_frequency_total_backside',
    'psd_min_peak_frequency_total_backside',
    'psd_mean_peak_distance_total_backside',
    'psd_std_peak_distance_total_backside',
    'psd_max_peak_distance_total_backside',
    'psd_min_peak_distance_total_backside',
    'mean_psd_total_backside',
    'std_psd_total_backside',
    'max_psd_total_backside',
    'min_psd_total_backside',
    'skewness_psd_total_backside',
    'kurtosis_psd_total_backside',
    'power_centroid_psd_total_backside',
    'psd_num_peaks_ae_backside',
    'psd_mean_peak_value_ae_backside',
    'psd_std_peak_value_ae_backside',
    'psd_max_peak_value_ae_backside',
    'psd_min_peak_value_ae_backside',
    'psd_mean_peak_frequency_ae_backside',
    'psd_std_peak_frequency_ae_backside',
    'psd_max_peak_frequency_ae_backside',
    'psd_min_peak_frequency_ae_backside',
    'psd_mean_peak_distance_ae_backside',
    'psd_std_peak_distance_ae_backside',
    'psd_max_peak_distance_ae_backside',
    'psd_min_peak_distance_ae_backside',
    'mean_psd_ae_backside',
    'std_psd_ae_backside',
    'max_psd_backside',
    'min_psd_ae_backside',
    'skewness_psd_ae_backside',
    'kurtosis_psd_ae_backside',
    'power_centroid_psd_ae_backside',
    'spectral_skewness_x_backside',
    'spectral_skewness_y_backside',
    'spectral_skewness_z_backside',
    'spectral_skewness_total_backside',
    'spectral_skewness_ae_backside',
    'weighted_mean_frequency_x_backside',
    'weighted_mean_frequency_y_backside',
    'weighted_mean_frequency_z_backside',
    'weighted_mean_frequency_total_backside',
    'weighted_mean_frequency_ae_backside',
    'spectral_spread_x_backside',
    'spectral_spread_y_backside',
    'spectral_spread_z_backside',
    'spectral_spread_total_backside',
    'spectral_spread_ae_backside',
    'spectral_energy_x_backside',
    'spectral_energy_y_backside',
    'spectral_energy_z_backside',
    'spectral_energy_total_backside',
    'spectral_energy_ae_backside',
    'num_peaks_x_backside',
    'mean_peak_height_x_backside',
    'std_peak_height_x_backside',
    'max_peak_height_x_backside',
    'min_peak_height_x_backside',
    'mean_peak_frequency_x_backside',
    'std_peak_frequency_x_backside',
    'max_peak_frequency_x_backside',
    'min_peak_frequency_x_backside',
    'mean_peak_energy_x_backside',
    'std_peak_energy_x_backside',
    'max_peak_energy_x_backside',
    'min_peak_energy_x_backside',
    'mean_peak_energy_ratio_x_backside',
    'std_peak_energy_ratio_x_backside',
    'max_peak_energy_ratio_x_backside',
    'min_peak_energy_ratio_x_backside',
    'num_significant_peaks_x_backside',
    'mean_peak_distance_x_backside',
    'std_peak_distance_x_backside',
    'max_peak_distance_x_backside',
    'min_peak_distance_x_backside',
    'num_peaks_y_backside',
    'mean_peak_height_y_backside',
    'std_peak_height_y_backside',
    'max_peak_height_y_backside',
    'min_peak_height_y_backside',
    'mean_peak_frequency_y_backside',
    'std_peak_frequency_y_backside',
    'max_peak_frequency_y_backside',
    'min_peak_frequency_y_backside',
    'mean_peak_energy_y_backside',
    'std_peak_energy_y_backside',
    'max_peak_energy_y_backside',
    'min_peak_energy_y_backside',
    'mean_peak_energy_ratio_y_backside',
    'std_peak_energy_ratio_y_backside',
    'max_peak_energy_ratio_y_backside',
    'min_peak_energy_ratio_y_backside',
    'num_significant_peaks_y_backside',
    'mean_peak_distance_y_backside',
    'std_peak_distance_y_backside',
    'max_peak_distance_y_backside',
    'min_peak_distance_y_backside',
    'num_peaks_z_backside',
    'mean_peak_height_z_backside',
    'std_peak_height_z_backside',
    'max_peak_height_z_backside',
    'min_peak_height_z_backside',
    'mean_peak_frequency_z_backside',
    'std_peak_frequency_z_backside',
    'max_peak_frequency_z_backside',
    'min_peak_frequency_z_backside',
    'mean_peak_energy_z_backside',
    'std_peak_energy_z_backside',
    'max_peak_energy_z_backside',
    'min_peak_energy_z_backside',
    'mean_peak_energy_ratio_z_backside',
    'std_peak_energy_ratio_z_backside',
    'max_peak_energy_ratio_z_backside',
    'min_peak_energy_ratio_z_backside',
    'num_significant_peaks_z_backside',
    'mean_peak_distance_z_backside',
    'std_peak_distance_z_backside',
    'max_peak_distance_z_backside',
    'min_peak_distance_z_backside',
    'num_peaks_total_backside',
    'mean_peak_height_total_backside',
    'std_peak_height_total_backside',
    'max_peak_height_total_backside',
    'min_peak_height_total_backside',
    'mean_peak_frequency_total_backside',
    'std_peak_frequency_total_backside',
    'max_peak_frequency_total_backside',
    'min_peak_frequency_total_backside',
    'mean_peak_energy_total_backside',
    'std_peak_energy_total_backside',
    'max_peak_energy_total_backside',
    'min_peak_energy_total_backside',
    'mean_peak_energy_ratio_total_backside',
    'std_peak_energy_ratio_total_backside',
    'max_peak_energy_ratio_total_backside',
    'min_peak_energy_ratio_total_backside',
    'num_significant_peaks_total_backside',
    'mean_peak_distance_total_backside',
    'std_peak_distance_total_backside',
    'max_peak_distance_total_backside',
    'min_peak_distance_total_backside',
    'num_peaks_ae_backside',
    'mean_peak_height_ae_backside',
    'std_peak_height_ae_backside',
    'max_peak_height_ae_backside',
    'min_peak_height_ae_backside',
    'mean_peak_frequency_ae_backside',
    'std_peak_frequency_ae_backside',
    'max_peak_frequency_ae_backside',
    'min_peak_frequency_ae_backside',
    'mean_peak_energy_ae_backside',
    'std_peak_energy_ae_backside',
    'max_peak_energy_ae_backside',
    'min_peak_energy_ae_backside',
    'mean_peak_energy_ratio_ae_backside',
    'std_peak_energy_ratio_ae_backside',
    'max_peak_energy_ratio_ae_backside',
    'min_peak_energy_ratio_ae_backside',
    'num_significant_peaks_ae_backside',
    'mean_peak_distance_ae_backside',
    'std_peak_distance_ae_backside',
    'max_peak_distance_ae_backside',
    'min_peak_distance_ae_backside',
    'spectral_kurtosis_x_backside',
    'spectral_kurtosis_y_backside',
    'spectral_kurtosis_z_backside',
    'spectral_kurtosis_total_backside',
    'spectral_kurtosis_ae_backside',
    'peak_frequency_x_backside',
    'peak_frequency_y_backside',
    'peak_frequency_z_backside',
    'peak_frequency_total_backside',
    'peak_frequency_ae_backside',
    'magnitude_mean_x_backside',
    'magnitude_std_x_backside',
    'magnitude_1_max_x_backside',
    'magnitude_2_max_x_backside',
    'magnitude_3_max_x_backside',
    'magnitude_min_x_backside',
    'magnitude_mean_y_backside',
    'magnitude_std_y_backside',
    'magnitude_1_max_y_backside',
    'magnitude_2_max_y_backside',
    'magnitude_3_max_y_backside',
    'magnitude_min_y_backside',
    'magnitude_mean_z_backside',
    'magnitude_std_z_backside',
    'magnitude_1_max_z_backside',
    'magnitude_2_max_z_backside',
    'magnitude_3_max_z_backside',
    'magnitude_min_z_backside',
    'magnitude_mean_total_backside',
    'magnitude_std_total_backside',
    'magnitude_1_max_total_backside',
    'magnitude_2_max_total_backside',
    'magnitude_3_max_total_backside',
    'magnitude_min_total_backside',
    'magnitude_mean_ae_backside',
    'magnitude_std_ae_backside',
    'magnitude_1_max_ae_backside',
    'magnitude_2_max_ae_backside',
    'magnitude_3_max_ae_backside',
    'magnitude_min_ae_backside',
    'fundamental_frequency_x_backside',
    'fundamental_frequency_y_backside',
    'fundamental_frequency_z_backside',
    'fundamental_frequency_total_backside',
    'fundamental_frequency_ae_backside',
    'spectral_flatness_x_backside',
    'spectral_flatness_y_backside',
    'spectral_flatness_z_backside',
    'spectral_flatness_total_backside',
    'spectral_flatness_ae_backside',
]



# time domain
def feature_extraction_time_domain(sensor_data_frontside, sensor_data_backside):
    default_fc_parameters = {
        'abs_energy': None,
        'absolute_maximum': None,
        'absolute_sum_of_changes': None,
        'benford_correlation': None,
        'count_above_mean': None,
        'count_below_mean': None,
        'fft_aggregated': [{"aggtype": "centroid"}, {"aggtype": "variance"}, {"aggtype": "skew"}, {"aggtype": "kurtosis"}],
        'first_location_of_maximum': None,
        'first_location_of_minimum': None,
        'has_duplicate': None,
        'has_duplicate_max': None,
        'has_duplicate_min': None,
        'kurtosis': None,
        'last_location_of_maximum': None,
        'last_location_of_minimum': None,
        'length': None,
        'linear_trend': [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"}, {"attr": "slope"}, {"attr": "stderr"}],
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'maximum': None,
        'mean': None,
        'mean_abs_change': None,
        'mean_change': None,
        'mean_second_derivative_central': None,
        'median': None,
        'minimum': None,
        'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
        'percentage_of_reoccurring_values_to_all_values': None,
        'ratio_value_number_to_time_series_length': None,
        'root_mean_square': None,
        'skewness': None,
        'standard_deviation': None,
        'sum_of_reoccurring_data_points': None,
        'sum_of_reoccurring_values': None,
        'sum_values': None,
        'variance': None,
        'variance_larger_than_standard_deviation': None,
        'variation_coefficient': None,
    }
    
    extracted_features_frontside = tsfresh.extract_features(
        sensor_data_frontside[['timestamp', 'part_id', 'acc_x', 'acc_y', 'acc_z', 'total_acc', 'ae']],
        column_id='part_id', column_sort='timestamp',
        default_fc_parameters=default_fc_parameters,
        n_jobs=4, disable_progressbar=True
    )
    extracted_features_frontside.columns = [f"{col}_frontside" for col in extracted_features_frontside.columns]
    
    extracted_features_backside = tsfresh.extract_features(
        sensor_data_backside[['timestamp', 'part_id', 'acc_x', 'acc_y', 'acc_z', 'total_acc', 'ae']],
        column_id='part_id', column_sort='timestamp',
        default_fc_parameters=default_fc_parameters,
        n_jobs=4, disable_progressbar=True
    )
    extracted_features_backside.columns = [f"{col}_backside" for col in extracted_features_backside.columns]

    extracted_features = pd.concat([
        extracted_features_frontside.reset_index(drop=True),
        extracted_features_backside.reset_index(drop=True)
    ], axis=1)

    return extracted_features



# frequency domain
def fft_calculation(sensor_data):
    # calculate sampling frequency
    time_differences = np.diff(sensor_data['timestamp'])
    average_sampling_interval = np.mean(time_differences)
    sampling_frequency = 1 / average_sampling_interval

    # number of data points
    data_points = len(sensor_data)

    # calculate frequency
    frequency = rfftfreq(data_points, d=1/sampling_frequency).reshape(-1, 1)

    # FFT
    fft_acc_x = rfft(sensor_data['acc_x'].values).reshape(-1, 1)
    fft_acc_y = rfft(sensor_data['acc_y'].values).reshape(-1, 1)
    fft_acc_z = rfft(sensor_data['acc_z'].values).reshape(-1, 1)
    fft_total_acc = rfft(sensor_data['total_acc'].values).reshape(-1, 1)
    fft_ae = rfft(sensor_data['ae'].values).reshape(-1, 1)

    fft_values = np.hstack((fft_acc_x, fft_acc_y, fft_acc_z, fft_total_acc, fft_ae))
    columns = ['fft_acc_x', 'fft_acc_y', 'fft_acc_z', 'fft_total_acc', 'fft_ae']
    fft_sensor_data = pd.DataFrame(fft_values, columns=columns)
    fft_sensor_data['frequency'] = frequency

    return fft_sensor_data



def fundamental_frequency(fft_values, frequencies):
    magnitude_spectrum = np.abs(fft_values)
    fundamental_frequency = frequencies[np.argmax(magnitude_spectrum)]
    return fundamental_frequency


def calculate_fundamental_frequency(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    frequencies_frontside = fft_sensor_data_frontside['frequency'].values
    frequencies_backside = fft_sensor_data_backside['frequency'].values

    columns_frontside = [
        'fundamental_frequency_x_frontside',
        'fundamental_frequency_y_frontside',
        'fundamental_frequency_z_frontside',
        'fundamental_frequency_total_frontside',
        'fundamental_frequency_ae_frontside'
    ]
    columns_backside = [
        'fundamental_frequency_x_backside',
        'fundamental_frequency_y_backside',
        'fundamental_frequency_z_backside',
        'fundamental_frequency_total_backside',
        'fundamental_frequency_ae_backside'
    ]

    fundamental_frequency_x_frontside = fundamental_frequency(
        fft_values_frontside['fft_acc_x'].values,
        frequencies_frontside
    )
    fundamental_frequency_y_frontside = fundamental_frequency(
        fft_values_frontside['fft_acc_y'].values,
        frequencies_frontside
    )
    fundamental_frequency_z_frontside = fundamental_frequency(
        fft_values_frontside['fft_acc_z'].values,
        frequencies_frontside
    )
    fundamental_frequency_total_frontside = fundamental_frequency(
        fft_values_frontside['fft_total_acc'].values,
        frequencies_frontside
    )
    fundamental_frequency_ae_frontside = fundamental_frequency(
        fft_values_frontside['fft_ae'].values,
        frequencies_frontside
    )


    fundamental_frequency_frontside = pd.DataFrame([[
        fundamental_frequency_x_frontside,
        fundamental_frequency_y_frontside,
        fundamental_frequency_z_frontside,
        fundamental_frequency_total_frontside,
        fundamental_frequency_ae_frontside
    ]], columns=columns_frontside)


    fundamental_frequency_x_backside = fundamental_frequency(
        fft_values_backside['fft_acc_x'].values,
        frequencies_backside
    )
    fundamental_frequency_y_backside = fundamental_frequency(
        fft_values_backside['fft_acc_y'].values,
        frequencies_backside
    )
    fundamental_frequency_z_backside = fundamental_frequency(
        fft_values_backside['fft_acc_z'].values,
        frequencies_backside
    )
    fundamental_frequency_total_backside = fundamental_frequency(
        fft_values_backside['fft_total_acc'].values,
        frequencies_backside
    )
    fundamental_frequency_ae_backside = fundamental_frequency(
        fft_values_backside['fft_ae'].values,
        frequencies_backside
    )


    fundamental_frequency_backside = pd.DataFrame([[
        fundamental_frequency_x_backside,
        fundamental_frequency_y_backside,
        fundamental_frequency_z_backside,
        fundamental_frequency_total_backside,
        fundamental_frequency_ae_backside
    ]], columns=columns_backside)

    feature_fundamental_frequency = pd.concat([
        fundamental_frequency_frontside,
        fundamental_frequency_backside
    ], axis=1)

    return feature_fundamental_frequency



def spectral_energy(fft_values):
    magnitude_spectrum = np.abs(fft_values)
    spectral_energy = np.sum(magnitude_spectrum ** 2) / len(fft_values)
    return spectral_energy


def calculate_spectral_energy(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]


    columns_frontside = [
        'spectral_energy_x_frontside',
        'spectral_energy_y_frontside',
        'spectral_energy_z_frontside',
        'spectral_energy_total_frontside',
        'spectral_energy_ae_frontside'
    ]
    columns_backside = [
        'spectral_energy_x_backside',
        'spectral_energy_y_backside',
        'spectral_energy_z_backside',
        'spectral_energy_total_backside',
        'spectral_energy_ae_backside'
    ]

    spectral_energy_x_frontside = spectral_energy(
        fft_values_frontside['fft_acc_x'].values,
    )
    spectral_energy_y_frontside = spectral_energy(
        fft_values_frontside['fft_acc_y'].values,
    )
    spectral_energy_z_frontside = spectral_energy(
        fft_values_frontside['fft_acc_z'].values,
    )
    spectral_energy_total_frontside = spectral_energy(
        fft_values_frontside['fft_total_acc'].values,
    )
    spectral_energy_ae_frontside = spectral_energy(
        fft_values_frontside['fft_ae'].values,
    )


    spectral_energy_frontside = pd.DataFrame([[
        spectral_energy_x_frontside,
        spectral_energy_y_frontside,
        spectral_energy_z_frontside,
        spectral_energy_total_frontside,
        spectral_energy_ae_frontside
    ]], columns=columns_frontside)


    spectral_energy_x_backside = spectral_energy(
        fft_values_backside['fft_acc_x'].values,
    )
    spectral_energy_y_backside = spectral_energy(
        fft_values_backside['fft_acc_y'].values,
    )
    spectral_energy_z_backside = spectral_energy(
        fft_values_backside['fft_acc_z'].values,
    )
    spectral_energy_total_backside = spectral_energy(
        fft_values_backside['fft_total_acc'].values,
    )
    spectral_energy_ae_backside = spectral_energy(
        fft_values_backside['fft_ae'].values,
    )


    spectral_energy_backside = pd.DataFrame([[
        spectral_energy_x_backside,
        spectral_energy_y_backside,
        spectral_energy_z_backside,
        spectral_energy_total_backside,
        spectral_energy_ae_backside
    ]], columns=columns_backside)

    feature_spectral_energy = pd.concat([
        spectral_energy_frontside,
        spectral_energy_backside
    ], axis=1)

    return feature_spectral_energy


def spectral_entropy(fft_values):
    magnitude_spectrum = np.abs(fft_values)
    spectral_energy_distribution = magnitude_spectrum ** 2
    normalized_spectral_energy_distribution = spectral_energy_distribution / np.sum(spectral_energy_distribution)
    spectral_entropy = entropy(normalized_spectral_energy_distribution)
    return spectral_entropy


def calculate_spectral_entropy(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]


    columns_frontside = [
        'spectral_entropy_x_frontside',
        'spectral_entropy_y_frontside',
        'spectral_entropy_z_frontside',
        'spectral_entropy_total_frontside',
        'spectral_entropy_ae_frontside'
    ]
    columns_backside = [
        'spectral_entropy_x_backside',
        'spectral_entropy_y_backside',
        'spectral_entropy_z_backside',
        'spectral_entropy_total_backside',
        'spectral_entropy_ae_backside'
    ]

    spectral_entropy_x_frontside = spectral_entropy(
        fft_values_frontside['fft_acc_x'].values,
    )
    spectral_entropy_y_frontside = spectral_entropy(
        fft_values_frontside['fft_acc_y'].values,
    )
    spectral_entropy_z_frontside = spectral_entropy(
        fft_values_frontside['fft_acc_z'].values,
    )
    spectral_entropy_total_frontside = spectral_entropy(
        fft_values_frontside['fft_total_acc'].values,
    )
    spectral_entropy_ae_frontside = spectral_entropy(
        fft_values_frontside['fft_ae'].values,
    )


    spectral_entropy_frontside = pd.DataFrame([[
        spectral_entropy_x_frontside,
        spectral_entropy_y_frontside,
        spectral_entropy_z_frontside,
        spectral_entropy_total_frontside,
        spectral_entropy_ae_frontside
    ]], columns=columns_frontside)


    spectral_entropy_x_backside = spectral_entropy(
        fft_values_backside['fft_acc_x'].values,
    )
    spectral_entropy_y_backside = spectral_entropy(
        fft_values_backside['fft_acc_y'].values,
    )
    spectral_entropy_z_backside = spectral_entropy(
        fft_values_backside['fft_acc_z'].values,
    )
    spectral_entropy_total_backside = spectral_entropy(
        fft_values_backside['fft_total_acc'].values,
    )
    spectral_entropy_ae_backside = spectral_entropy(
        fft_values_backside['fft_ae'].values,
    )


    spectral_entropy_backside = pd.DataFrame([[
        spectral_entropy_x_backside,
        spectral_entropy_y_backside,
        spectral_entropy_z_backside,
        spectral_entropy_total_backside,
        spectral_entropy_ae_backside
    ]], columns=columns_backside)

    feature_spectral_entropy = pd.concat([
        spectral_entropy_frontside,
        spectral_entropy_backside
    ], axis=1)

    return feature_spectral_entropy



def spectral_centroid(fft_values, frequencies):
    magnitude_spectrum = np.abs(fft_values)
    power_spectrum = magnitude_spectrum ** 2
    frequencies = np.abs(frequencies)
    spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
    return spectral_centroid


def calculate_spectral_centroid(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    frequencies_frontside = fft_sensor_data_frontside['frequency'].values
    frequencies_backside = fft_sensor_data_backside['frequency'].values

    columns_frontside = [
        'spectral_centroid_x_frontside',
        'spectral_centroid_y_frontside',
        'spectral_centroid_z_frontside',
        'spectral_centroid_total_frontside',
        'spectral_centroid_ae_frontside'
    ]
    columns_backside = [
        'spectral_centroid_x_backside',
        'spectral_centroid_y_backside',
        'spectral_centroid_z_backside',
        'spectral_centroid_total_backside',
        'spectral_centroid_ae_backside'
    ]

    spectral_centroid_x_frontside = spectral_centroid(
        fft_values_frontside['fft_acc_x'].values,
        frequencies_frontside
    )
    spectral_centroid_y_frontside = spectral_centroid(
        fft_values_frontside['fft_acc_y'].values,
        frequencies_frontside
    )
    spectral_centroid_z_frontside = spectral_centroid(
        fft_values_frontside['fft_acc_z'].values,
        frequencies_frontside
    )
    spectral_centroid_total_frontside = spectral_centroid(
        fft_values_frontside['fft_total_acc'].values,
        frequencies_frontside
    )
    spectral_centroid_ae_frontside = spectral_centroid(
        fft_values_frontside['fft_ae'].values,
        frequencies_frontside
    )


    spectral_centroid_frontside = pd.DataFrame([[
        spectral_centroid_x_frontside,
        spectral_centroid_y_frontside,
        spectral_centroid_z_frontside,
        spectral_centroid_total_frontside,
        spectral_centroid_ae_frontside
    ]], columns=columns_frontside)


    spectral_centroid_x_backside = spectral_centroid(
        fft_values_backside['fft_acc_x'].values,
        frequencies_backside
    )
    spectral_centroid_y_backside = spectral_centroid(
        fft_values_backside['fft_acc_y'].values,
        frequencies_backside
    )
    spectral_centroid_z_backside = spectral_centroid(
        fft_values_backside['fft_acc_z'].values,
        frequencies_backside
    )
    spectral_centroid_total_backside = spectral_centroid(
        fft_values_backside['fft_total_acc'].values,
        frequencies_backside
    )
    spectral_centroid_ae_backside = spectral_centroid(
        fft_values_backside['fft_ae'].values,
        frequencies_backside
    )


    spectral_centroid_backside = pd.DataFrame([[
        spectral_centroid_x_backside,
        spectral_centroid_y_backside,
        spectral_centroid_z_backside,
        spectral_centroid_total_backside,
        spectral_centroid_ae_backside
    ]], columns=columns_backside)

    feature_spectral_centroid = pd.concat([
        spectral_centroid_frontside,
        spectral_centroid_backside
    ], axis=1)

    return feature_spectral_centroid



def spectral_flatness(fft_values):
    magnitude_spectrum = np.abs(fft_values)
    geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum[magnitude_spectrum > 0])))
    arithmetic_mean = np.mean(magnitude_spectrum)
    spectral_flatness = geometric_mean / arithmetic_mean
    return spectral_flatness


def calculate_spectral_flatness(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]


    columns_frontside = [
        'spectral_flatness_x_frontside',
        'spectral_flatness_y_frontside',
        'spectral_flatness_z_frontside',
        'spectral_flatness_total_frontside',
        'spectral_flatness_ae_frontside'
    ]
    columns_backside = [
        'spectral_flatness_x_backside',
        'spectral_flatness_y_backside',
        'spectral_flatness_z_backside',
        'spectral_flatness_total_backside',
        'spectral_flatness_ae_backside'
    ]

    spectral_flatness_x_frontside = spectral_flatness(
        fft_values_frontside['fft_acc_x'].values,
    )
    spectral_flatness_y_frontside = spectral_flatness(
        fft_values_frontside['fft_acc_y'].values,
    )
    spectral_flatness_z_frontside = spectral_flatness(
        fft_values_frontside['fft_acc_z'].values,
    )
    spectral_flatness_total_frontside = spectral_flatness(
        fft_values_frontside['fft_total_acc'].values,
    )
    spectral_flatness_ae_frontside = spectral_flatness(
        fft_values_frontside['fft_ae'].values,
    )


    spectral_flatness_frontside = pd.DataFrame([[
        spectral_flatness_x_frontside,
        spectral_flatness_y_frontside,
        spectral_flatness_z_frontside,
        spectral_flatness_total_frontside,
        spectral_flatness_ae_frontside
    ]], columns=columns_frontside)


    spectral_flatness_x_backside = spectral_flatness(
        fft_values_backside['fft_acc_x'].values,
    )
    spectral_flatness_y_backside = spectral_flatness(
        fft_values_backside['fft_acc_y'].values,
    )
    spectral_flatness_z_backside = spectral_flatness(
        fft_values_backside['fft_acc_z'].values,
    )
    spectral_flatness_total_backside = spectral_flatness(
        fft_values_backside['fft_total_acc'].values,
    )
    spectral_flatness_ae_backside = spectral_flatness(
        fft_values_backside['fft_ae'].values,
    )


    spectral_flatness_backside = pd.DataFrame([[
        spectral_flatness_x_backside,
        spectral_flatness_y_backside,
        spectral_flatness_z_backside,
        spectral_flatness_total_backside,
        spectral_flatness_ae_backside
    ]], columns=columns_backside)

    feature_spectral_flatness = pd.concat([
        spectral_flatness_frontside,
        spectral_flatness_backside
    ], axis=1)

    return feature_spectral_flatness



def spectral_roll_off_point(fft_values, frequencise, threshold=0.85):
    magnitude_spectrum = np.abs(fft_values)
    spectral_energy = magnitude_spectrum ** 2
    cumulative_energy = np.cumsum(spectral_energy)
    roll_off_threshold = threshold * cumulative_energy[-1]
    roll_off_point = np.where(cumulative_energy >= roll_off_threshold)[0][0]
    roll_off_frequency = frequencise[roll_off_point]
    return roll_off_frequency



def calculate_spectral_roll_off_point(fft_sensor_data_frontside, fft_sensor_data_backside, threshold=0.85):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    frequencies_frontside = fft_sensor_data_frontside['frequency'].values
    frequencies_backside = fft_sensor_data_backside['frequency'].values

    columns_frontside = [
        'spectral_roll_off_point_x_frontside',
        'spectral_roll_off_point_y_frontside',
        'spectral_roll_off_point_z_frontside',
        'spectral_roll_off_point_total_frontside',
        'spectral_roll_off_point_ae_frontside'
    ]
    columns_backside = [
        'spectral_roll_off_point_x_backside',
        'spectral_roll_off_point_y_backside',
        'spectral_roll_off_point_z_backside',
        'spectral_roll_off_point_total_backside',
        'spectral_roll_off_point_ae_backside'
    ]

    spectral_roll_off_point_x_frontside = spectral_roll_off_point(
        fft_values_frontside['fft_acc_x'].values,
        frequencies_frontside,
        threshold
    )
    spectral_roll_off_point_y_frontside = spectral_roll_off_point(
        fft_values_frontside['fft_acc_y'].values,
        frequencies_frontside,
        threshold
    )
    spectral_roll_off_point_z_frontside = spectral_roll_off_point(
        fft_values_frontside['fft_acc_z'].values,
        frequencies_frontside,
        threshold
    )
    spectral_roll_off_point_total_frontside = spectral_roll_off_point(
        fft_values_frontside['fft_total_acc'].values,
        frequencies_frontside,
        threshold
    )
    spectral_roll_off_point_ae_frontside = spectral_roll_off_point(
        fft_values_frontside['fft_ae'].values,
        frequencies_frontside,
        threshold
    )


    spectral_roll_off_point_frontside = pd.DataFrame([[
        spectral_roll_off_point_x_frontside,
        spectral_roll_off_point_y_frontside,
        spectral_roll_off_point_z_frontside,
        spectral_roll_off_point_total_frontside,
        spectral_roll_off_point_ae_frontside
    ]], columns=columns_frontside)


    spectral_roll_off_point_x_backside = spectral_roll_off_point(
        fft_values_backside['fft_acc_x'].values,
        frequencies_backside,
        threshold
    )
    spectral_roll_off_point_y_backside = spectral_roll_off_point(
        fft_values_backside['fft_acc_y'].values,
        frequencies_backside,
        threshold
    )
    spectral_roll_off_point_z_backside = spectral_roll_off_point(
        fft_values_backside['fft_acc_z'].values,
        frequencies_backside,
        threshold
    )
    spectral_roll_off_point_total_backside = spectral_roll_off_point(
        fft_values_backside['fft_total_acc'].values,
        frequencies_backside,
        threshold
    )
    spectral_roll_off_point_ae_backside = spectral_roll_off_point(
        fft_values_backside['fft_ae'].values,
        frequencies_backside,
        threshold
    )


    spectral_roll_off_point_backside = pd.DataFrame([[
        spectral_roll_off_point_x_backside,
        spectral_roll_off_point_y_backside,
        spectral_roll_off_point_z_backside,
        spectral_roll_off_point_total_backside,
        spectral_roll_off_point_ae_backside
    ]], columns=columns_backside)

    feature_spectral_roll_off_point = pd.concat([
        spectral_roll_off_point_frontside,
        spectral_roll_off_point_backside
    ], axis=1)

    return feature_spectral_roll_off_point


def spectral_skewness(fft_values):
    magnitude_spectrum = np.abs(fft_values)
    spectral_skewness = skew(magnitude_spectrum)
    return spectral_skewness


def calculate_spectral_skewness(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]


    columns_frontside = [
        'spectral_skewness_x_frontside',
        'spectral_skewness_y_frontside',
        'spectral_skewness_z_frontside',
        'spectral_skewness_total_frontside',
        'spectral_skewness_ae_frontside'
    ]
    columns_backside = [
        'spectral_skewness_x_backside',
        'spectral_skewness_y_backside',
        'spectral_skewness_z_backside',
        'spectral_skewness_total_backside',
        'spectral_skewness_ae_backside'
    ]

    spectral_skewness_x_frontside = spectral_skewness(
        fft_values_frontside['fft_acc_x'].values,
    )
    spectral_skewness_y_frontside = spectral_skewness(
        fft_values_frontside['fft_acc_y'].values,
    )
    spectral_skewness_z_frontside = spectral_skewness(
        fft_values_frontside['fft_acc_z'].values,
    )
    spectral_skewness_total_frontside = spectral_skewness(
        fft_values_frontside['fft_total_acc'].values,
    )
    spectral_skewness_ae_frontside = spectral_skewness(
        fft_values_frontside['fft_ae'].values,
    )


    spectral_skewness_frontside = pd.DataFrame([[
        spectral_skewness_x_frontside,
        spectral_skewness_y_frontside,
        spectral_skewness_z_frontside,
        spectral_skewness_total_frontside,
        spectral_skewness_ae_frontside
    ]], columns=columns_frontside)


    spectral_skewness_x_backside = spectral_skewness(
        fft_values_backside['fft_acc_x'].values,
    )
    spectral_skewness_y_backside = spectral_skewness(
        fft_values_backside['fft_acc_y'].values,
    )
    spectral_skewness_z_backside = spectral_skewness(
        fft_values_backside['fft_acc_z'].values,
    )
    spectral_skewness_total_backside = spectral_skewness(
        fft_values_backside['fft_total_acc'].values,
    )
    spectral_skewness_ae_backside = spectral_skewness(
        fft_values_backside['fft_ae'].values,
    )


    spectral_skewness_backside = pd.DataFrame([[
        spectral_skewness_x_backside,
        spectral_skewness_y_backside,
        spectral_skewness_z_backside,
        spectral_skewness_total_backside,
        spectral_skewness_ae_backside
    ]], columns=columns_backside)

    feature_spectral_skewness = pd.concat([
        spectral_skewness_frontside,
        spectral_skewness_backside
    ], axis=1)

    return feature_spectral_skewness


def spectral_kurtosis(fft_values):
    magnitude_spectrum = np.abs(fft_values)
    spectral_kurtosis = kurtosis(magnitude_spectrum)
    return spectral_kurtosis


def calculate_spectral_kurtosis(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]

    columns_frontside = [
        'spectral_kurtosis_x_frontside',
        'spectral_kurtosis_y_frontside',
        'spectral_kurtosis_z_frontside',
        'spectral_kurtosis_total_frontside',
        'spectral_kurtosis_ae_frontside'
    ]
    columns_backside = [
        'spectral_kurtosis_x_backside',
        'spectral_kurtosis_y_backside',
        'spectral_kurtosis_z_backside',
        'spectral_kurtosis_total_backside',
        'spectral_kurtosis_ae_backside'
    ]

    spectral_kurtosis_x_frontside = spectral_kurtosis(
        fft_values_frontside['fft_acc_x'].values,
    )
    spectral_kurtosis_y_frontside = spectral_kurtosis(
        fft_values_frontside['fft_acc_y'].values,
    )
    spectral_kurtosis_z_frontside = spectral_kurtosis(
        fft_values_frontside['fft_acc_z'].values,
    )
    spectral_kurtosis_total_frontside = spectral_kurtosis(
        fft_values_frontside['fft_total_acc'].values,
    )
    spectral_kurtosis_ae_frontside = spectral_kurtosis(
        fft_values_frontside['fft_ae'].values,
    )


    spectral_kurtosis_frontside = pd.DataFrame([[
        spectral_kurtosis_x_frontside,
        spectral_kurtosis_y_frontside,
        spectral_kurtosis_z_frontside,
        spectral_kurtosis_total_frontside,
        spectral_kurtosis_ae_frontside
    ]], columns=columns_frontside)


    spectral_kurtosis_x_backside = spectral_kurtosis(
        fft_values_backside['fft_acc_x'].values,
    )
    spectral_kurtosis_y_backside = spectral_kurtosis(
        fft_values_backside['fft_acc_y'].values,
    )
    spectral_kurtosis_z_backside = spectral_kurtosis(
        fft_values_backside['fft_acc_z'].values,
    )
    spectral_kurtosis_total_backside = spectral_kurtosis(
        fft_values_backside['fft_total_acc'].values,
    )
    spectral_kurtosis_ae_backside = spectral_kurtosis(
        fft_values_backside['fft_ae'].values,
    )


    spectral_kurtosis_backside = pd.DataFrame([[
        spectral_kurtosis_x_backside,
        spectral_kurtosis_y_backside,
        spectral_kurtosis_z_backside,
        spectral_kurtosis_total_backside,
        spectral_kurtosis_ae_backside
    ]], columns=columns_backside)

    feature_spectral_kurtosis = pd.concat([
        spectral_kurtosis_frontside,
        spectral_kurtosis_backside
    ], axis=1)

    return feature_spectral_kurtosis


def spectral_peaks(fft_values, frequencies, threshold=0.95):
    """
    This function calculate some features about spectral peaks.
    1. statistical features:
        - num_peaks
        - peak_height
            - mean_peak_height
            - std_peak_height
            - max_peak_height
            - min_peak_height
        - peak frequency
            - mean_peak_frequency
            - std_peak_frequency
            - max_peak_frequency
            - min_peak_frequency
    2. energy features
        - peak energy
            - mean_peak_energy
            - std_peak_energy
            - max_peak_energy
            - min_peak_energy
        - peak energy ratio
            - mean_peak_energy_ratio
            - std_peak_energy_ratio
            - max_peak_energy_ratio
            - min_peak_energy_ratio
        - energy concentration
            - num_significant_peaks
    3. structural features
        - peak distances
            - mean_peak_distance
            - std_peak_distance
            - max_peak_distance
            - min_peak_distance
    """
    magnitude_spectrum = np.abs(fft_values)
    # recognize peaks
    peaks, _ = find_peaks(magnitude_spectrum)
    peak_frequencies = frequencies[peaks]
    
    # peak number
    num_peaks = len(peak_frequencies)
    
    # statistical features about peak height
    peak_heights = magnitude_spectrum[peaks]
    mean_peak_height = np.mean(peak_heights)
    std_peak_height = np.std(peak_heights)
    max_peak_height = np.max(peak_heights)
    min_peak_height = np.min(peak_heights)

    # statistical features about peak frequency
    mean_peak_frequency = np.mean(peak_frequencies)
    std_peak_frequency = np.std(peak_frequencies)
    max_peak_frequency = np.max(peak_frequencies)
    min_peak_frequency = np.min(peak_frequencies)

    # energy fetures
    total_energy = np.sum(magnitude_spectrum ** 2)
    
    peak_energies = magnitude_spectrum[peaks] ** 2
    mean_peak_energy = np.mean(peak_energies)
    std_peak_energy = np.std(peak_energies)
    max_peak_energy = np.max(peak_energies)
    min_peak_energy = np.min(peak_energies)
    
    peak_energy_ratio = peak_energies / total_energy
    mean_peak_energy_ratio = np.mean(peak_energy_ratio)
    std_peak_energy_ratio = np.std(peak_energy_ratio)
    max_peak_energy_ratio = np.max(peak_energy_ratio)
    min_peak_energy_ratio = np.min(peak_energy_ratio)
    
    # energy concentration
    energy_sort_index = np.argsort(-peak_energies)
    cumulative_energy = np.cumsum(peak_energies[energy_sort_index])
    significant_peaks = np.where(cumulative_energy <= threshold * total_energy)[0]
    num_significant_peaks = len(significant_peaks)

    # peak structural features
    peak_distances = np.diff(peak_frequencies)
    mean_peak_distance = np.mean(peak_distances) if len(peak_distances) > 0 else 0
    std_peak_distance = np.std(peak_distances) if len(peak_distances) > 0 else 0
    max_peak_distance = np.max(peak_distances) if len(peak_distances) > 0 else 0
    min_peak_distance = np.min(peak_distances) if len(peak_distances) > 0 else 0

    features = np.array([
        num_peaks,
        mean_peak_height, std_peak_height, max_peak_height, min_peak_height,
        mean_peak_frequency, std_peak_frequency, max_peak_frequency, min_peak_frequency,
        mean_peak_energy, std_peak_energy, max_peak_energy, min_peak_energy,
        mean_peak_energy_ratio, std_peak_energy_ratio, max_peak_energy_ratio, min_peak_energy_ratio,
        num_significant_peaks,
        mean_peak_distance, std_peak_distance, max_peak_distance, min_peak_distance
    ])

    return features.reshape(1, -1)



def calculate_spectral_peaks(fft_sensor_data_frontside, fft_sensor_data_backside, threshold=0.95):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    frequencies_frontside = fft_sensor_data_frontside['frequency'].values
    frequencies_backside = fft_sensor_data_backside['frequency'].values

    columns_x_frontside = [
        'num_peaks_x_frontside',
        'mean_peak_height_x_frontside',
        'std_peak_height_x_frontside',
        'max_peak_height_x_frontside',
        'min_peak_height_x_frontside',
        'mean_peak_frequency_x_frontside',
        'std_peak_frequency_x_frontside',
        'max_peak_frequency_x_frontside',
        'min_peak_frequency_x_frontside',
        'mean_peak_energy_x_frontside',
        'std_peak_energy_x_frontside',
        'max_peak_energy_x_frontside',
        'min_peak_energy_x_frontside',
        'mean_peak_energy_ratio_x_frontside',
        'std_peak_energy_ratio_x_frontside',
        'max_peak_energy_ratio_x_frontside',
        'min_peak_energy_ratio_x_frontside',
        'num_significant_peaks_x_frontside',
        'mean_peak_distance_x_frontside',
        'std_peak_distance_x_frontside',
        'max_peak_distance_x_frontside',
        'min_peak_distance_x_frontside'
    ]
    columns_y_frontside = [
        'num_peaks_y_frontside',
        'mean_peak_height_y_frontside',
        'std_peak_height_y_frontside',
        'max_peak_height_y_frontside',
        'min_peak_height_y_frontside',
        'mean_peak_frequency_y_frontside',
        'std_peak_frequency_y_frontside',
        'max_peak_frequency_y_frontside',
        'min_peak_frequency_y_frontside',
        'mean_peak_energy_y_frontside',
        'std_peak_energy_y_frontside',
        'max_peak_energy_y_frontside',
        'min_peak_energy_y_frontside',
        'mean_peak_energy_ratio_y_frontside',
        'std_peak_energy_ratio_y_frontside',
        'max_peak_energy_ratio_y_frontside',
        'min_peak_energy_ratio_y_frontside',
        'num_significant_peaks_y_frontside',
        'mean_peak_distance_y_frontside',
        'std_peak_distance_y_frontside',
        'max_peak_distance_y_frontside',
        'min_peak_distance_y_frontside'
    ]
    columns_z_frontside = [
        'num_peaks_z_frontside',
        'mean_peak_height_z_frontside',
        'std_peak_height_z_frontside',
        'max_peak_height_z_frontside',
        'min_peak_height_z_frontside',
        'mean_peak_frequency_z_frontside',
        'std_peak_frequency_z_frontside',
        'max_peak_frequency_z_frontside',
        'min_peak_frequency_z_frontside',
        'mean_peak_energy_z_frontside',
        'std_peak_energy_z_frontside',
        'max_peak_energy_z_frontside',
        'min_peak_energy_z_frontside',
        'mean_peak_energy_ratio_z_frontside',
        'std_peak_energy_ratio_z_frontside',
        'max_peak_energy_ratio_z_frontside',
        'min_peak_energy_ratio_z_frontside',
        'num_significant_peaks_z_frontside',
        'mean_peak_distance_z_frontside',
        'std_peak_distance_z_frontside',
        'max_peak_distance_z_frontside',
        'min_peak_distance_z_frontside'
    ]
    columns_total_frontside = [
        'num_peaks_total_frontside',
        'mean_peak_height_total_frontside',
        'std_peak_height_total_frontside',
        'max_peak_height_total_frontside',
        'min_peak_height_total_frontside',
        'mean_peak_frequency_total_frontside',
        'std_peak_frequency_total_frontside',
        'max_peak_frequency_total_frontside',
        'min_peak_frequency_total_frontside',
        'mean_peak_energy_total_frontside',
        'std_peak_energy_total_frontside',
        'max_peak_energy_total_frontside',
        'min_peak_energy_total_frontside',
        'mean_peak_energy_ratio_total_frontside',
        'std_peak_energy_ratio_total_frontside',
        'max_peak_energy_ratio_total_frontside',
        'min_peak_energy_ratio_total_frontside',
        'num_significant_peaks_total_frontside',
        'mean_peak_distance_total_frontside',
        'std_peak_distance_total_frontside',
        'max_peak_distance_total_frontside',
        'min_peak_distance_total_frontside'
    ]
    columns_ae_frontside = [
        'num_peaks_ae_frontside',
        'mean_peak_height_ae_frontside',
        'std_peak_height_ae_frontside',
        'max_peak_height_ae_frontside',
        'min_peak_height_ae_frontside',
        'mean_peak_frequency_ae_frontside',
        'std_peak_frequency_ae_frontside',
        'max_peak_frequency_ae_frontside',
        'min_peak_frequency_ae_frontside',
        'mean_peak_energy_ae_frontside',
        'std_peak_energy_ae_frontside',
        'max_peak_energy_ae_frontside',
        'min_peak_energy_ae_frontside',
        'mean_peak_energy_ratio_ae_frontside',
        'std_peak_energy_ratio_ae_frontside',
        'max_peak_energy_ratio_ae_frontside',
        'min_peak_energy_ratio_ae_frontside',
        'num_significant_peaks_ae_frontside',
        'mean_peak_distance_ae_frontside',
        'std_peak_distance_ae_frontside',
        'max_peak_distance_ae_frontside',
        'min_peak_distance_ae_frontside'
    ]
    columns_x_backside = [
        'num_peaks_x_backside',
        'mean_peak_height_x_backside',
        'std_peak_height_x_backside',
        'max_peak_height_x_backside',
        'min_peak_height_x_backside',
        'mean_peak_frequency_x_backside',
        'std_peak_frequency_x_backside',
        'max_peak_frequency_x_backside',
        'min_peak_frequency_x_backside',
        'mean_peak_energy_x_backside',
        'std_peak_energy_x_backside',
        'max_peak_energy_x_backside',
        'min_peak_energy_x_backside',
        'mean_peak_energy_ratio_x_backside',
        'std_peak_energy_ratio_x_backside',
        'max_peak_energy_ratio_x_backside',
        'min_peak_energy_ratio_x_backside',
        'num_significant_peaks_x_backside',
        'mean_peak_distance_x_backside',
        'std_peak_distance_x_backside',
        'max_peak_distance_x_backside',
        'min_peak_distance_x_backside'
    ]
    columns_y_backside = [
        'num_peaks_y_backside',
        'mean_peak_height_y_backside',
        'std_peak_height_y_backside',
        'max_peak_height_y_backside',
        'min_peak_height_y_backside',
        'mean_peak_frequency_y_backside',
        'std_peak_frequency_y_backside',
        'max_peak_frequency_y_backside',
        'min_peak_frequency_y_backside',
        'mean_peak_energy_y_backside',
        'std_peak_energy_y_backside',
        'max_peak_energy_y_backside',
        'min_peak_energy_y_backside',
        'mean_peak_energy_ratio_y_backside',
        'std_peak_energy_ratio_y_backside',
        'max_peak_energy_ratio_y_backside',
        'min_peak_energy_ratio_y_backside',
        'num_significant_peaks_y_backside',
        'mean_peak_distance_y_backside',
        'std_peak_distance_y_backside',
        'max_peak_distance_y_backside',
        'min_peak_distance_y_backside'
    ]
    columns_z_backside = [
        'num_peaks_z_backside',
        'mean_peak_height_z_backside',
        'std_peak_height_z_backside',
        'max_peak_height_z_backside',
        'min_peak_height_z_backside',
        'mean_peak_frequency_z_backside',
        'std_peak_frequency_z_backside',
        'max_peak_frequency_z_backside',
        'min_peak_frequency_z_backside',
        'mean_peak_energy_z_backside',
        'std_peak_energy_z_backside',
        'max_peak_energy_z_backside',
        'min_peak_energy_z_backside',
        'mean_peak_energy_ratio_z_backside',
        'std_peak_energy_ratio_z_backside',
        'max_peak_energy_ratio_z_backside',
        'min_peak_energy_ratio_z_backside',
        'num_significant_peaks_z_backside',
        'mean_peak_distance_z_backside',
        'std_peak_distance_z_backside',
        'max_peak_distance_z_backside',
        'min_peak_distance_z_backside'
    ]
    columns_total_backside = [
        'num_peaks_total_backside',
        'mean_peak_height_total_backside',
        'std_peak_height_total_backside',
        'max_peak_height_total_backside',
        'min_peak_height_total_backside',
        'mean_peak_frequency_total_backside',
        'std_peak_frequency_total_backside',
        'max_peak_frequency_total_backside',
        'min_peak_frequency_total_backside',
        'mean_peak_energy_total_backside',
        'std_peak_energy_total_backside',
        'max_peak_energy_total_backside',
        'min_peak_energy_total_backside',
        'mean_peak_energy_ratio_total_backside',
        'std_peak_energy_ratio_total_backside',
        'max_peak_energy_ratio_total_backside',
        'min_peak_energy_ratio_total_backside',
        'num_significant_peaks_total_backside',
        'mean_peak_distance_total_backside',
        'std_peak_distance_total_backside',
        'max_peak_distance_total_backside',
        'min_peak_distance_total_backside'
    ]
    columns_ae_backside = [
        'num_peaks_ae_backside',
        'mean_peak_height_ae_backside',
        'std_peak_height_ae_backside',
        'max_peak_height_ae_backside',
        'min_peak_height_ae_backside',
        'mean_peak_frequency_ae_backside',
        'std_peak_frequency_ae_backside',
        'max_peak_frequency_ae_backside',
        'min_peak_frequency_ae_backside',
        'mean_peak_energy_ae_backside',
        'std_peak_energy_ae_backside',
        'max_peak_energy_ae_backside',
        'min_peak_energy_ae_backside',
        'mean_peak_energy_ratio_ae_backside',
        'std_peak_energy_ratio_ae_backside',
        'max_peak_energy_ratio_ae_backside',
        'min_peak_energy_ratio_ae_backside',
        'num_significant_peaks_ae_backside',
        'mean_peak_distance_ae_backside',
        'std_peak_distance_ae_backside',
        'max_peak_distance_ae_backside',
        'min_peak_distance_ae_backside'
    ]

    spectral_peaks_x_frontside = pd.DataFrame(spectral_peaks(
        fft_values_frontside['fft_acc_x'].values,
        frequencies_frontside
    ).reshape(1, -1), columns=columns_x_frontside)
    spectral_peaks_y_frontside = pd.DataFrame(spectral_peaks(
        fft_values_frontside['fft_acc_y'].values,
        frequencies_frontside
    ).reshape(1, -1), columns=columns_y_frontside)
    spectral_peaks_z_frontside = pd.DataFrame(spectral_peaks(
        fft_values_frontside['fft_acc_z'].values,
        frequencies_frontside
    ).reshape(1, -1), columns=columns_z_frontside)
    spectral_peaks_total_frontside = pd.DataFrame(spectral_peaks(
        fft_values_frontside['fft_total_acc'].values,
        frequencies_frontside
    ).reshape(1, -1), columns=columns_total_frontside)
    spectral_peaks_ae_frontside = pd.DataFrame(spectral_peaks(
        fft_values_frontside['fft_ae'].values,
        frequencies_frontside
    ).reshape(1, -1), columns=columns_ae_frontside)


    spectral_peaks_frontside = pd.concat([
        spectral_peaks_x_frontside,
        spectral_peaks_y_frontside,
        spectral_peaks_z_frontside,
        spectral_peaks_total_frontside,
        spectral_peaks_ae_frontside
    ], axis=1)

    spectral_peaks_x_backside = pd.DataFrame(spectral_peaks(
        fft_values_backside['fft_acc_x'].values,
        frequencies_backside
    ).reshape(1, -1), columns=columns_x_backside)
    spectral_peaks_y_backside = pd.DataFrame(spectral_peaks(
        fft_values_backside['fft_acc_y'].values,
        frequencies_backside
    ).reshape(1, -1), columns=columns_y_backside)
    spectral_peaks_z_backside = pd.DataFrame(spectral_peaks(
        fft_values_backside['fft_acc_z'].values,
        frequencies_backside
    ).reshape(1, -1), columns=columns_z_backside)
    spectral_peaks_total_backside = pd.DataFrame(spectral_peaks(
        fft_values_backside['fft_total_acc'].values,
        frequencies_backside
    ).reshape(1, -1), columns=columns_total_backside)
    spectral_peaks_ae_backside = pd.DataFrame(spectral_peaks(
        fft_values_backside['fft_ae'].values,
        frequencies_backside
    ).reshape(1, -1), columns=columns_ae_backside)


    spectral_peaks_backside = pd.concat([
        spectral_peaks_x_backside,
        spectral_peaks_y_backside,
        spectral_peaks_z_backside,
        spectral_peaks_total_backside,
        spectral_peaks_ae_backside
    ], axis=1)

    feature_spectral_peaks = pd.concat([
        spectral_peaks_frontside,
        spectral_peaks_backside
    ], axis=1)

    return feature_spectral_peaks



def weighted_mean_frequency(fft_values, frequencies):
    magnitude_spectrum = np.abs(fft_values)
    weighted_mean_frequency = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum)
    return weighted_mean_frequency


def calculate_weighted_mean_frequency(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    frequencies_frontside = fft_sensor_data_frontside['frequency'].values
    frequencies_backside = fft_sensor_data_backside['frequency'].values

    columns_frontside = [
        'weighted_mean_frequency_x_frontside',
        'weighted_mean_frequency_y_frontside',
        'weighted_mean_frequency_z_frontside',
        'weighted_mean_frequency_total_frontside',
        'weighted_mean_frequency_ae_frontside'
    ]
    columns_backside = [
        'weighted_mean_frequency_x_backside',
        'weighted_mean_frequency_y_backside',
        'weighted_mean_frequency_z_backside',
        'weighted_mean_frequency_total_backside',
        'weighted_mean_frequency_ae_backside'
    ]

    weighted_mean_frequency_x_frontside = weighted_mean_frequency(
        fft_values_frontside['fft_acc_x'].values,
        frequencies_frontside
    )
    weighted_mean_frequency_y_frontside = weighted_mean_frequency(
        fft_values_frontside['fft_acc_y'].values,
        frequencies_frontside
    )
    weighted_mean_frequency_z_frontside = weighted_mean_frequency(
        fft_values_frontside['fft_acc_z'].values,
        frequencies_frontside
    )
    weighted_mean_frequency_total_frontside = weighted_mean_frequency(
        fft_values_frontside['fft_total_acc'].values,
        frequencies_frontside
    )
    weighted_mean_frequency_ae_frontside = weighted_mean_frequency(
        fft_values_frontside['fft_ae'].values,
        frequencies_frontside
    )


    weighted_mean_frequency_frontside = pd.DataFrame([[
        weighted_mean_frequency_x_frontside,
        weighted_mean_frequency_y_frontside,
        weighted_mean_frequency_z_frontside,
        weighted_mean_frequency_total_frontside,
        weighted_mean_frequency_ae_frontside
    ]], columns=columns_frontside)


    weighted_mean_frequency_x_backside = weighted_mean_frequency(
        fft_values_backside['fft_acc_x'].values,
        frequencies_backside
    )
    weighted_mean_frequency_y_backside = weighted_mean_frequency(
        fft_values_backside['fft_acc_y'].values,
        frequencies_backside
    )
    weighted_mean_frequency_z_backside = weighted_mean_frequency(
        fft_values_backside['fft_acc_z'].values,
        frequencies_backside
    )
    weighted_mean_frequency_total_backside = weighted_mean_frequency(
        fft_values_backside['fft_total_acc'].values,
        frequencies_backside
    )
    weighted_mean_frequency_ae_backside = weighted_mean_frequency(
        fft_values_backside['fft_ae'].values,
        frequencies_backside
    )


    weighted_mean_frequency_backside = pd.DataFrame([[
        weighted_mean_frequency_x_backside,
        weighted_mean_frequency_y_backside,
        weighted_mean_frequency_z_backside,
        weighted_mean_frequency_total_backside,
        weighted_mean_frequency_ae_backside
    ]], columns=columns_backside)

    feature_weighted_mean_frequency = pd.concat([
        weighted_mean_frequency_frontside,
        weighted_mean_frequency_backside
    ], axis=1)

    return feature_weighted_mean_frequency


def spectral_coefficient_of_variation(fft_result):
    magnitude_spectrum = np.abs(fft_result)
    std_magnitude = np.std(magnitude_spectrum)
    mean_magnitude = np.mean(magnitude_spectrum)
    spectral_coefficient_of_variation = std_magnitude / mean_magnitude if mean_magnitude !=0 else 0
    return spectral_coefficient_of_variation


def calculate_spectral_coefficient_of_variation(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]


    columns_frontside = [
        'spectral_coefficient_of_variation_x_frontside',
        'spectral_coefficient_of_variation_y_frontside',
        'spectral_coefficient_of_variation_z_frontside',
        'spectral_coefficient_of_variazion_total_frontside',
        'spectral_coefficient_of_variation_ae_frontside'
    ]
    columns_backside = [
        'spectral_coefficient_of_variation_x_backside',
        'spectral_coefficient_of_variation_y_backside',
        'spectral_coefficient_of_variation_z_backside',
        'spectral_coefficient_of_variazion_total_backside',
        'spectral_coefficient_of_variation_ae_backside'
    ]

    spectral_coefficient_of_variation_x_frontside = spectral_coefficient_of_variation(
        fft_values_frontside['fft_acc_x'].values,
    )
    spectral_coefficient_of_variation_y_frontside = spectral_coefficient_of_variation(
        fft_values_frontside['fft_acc_y'].values,
    )
    spectral_coefficient_of_variation_z_frontside = spectral_coefficient_of_variation(
        fft_values_frontside['fft_acc_z'].values,
    )
    spectral_coefficient_of_variation_total_frontside = spectral_coefficient_of_variation(
        fft_values_frontside['fft_total_acc'].values,
    )
    spectral_coefficient_of_variation_ae_frontside = spectral_coefficient_of_variation(
        fft_values_frontside['fft_ae'].values,
    )


    spectral_coefficient_of_variation_frontside = pd.DataFrame([[
        spectral_coefficient_of_variation_x_frontside,
        spectral_coefficient_of_variation_y_frontside,
        spectral_coefficient_of_variation_z_frontside,
        spectral_coefficient_of_variation_total_frontside,
        spectral_coefficient_of_variation_ae_frontside
    ]], columns=columns_frontside)


    spectral_coefficient_of_variation_x_backside = spectral_coefficient_of_variation(
        fft_values_backside['fft_acc_x'].values,
    )
    spectral_coefficient_of_variation_y_backside = spectral_coefficient_of_variation(
        fft_values_backside['fft_acc_y'].values,
    )
    spectral_coefficient_of_variation_z_backside = spectral_coefficient_of_variation(
        fft_values_backside['fft_acc_z'].values,
    )
    spectral_coefficient_of_variation_total_backside = spectral_coefficient_of_variation(
        fft_values_backside['fft_total_acc'].values,
    )
    spectral_coefficient_of_variation_ae_backside = spectral_coefficient_of_variation(
        fft_values_backside['fft_ae'].values,
    )


    spectral_coefficient_of_variation_backside = pd.DataFrame([[
        spectral_coefficient_of_variation_x_backside,
        spectral_coefficient_of_variation_y_backside,
        spectral_coefficient_of_variation_z_backside,
        spectral_coefficient_of_variation_total_backside,
        spectral_coefficient_of_variation_ae_backside
    ]], columns=columns_backside)

    feature_spectral_coefficient_of_variation = pd.concat([
        spectral_coefficient_of_variation_frontside,
        spectral_coefficient_of_variation_backside
    ], axis=1)

    return feature_spectral_coefficient_of_variation


def power_spectral_density(fft_values, frequencies):
    power_spectrum = np.abs(fft_values) ** 2
    psd = power_spectrum / len(fft_values)

    # peak feature
    peaks, _ = find_peaks(psd)
    peak_values = psd[peaks]
    peak_frequencies = frequencies[peaks]
    
    psd_num_peaks = len(peak_values)
    psd_mean_peak_value = np.mean(peak_values)
    psd_std_peak_value = np.std(peak_values)
    psd_max_peak_value = np.max(peak_values)
    psd_min_peak_value = np.min(peak_values)

    psd_mean_peak_frequency = np.mean(peak_frequencies)
    psd_std_peak_frequency = np.std(peak_frequencies)
    psd_max_peak_frequency = np.max(peak_frequencies)
    psd_min_peak_frequency = np.min(peak_frequencies)

    if len(peak_frequencies) > 1:
        peak_distances = np.diff(peak_frequencies)
        psd_mean_peak_distance = np.mean(peak_distances)
        psd_std_peak_distance = np.std(peak_distances)
        psd_max_peak_distance = np.max(peak_distances)
        psd_min_peak_distance = np.min(peak_distances)
    else:
        psd_mean_peak_distance = 0
        psd_std_peak_distance = 0
        psd_max_peak_distance = 0
        psd_min_peak_distance = 0
        
    # statistical feature
    mean_psd = np.mean(psd)
    std_psd = np.std(psd)
    max_psd = np.max(psd)
    min_psd = np.min(psd)
    skewness_psd = skew(psd)
    kurtosis_psd = kurtosis(psd)

    # energy concentration
    power_centroid_psd = np.sum(frequencies * psd) / np.sum(psd)

    features = np.array([
        psd_num_peaks,
        psd_mean_peak_value, psd_std_peak_value, psd_max_peak_value, psd_min_peak_value,
        psd_mean_peak_frequency, psd_std_peak_frequency, psd_max_peak_frequency, psd_min_peak_frequency,
        psd_mean_peak_distance, psd_std_peak_distance, psd_max_peak_distance, psd_min_peak_distance,
        mean_psd, std_psd, max_psd, min_psd, skewness_psd, kurtosis_psd,
        power_centroid_psd
    ])

    return features



def calculate_power_spectral_density(fft_sensor_data_frontside, fft_sensor_data_backside, threshold=0.95):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    frequencies_frontside = fft_sensor_data_frontside['frequency'].values
    frequencies_backside = fft_sensor_data_backside['frequency'].values

    columns_x_frontside = [
        'psd_num_peaks_x_frontside',
        'psd_mean_peak_value_x_frontside',
        'psd_std_peak_value_x_frontside',
        'psd_max_peak_value_x_frontside',
        'psd_min_peak_value_x_frontside',
        'psd_mean_peak_frequency_x_frontside',
        'psd_std_peak_frequency_x_frontside',
        'psd_max_peak_frequency_x_frontside',
        'psd_min_peak_frequency_x_frontside',
        'psd_mean_peak_distance_x_frontside',
        'psd_std_peak_distance_x_frontside',
        'psd_max_peak_distance_x_frontside',
        'psd_min_peak_distance_x_frontside',
        'mean_psd_x_frontside',
        'std_psd_x_frontside',
        'max_psd_x_frontside',
        'min_psd_x_frontside',
        'skewness_psd_x_frontside',
        'kurtosis_psd_x_frontside',
        'power_centroid_psd_x_frontside'
    ]
    columns_y_frontside = [
        'psd_num_peaks_y_frontside',
        'psd_mean_peak_value_y_frontside',
        'psd_std_peak_value_y_frontside',
        'psd_max_peak_value_y_frontside',
        'psd_min_peak_value_y_frontside',
        'psd_mean_peak_frequency_y_frontside',
        'psd_std_peak_frequency_y_frontside',
        'psd_max_peak_frequency_y_frontside',
        'psd_min_peak_frequency_y_frontside',
        'psd_mean_peak_distance_y_frontside',
        'psd_std_peak_distance_y_frontside',
        'psd_max_peak_distance_y_frontside',
        'psd_min_peak_distance_y_frontside',
        'mean_psd_y_frontside',
        'std_psd_y_frontside',
        'max_psd_y_frontside',
        'min_psd_y_frontside',
        'skewness_psd_y_frontside',
        'kurtosis_psd_y_frontside',
        'power_centroid_psd_y_frontside'
    ]
    columns_z_frontside = [
        'psd_num_peaks_z_frontside',
        'psd_mean_peak_value_z_frontside',
        'psd_std_peak_value_z_frontside',
        'psd_max_peak_value_z_frontside',
        'psd_min_peak_value_z_frontside',
        'psd_mean_peak_frequency_z_frontside',
        'psd_std_peak_frequency_z_frontside',
        'psd_max_peak_frequency_z_frontside',
        'psd_min_peak_frequency_z_frontside',
        'psd_mean_peak_distance_z_frontside',
        'psd_std_peak_distance_z_frontside',
        'psd_max_peak_distance_z_frontside',
        'psd_min_peak_distance_z_frontside',
        'mean_psd_z_frontside',
        'std_psd_z_frontside', 
        'max_psd_z_frontside', 
        'min_psd_z_frontside', 
        'skewness_psd_z_frontside', 
        'kurtosis_psd_z_frontside',
        'power_centroid_psd_z_frontside'
    ]
    columns_total_frontside = [
        'psd_num_peaks_total_frontside',
        'psd_mean_peak_value_total_frontside',
        'psd_std_peak_value_total_frontside', 
        'psd_max_peak_value_total_frontside', 
        'psd_min_peak_value_total_frontside',
        'psd_mean_peak_frequency_total_frontside', 
        'psd_std_peak_frequency_total_frontside', 
        'psd_max_peak_frequency_total_frontside', 
        'psd_min_peak_frequency_total_frontside',
        'psd_mean_peak_distance_total_frontside', 
        'psd_std_peak_distance_total_frontside', 
        'psd_max_peak_distance_total_frontside', 
        'psd_min_peak_distance_total_frontside',
        'mean_psd_total_frontside', 
        'std_psd_total_frontside', 
        'max_psd_total_frontside', 
        'min_psd_total_frontside', 
        'skewness_psd_total_frontside', 
        'kurtosis_psd_total_frontside',
        'power_centroid_psd_total_frontside'
    ]
    columns_ae_frontside = [
        'psd_num_peaks_ae_frontside',
        'psd_mean_peak_value_ae_frontside', 
        'psd_std_peak_value_ae_frontside', 
        'psd_max_peak_value_ae_frontside', 
        'psd_min_peak_value_ae_frontside',
        'psd_mean_peak_frequency_ae_frontside', 
        'psd_std_peak_frequency_ae_frontside', 
        'psd_max_peak_frequency_ae_frontside', 
        'psd_min_peak_frequency_ae_frontside',
        'psd_mean_peak_distance_ae_frontside', 
        'psd_std_peak_distance_ae_frontside', 
        'psd_max_peak_distance_ae_frontside', 
        'psd_min_peak_distance_ae_frontside',
        'mean_psd_ae_frontside', 
        'std_psd_ae_frontside', 
        'max_psd_frontside', 
        'min_psd_ae_frontside', 
        'skewness_psd_ae_frontside', 
        'kurtosis_psd_ae_frontside',
        'power_centroid_psd_ae_frontside'
    ]
    columns_x_backside = [
        'psd_num_peaks_x_backside',
        'psd_mean_peak_value_x_backside',
        'psd_std_peak_value_x_backside',
        'psd_max_peak_value_x_backside',
        'psd_min_peak_value_x_backside',
        'psd_mean_peak_frequency_x_backside',
        'psd_std_peak_frequency_x_backside',
        'psd_max_peak_frequency_x_backside',
        'psd_min_peak_frequency_x_backside',
        'psd_mean_peak_distance_x_backside',
        'psd_std_peak_distance_x_backside',
        'psd_max_peak_distance_x_backside',
        'psd_min_peak_distance_x_backside',
        'mean_psd_x_backside',
        'std_psd_x_backside',
        'max_psd_x_backside',
        'min_psd_x_backside',
        'skewness_psd_x_backside',
        'kurtosis_psd_x_backside',
        'power_centroid_psd_x_backside'
    ]
    columns_y_backside = [
        'psd_num_peaks_y_backside',
        'psd_mean_peak_value_y_backside',
        'psd_std_peak_value_y_backside',
        'psd_max_peak_value_y_backside',
        'psd_min_peak_value_y_backside',
        'psd_mean_peak_frequency_y_backside',
        'psd_std_peak_frequency_y_backside',
        'psd_max_peak_frequency_y_backside',
        'psd_min_peak_frequency_y_backside',
        'psd_mean_peak_distance_y_backside',
        'psd_std_peak_distance_y_backside',
        'psd_max_peak_distance_y_backside',
        'psd_min_peak_distance_y_backside',
        'mean_psd_y_backside',
        'std_psd_y_backside',
        'max_psd_y_backside',
        'min_psd_y_backside',
        'skewness_psd_y_backside',
        'kurtosis_psd_y_backside',
        'power_centroid_psd_y_backside'
    ]
    columns_z_backside = [
        'psd_num_peaks_z_backside',
        'psd_mean_peak_value_z_backside',
        'psd_std_peak_value_z_backside',
        'psd_max_peak_value_z_backside',
        'psd_min_peak_value_z_backside',
        'psd_mean_peak_frequency_z_backside',
        'psd_std_peak_frequency_z_backside',
        'psd_max_peak_frequency_z_backside',
        'psd_min_peak_frequency_z_backside',
        'psd_mean_peak_distance_z_backside',
        'psd_std_peak_distance_z_backside',
        'psd_max_peak_distance_z_backside',
        'psd_min_peak_distance_z_backside',
        'mean_psd_z_backside',
        'std_psd_z_backside', 
        'max_psd_z_backside', 
        'min_psd_z_backside', 
        'skewness_psd_z_backside', 
        'kurtosis_psd_z_backside',
        'power_centroid_psd_z_backside'
    ]
    columns_total_backside = [
        'psd_num_peaks_total_backside',
        'psd_mean_peak_value_total_backside',
        'psd_std_peak_value_total_backside', 
        'psd_max_peak_value_total_backside', 
        'psd_min_peak_value_total_backside',
        'psd_mean_peak_frequency_total_backside', 
        'psd_std_peak_frequency_total_backside', 
        'psd_max_peak_frequency_total_backside', 
        'psd_min_peak_frequency_total_backside',
        'psd_mean_peak_distance_total_backside', 
        'psd_std_peak_distance_total_backside', 
        'psd_max_peak_distance_total_backside', 
        'psd_min_peak_distance_total_backside',
        'mean_psd_total_backside', 
        'std_psd_total_backside', 
        'max_psd_total_backside', 
        'min_psd_total_backside', 
        'skewness_psd_total_backside', 
        'kurtosis_psd_total_backside',
        'power_centroid_psd_total_backside'
    ]
    columns_ae_backside = [
        'psd_num_peaks_ae_backside',
        'psd_mean_peak_value_ae_backside', 
        'psd_std_peak_value_ae_backside', 
        'psd_max_peak_value_ae_backside', 
        'psd_min_peak_value_ae_backside',
        'psd_mean_peak_frequency_ae_backside', 
        'psd_std_peak_frequency_ae_backside', 
        'psd_max_peak_frequency_ae_backside', 
        'psd_min_peak_frequency_ae_backside',
        'psd_mean_peak_distance_ae_backside', 
        'psd_std_peak_distance_ae_backside', 
        'psd_max_peak_distance_ae_backside', 
        'psd_min_peak_distance_ae_backside',
        'mean_psd_ae_backside', 
        'std_psd_ae_backside', 
        'max_psd_backside', 
        'min_psd_ae_backside', 
        'skewness_psd_ae_backside', 
        'kurtosis_psd_ae_backside',
        'power_centroid_psd_ae_backside'
    ]

    power_spectral_density_x_frontside = pd.DataFrame(power_spectral_density(
        fft_values_frontside['fft_acc_x'].values,
        frequencies_frontside
    ).reshape(1, -1), columns=columns_x_frontside)
    power_spectral_density_y_frontside = pd.DataFrame(power_spectral_density(
        fft_values_frontside['fft_acc_y'].values,
        frequencies_frontside
    ).reshape(1, -1), columns=columns_y_frontside)
    power_spectral_density_z_frontside = pd.DataFrame(power_spectral_density(
        fft_values_frontside['fft_acc_z'].values,
        frequencies_frontside
    ).reshape(1, -1), columns=columns_z_frontside)
    power_spectral_density_total_frontside = pd.DataFrame(power_spectral_density(
        fft_values_frontside['fft_total_acc'].values,
        frequencies_frontside
    ).reshape(1, -1), columns=columns_total_frontside)
    power_spectral_density_ae_frontside = pd.DataFrame(power_spectral_density(
        fft_values_frontside['fft_ae'].values,
        frequencies_frontside
    ).reshape(1, -1), columns=columns_ae_frontside)


    power_spectral_density_frontside = pd.concat([
        power_spectral_density_x_frontside,
        power_spectral_density_y_frontside,
        power_spectral_density_z_frontside,
        power_spectral_density_total_frontside,
        power_spectral_density_ae_frontside
    ], axis=1)

    power_spectral_density_x_backside = pd.DataFrame(power_spectral_density(
        fft_values_backside['fft_acc_x'].values,
        frequencies_backside
    ).reshape(1, -1), columns=columns_x_backside)
    power_spectral_density_y_backside = pd.DataFrame(power_spectral_density(
        fft_values_backside['fft_acc_y'].values,
        frequencies_backside
    ).reshape(1, -1), columns=columns_y_backside)
    power_spectral_density_z_backside = pd.DataFrame(power_spectral_density(
        fft_values_backside['fft_acc_z'].values,
        frequencies_backside
    ).reshape(1, -1), columns=columns_z_backside)
    power_spectral_density_total_backside = pd.DataFrame(power_spectral_density(
        fft_values_backside['fft_total_acc'].values,
        frequencies_backside
    ).reshape(1, -1), columns=columns_total_backside)
    power_spectral_density_ae_backside = pd.DataFrame(power_spectral_density(
        fft_values_backside['fft_ae'].values,
        frequencies_backside
    ).reshape(1, -1), columns=columns_ae_backside)


    power_spectral_density_backside = pd.concat([
        power_spectral_density_x_backside,
        power_spectral_density_y_backside,
        power_spectral_density_z_backside,
        power_spectral_density_total_backside,
        power_spectral_density_ae_backside
    ], axis=1)

    feature_power_spectral_density = pd.concat([
        power_spectral_density_frontside,
        power_spectral_density_backside
    ], axis=1)

    return feature_power_spectral_density


def peak_frequency(fft_values, frequencies):
    magnitude_spectrum = np.abs(fft_values)
    max_value_index = np.argmax(magnitude_spectrum)
    peak_frequency = frequencies[max_value_index]
    return peak_frequency


def calculate_peak_frequency(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    frequencies_frontside = fft_sensor_data_frontside['frequency'].values
    frequencies_backside = fft_sensor_data_backside['frequency'].values

    columns_frontside = [
        'peak_frequency_x_frontside',
        'peak_frequency_y_frontside',
        'peak_frequency_z_frontside',
        'peak_frequency_total_frontside',
        'peak_frequency_ae_frontside'
    ]
    columns_backside = [
        'peak_frequency_x_backside',
        'peak_frequency_y_backside',
        'peak_frequency_z_backside',
        'peak_frequency_total_backside',
        'peak_frequency_ae_backside'
    ]

    peak_frequency_x_frontside = peak_frequency(
        fft_values_frontside['fft_acc_x'].values,
        frequencies_frontside
    )
    peak_frequency_y_frontside = peak_frequency(
        fft_values_frontside['fft_acc_y'].values,
        frequencies_frontside
    )
    peak_frequency_z_frontside = peak_frequency(
        fft_values_frontside['fft_acc_z'].values,
        frequencies_frontside
    )
    peak_frequency_total_frontside = peak_frequency(
        fft_values_frontside['fft_total_acc'].values,
        frequencies_frontside
    )
    peak_frequency_ae_frontside = peak_frequency(
        fft_values_frontside['fft_ae'].values,
        frequencies_frontside
    )


    peak_frequency_frontside = pd.DataFrame([[
        peak_frequency_x_frontside,
        peak_frequency_y_frontside,
        peak_frequency_z_frontside,
        peak_frequency_total_frontside,
        peak_frequency_ae_frontside
    ]], columns=columns_frontside)


    peak_frequency_x_backside = peak_frequency(
        fft_values_backside['fft_acc_x'].values,
        frequencies_backside
    )
    peak_frequency_y_backside = peak_frequency(
        fft_values_backside['fft_acc_y'].values,
        frequencies_backside
    )
    peak_frequency_z_backside = peak_frequency(
        fft_values_backside['fft_acc_z'].values,
        frequencies_backside
    )
    peak_frequency_total_backside = peak_frequency(
        fft_values_backside['fft_total_acc'].values,
        frequencies_backside
    )
    peak_frequency_ae_backside = peak_frequency(
        fft_values_backside['fft_ae'].values,
        frequencies_backside
    )


    peak_frequency_backside = pd.DataFrame([[
        peak_frequency_x_backside,
        peak_frequency_y_backside,
        peak_frequency_z_backside,
        peak_frequency_total_backside,
        peak_frequency_ae_backside
    ]], columns=columns_backside)

    feature_peak_frequency = pd.concat([
        peak_frequency_frontside,
        peak_frequency_backside
    ], axis=1)

    return feature_peak_frequency


def sepctral_statistics_features(fft_values):
    magnitude_spectrum = np.abs(fft_values)
    magnitude_mean = np.mean(magnitude_spectrum)
    magnitude_std = np.std(magnitude_spectrum)
    magnitude_sort_index = np.argsort(magnitude_spectrum)
    magnitude_1_max = magnitude_spectrum[magnitude_sort_index][-1]
    magnitude_2_max = magnitude_spectrum[magnitude_sort_index][-2]
    magnitude_3_max = magnitude_spectrum[magnitude_sort_index][-3]
    magnitude_min = magnitude_spectrum[magnitude_sort_index][0]

    features = np.array([
        magnitude_mean, magnitude_std,
        magnitude_1_max, magnitude_2_max, magnitude_3_max,
        magnitude_min
    ])

    return features


def calculate_spectral_statistics_features(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]


    columns_x_frontside = [
        'magnitude_mean_x_frontside',
        'magnitude_std_x_frontside',
        'magnitude_1_max_x_frontside',
        'magnitude_2_max_x_frontside',
        'magnitude_3_max_x_frontside',
        'magnitude_min_x_frontside'
    ]
    columns_y_frontside = [
        'magnitude_mean_y_frontside', 
        'magnitude_std_y_frontside',
        'magnitude_1_max_y_frontside', 
        'magnitude_2_max_y_frontside', 
        'magnitude_3_max_y_frontside',
        'magnitude_min_y_frontside'
    ]
    columns_z_frontside = [
        'magnitude_mean_z_frontside', 
        'magnitude_std_z_frontside',
        'magnitude_1_max_z_frontside', 
        'magnitude_2_max_z_frontside', 
        'magnitude_3_max_z_frontside',
        'magnitude_min_z_frontside'
    ]
    columns_total_frontside = [
        'magnitude_mean_total_frontside', 
        'magnitude_std_total_frontside',
        'magnitude_1_max_total_frontside', 
        'magnitude_2_max_total_frontside', 
        'magnitude_3_max_total_frontside',
        'magnitude_min_total_frontside'
    ]
    columns_ae_frontside = [
        'magnitude_mean_ae_frontside', 
        'magnitude_std_ae_frontside',
        'magnitude_1_max_ae_frontside', 
        'magnitude_2_max_ae_frontside', 
        'magnitude_3_max_ae_frontside',
        'magnitude_min_ae_frontside'
    ]
    columns_x_backside = [
        'magnitude_mean_x_backside',
        'magnitude_std_x_backside',
        'magnitude_1_max_x_backside',
        'magnitude_2_max_x_backside',
        'magnitude_3_max_x_backside',
        'magnitude_min_x_backside'
    ]
    columns_y_backside = [
        'magnitude_mean_y_backside', 
        'magnitude_std_y_backside',
        'magnitude_1_max_y_backside', 
        'magnitude_2_max_y_backside', 
        'magnitude_3_max_y_backside',
        'magnitude_min_y_backside'
    ]
    columns_z_backside = [
        'magnitude_mean_z_backside', 
        'magnitude_std_z_backside',
        'magnitude_1_max_z_backside', 
        'magnitude_2_max_z_backside', 
        'magnitude_3_max_z_backside',
        'magnitude_min_z_backside'
    ]
    columns_total_backside = [
        'magnitude_mean_total_backside', 
        'magnitude_std_total_backside',
        'magnitude_1_max_total_backside', 
        'magnitude_2_max_total_backside', 
        'magnitude_3_max_total_backside',
        'magnitude_min_total_backside'
    ]
    columns_ae_backside = [
        'magnitude_mean_ae_backside', 
        'magnitude_std_ae_backside',
        'magnitude_1_max_ae_backside', 
        'magnitude_2_max_ae_backside', 
        'magnitude_3_max_ae_backside',
        'magnitude_min_ae_backside'
    ]

    sepctral_statistics_features_x_frontside = pd.DataFrame(sepctral_statistics_features(
        fft_values_frontside['fft_acc_x'].values,
    ).reshape(1, -1), columns=columns_x_frontside)
    sepctral_statistics_features_y_frontside = pd.DataFrame(sepctral_statistics_features(
        fft_values_frontside['fft_acc_y'].values,
    ).reshape(1, -1), columns=columns_y_frontside)
    sepctral_statistics_features_z_frontside = pd.DataFrame(sepctral_statistics_features(
        fft_values_frontside['fft_acc_z'].values,
    ).reshape(1, -1), columns=columns_z_frontside)
    sepctral_statistics_features_total_frontside = pd.DataFrame(sepctral_statistics_features(
        fft_values_frontside['fft_total_acc'].values,
    ).reshape(1, -1), columns=columns_total_frontside)
    sepctral_statistics_features_ae_frontside = pd.DataFrame(sepctral_statistics_features(
        fft_values_frontside['fft_ae'].values,
    ).reshape(1, -1), columns=columns_ae_frontside)


    sepctral_statistics_features_frontside = pd.concat([
        sepctral_statistics_features_x_frontside,
        sepctral_statistics_features_y_frontside,
        sepctral_statistics_features_z_frontside,
        sepctral_statistics_features_total_frontside,
        sepctral_statistics_features_ae_frontside
    ], axis=1)

    sepctral_statistics_features_x_backside = pd.DataFrame(sepctral_statistics_features(
        fft_values_backside['fft_acc_x'].values,
    ).reshape(1, -1), columns=columns_x_backside)
    sepctral_statistics_features_y_backside = pd.DataFrame(sepctral_statistics_features(
        fft_values_backside['fft_acc_y'].values,
    ).reshape(1, -1), columns=columns_y_backside)
    sepctral_statistics_features_z_backside = pd.DataFrame(sepctral_statistics_features(
        fft_values_backside['fft_acc_z'].values,
    ).reshape(1, -1), columns=columns_z_backside)
    sepctral_statistics_features_total_backside = pd.DataFrame(sepctral_statistics_features(
        fft_values_backside['fft_total_acc'].values,
    ).reshape(1, -1), columns=columns_total_backside)
    sepctral_statistics_features_ae_backside = pd.DataFrame(sepctral_statistics_features(
        fft_values_backside['fft_ae'].values,
    ).reshape(1, -1), columns=columns_ae_backside)


    sepctral_statistics_features_backside = pd.concat([
        sepctral_statistics_features_x_backside,
        sepctral_statistics_features_y_backside,
        sepctral_statistics_features_z_backside,
        sepctral_statistics_features_total_backside,
        sepctral_statistics_features_ae_backside
    ], axis=1)

    feature_sepctral_statistics_features = pd.concat([
        sepctral_statistics_features_frontside,
        sepctral_statistics_features_backside
    ], axis=1)

    return feature_sepctral_statistics_features


def spectral_spread(fft_values, frequencies):
    magnitude_spectrum = np.abs(fft_values)
    power_spectrum = magnitude_spectrum ** 2
    
    spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
    spectral_spread = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * power_spectrum) / np.sum(power_spectrum))
    return spectral_spread



def calculate_spectral_spread(fft_sensor_data_frontside, fft_sensor_data_backside):
    fft_values_frontside = fft_sensor_data_frontside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    fft_values_backside = fft_sensor_data_backside[[
        'fft_acc_x',
        'fft_acc_y',
        'fft_acc_z',
        'fft_total_acc',
        'fft_ae'
    ]]
    frequencies_frontside = fft_sensor_data_frontside['frequency'].values
    frequencies_backside = fft_sensor_data_backside['frequency'].values

    columns_frontside = [
        'spectral_spread_x_frontside',
        'spectral_spread_y_frontside',
        'spectral_spread_z_frontside',
        'spectral_spread_total_frontside',
        'spectral_spread_ae_frontside'
    ]
    columns_backside = [
        'spectral_spread_x_backside',
        'spectral_spread_y_backside',
        'spectral_spread_z_backside',
        'spectral_spread_total_backside',
        'spectral_spread_ae_backside'
    ]

    spectral_spread_x_frontside = spectral_spread(
        fft_values_frontside['fft_acc_x'].values,
        frequencies_frontside
    )
    spectral_spread_y_frontside = spectral_spread(
        fft_values_frontside['fft_acc_y'].values,
        frequencies_frontside
    )
    spectral_spread_z_frontside = spectral_spread(
        fft_values_frontside['fft_acc_z'].values,
        frequencies_frontside
    )
    spectral_spread_total_frontside = spectral_spread(
        fft_values_frontside['fft_total_acc'].values,
        frequencies_frontside
    )
    spectral_spread_ae_frontside = spectral_spread(
        fft_values_frontside['fft_ae'].values,
        frequencies_frontside
    )


    spectral_spread_frontside = pd.DataFrame([[
        spectral_spread_x_frontside,
        spectral_spread_y_frontside,
        spectral_spread_z_frontside,
        spectral_spread_total_frontside,
        spectral_spread_ae_frontside
    ]], columns=columns_frontside)


    spectral_spread_x_backside = spectral_spread(
        fft_values_backside['fft_acc_x'].values,
        frequencies_backside
    )
    spectral_spread_y_backside = spectral_spread(
        fft_values_backside['fft_acc_y'].values,
        frequencies_backside
    )
    spectral_spread_z_backside = spectral_spread(
        fft_values_backside['fft_acc_z'].values,
        frequencies_backside
    )
    spectral_spread_total_backside = spectral_spread(
        fft_values_backside['fft_total_acc'].values,
        frequencies_backside
    )
    spectral_spread_ae_backside = spectral_spread(
        fft_values_backside['fft_ae'].values,
        frequencies_backside
    )


    spectral_spread_backside = pd.DataFrame([[
        spectral_spread_x_backside,
        spectral_spread_y_backside,
        spectral_spread_z_backside,
        spectral_spread_total_backside,
        spectral_spread_ae_backside
    ]], columns=columns_backside)

    feature_spectral_spread = pd.concat([
        spectral_spread_frontside,
        spectral_spread_backside
    ], axis=1)

    return feature_spectral_spread


def calculate_frequency_domain_features(
        sensor_data_frontside,
        sensor_data_backside,
        threshold_spectral_roll_off_point=0.85,
        threshold_spectral_peaks=0.95
):
    
    fft_sensor_data_frontside = fft_calculation(sensor_data_frontside)
    fft_sensor_data_backside = fft_calculation(sensor_data_backside)


    feature_fundamental_frequency = calculate_fundamental_frequency(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )
    
    feature_spectral_energy = calculate_spectral_energy(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )
    feature_spectral_entropy = calculate_spectral_entropy(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )
    feature_spectral_centroid = calculate_spectral_centroid(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )

    feature_spectral_flatness = calculate_spectral_flatness(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )
    feature_spectral_roll_off_point = calculate_spectral_roll_off_point(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside, 
        threshold_spectral_roll_off_point
    )
    feature_spectral_skewness = calculate_spectral_skewness(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )
    feature_spectral_kurtosis = calculate_spectral_kurtosis(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )
    feature_spectral_peaks = calculate_spectral_peaks(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside, 
        threshold_spectral_peaks
    )
    feature_weighted_mean_frequency = calculate_weighted_mean_frequency(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )
    feature_spectral_coefficient_of_variation = calculate_spectral_coefficient_of_variation(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )
    feature_power_spectral_density = calculate_power_spectral_density(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )
    feature_peak_frequency = calculate_peak_frequency(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )
    feature_spectral_statistics_features = calculate_spectral_statistics_features(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )
    feature_spectral_spread = calculate_spectral_spread(
        fft_sensor_data_frontside, 
        fft_sensor_data_backside
    )

    features_frequency_domain = pd.concat([
        feature_fundamental_frequency,
        feature_spectral_energy,
        feature_spectral_entropy,
        feature_spectral_centroid,
        feature_spectral_flatness,
        feature_spectral_roll_off_point,
        feature_spectral_skewness,
        feature_spectral_kurtosis,
        feature_spectral_peaks,
        feature_weighted_mean_frequency,
        feature_spectral_coefficient_of_variation,
        feature_power_spectral_density,
        feature_peak_frequency,
        feature_spectral_statistics_features,
        feature_spectral_spread
    ], axis=1)

    return features_frequency_domain


def calculate_features(sensor_data_frontside, sensor_data_backside, part_id):
    features_time_domain = feature_extraction_time_domain(
        sensor_data_frontside,
        sensor_data_backside
    )
    features_frequency_domain = calculate_frequency_domain_features(
        sensor_data_frontside,
        sensor_data_backside
    )

    features = pd.concat([
        features_time_domain.reset_index(drop=True),
        features_frequency_domain.reset_index(drop=True)
    ], axis=1)
    features['part_id'] = part_id

    return features[feature_names]

