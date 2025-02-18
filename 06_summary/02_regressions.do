/*******************************************************************************
File: 02_regressions.do
Purpose: Making a summary stats table for the project
Author: Bishmay Barik
*******************************************************************************/

clear all
macro drop all

global dir "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas" // Calling the global macro to save the directory

* Loading the dataset

import delimited "$dir/01_data/02_processed/secc_combined_updated.csv", clear

* Creating a regression table to show that as nightlight concentration increases, consumption increases too.

gen urban_dummy = (area_type == "urban")

regress secc_cons dmsp_total_light urban_dummy c.dmsp_total_light#c.urban_dummy

est store pooled_model

esttab pooled_model using "$dir/05_reports/tables/regression_coefficients_pool.tex", ///
replace label title("Effect of Nightlight Intensity on SECC Consumption") ///
mtitles("Pooled Model with Interaction") ///
varlabels(dmsp_total_light "Nightlight Intensity" ///
         urban_dummy "Urban Dummy" ///
         c.dmsp_total_light#c.urban_dummy "Interaction Term") ///
stats(N r2, labels("Observations" "R-squared")) ///
se star(* 0.1 ** 0.05 *** 0.01) ///
booktabs nonotes addnotes("Dependent Variable: SECC Consumption")

/*

regress secc_cons dmsp_total_light if area_type == "rural"
regress secc_cons dmsp_total_light if area_type == "urban"

* Run regression for rural areas
regress secc_cons dmsp_total_light if area_type == "rural"
est store rural_model  

* Run regression for urban areas
regress secc_cons dmsp_total_light if area_type == "urban"
est store urban_model  

* Export regression table to LaTeX
esttab rural_model urban_model using ///
"$dir/05_reports/tables/regression_coefficients.tex", ///
replace label title("Effect of Nightlight Intensity on SECC Consumption") ///
mtitles("Rural Areas" "Urban Areas") ///
varlabels(dmsp_total_light "Nightlight Intensity") ///
stats(N r2, labels("Observations" "R-squared")) ///
se star(* 0.1 ** 0.05 *** 0.01) ///
booktabs nonotes addnotes("Dependent Variable: SECC Consumption")

*/
