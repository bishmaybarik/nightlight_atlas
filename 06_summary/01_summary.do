/*******************************************************************************
File: 01_summary.do
Purpose: Making a summary stats table for the project
Author: Bishmay Barik
*******************************************************************************/

clear all
macro drop all

global dir "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas" // Calling the global macro to save the directory

* Loading the dataset

import delimited "$dir/01_data/02_processed/secc_combined_updated.csv", clear

// No. of rural shrid2 - 563,741
// No. of urban shrid2 - 3,688

* shrid2 is a string variable. Generating a new variable to count the numbers

* Generate summary statistics for rural and urban separately

eststo clear

* DMSP Total Light
eststo: estpost summarize dmsp_total_light 
eststo: estpost summarize dmsp_total_light if area_type == "rural", detail
eststo: estpost summarize dmsp_total_light if area_type == "urban", detail

* SECC Consumption
eststo: estpost summarize secc_cons 
eststo: estpost summarize secc_cons if area_type == "rural", detail
eststo: estpost summarize secc_cons if area_type == "urban", detail

* SECC Consumption PC
eststo: estpost summarize secc_cons_pc 
eststo: estpost summarize secc_cons_pc if area_type == "rural", detail
eststo: estpost summarize secc_cons_pc if area_type == "urban", detail

* Combine the results into a table
* Store Summary Statistics
eststo clear

* DMSP Total Light
eststo: estpost summarize dmsp_total_light
eststo: estpost summarize dmsp_total_light if area_type == "rural", detail
eststo: estpost summarize dmsp_total_light if area_type == "urban", detail

* SECC Consumption
eststo: estpost summarize secc_cons
eststo: estpost summarize secc_cons if area_type == "rural", detail
eststo: estpost summarize secc_cons if area_type == "urban", detail

* SECC Consumption Per Capita
eststo: estpost summarize secc_cons_pc
eststo: estpost summarize secc_cons_pc if area_type == "rural", detail
eststo: estpost summarize secc_cons_pc if area_type == "urban", detail

* Export as LaTeX Table
esttab using "$dir/05_reports/tables/summary_new.tex", ///
    cells("mean sd min max p50") ///
    replace ///
    label ///
    nonumber ///
    nomtitles ///
    booktabs ///
    compress ///
    alignment(D{.}{.}{-1})


* Display a message indicating the file has been saved
di "LaTeX table saved to $dir/05_reports/tables/summary_stats.tex"
