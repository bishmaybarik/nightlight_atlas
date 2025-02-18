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
esttab using "$dir/05_reports/tables/summary_stats.tex", ///
    cells("N Mean(fmt(2)) SD(fmt(2)) Min(fmt(0)) Max(fmt(0))") ///
    collabels("N" "Mean" "SD" "Min" "Max") ///
    label booktabs ///
    replace ///
    title("Summary Statistics by Area Type") ///
    addnotes("Source: Your data source here") ///
    nonumbers ///
    noobs ///
    prehead("\begin{table}[h]" ///
            "\centering" ///
            "\renewcommand{\arraystretch}{1.2}" ///
            "\setlength{\tabcolsep}{8pt}" ///
            "\begin{tabular}{lrrrrr}" ///
            "\toprule\toprule" ///
            "\textbf{Variable} & \textbf{N} & \textbf{Mean} & \textbf{SD} & \textbf{Min} & \textbf{Max} \\" ///
            "\midrule") ///
    postfoot("\toprule\toprule" ///
             "\end{tabular}" ///
             "\caption{Summary Statistics by Area Type}" ///
             "\label{tab:summary_stats}" ///
             "\end{table}")

* Display a message indicating the file has been saved
di "LaTeX table saved to $dir/05_reports/tables/summary_stats.tex"
