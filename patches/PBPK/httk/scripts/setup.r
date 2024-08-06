# Ensure the httk package is installed
if (!requireNamespace("httk", quietly = TRUE)) {
    install.packages("./download/PBK ALTERNATIVE/httk/modified_package_tar/httkfb2.tar", repos = NULL, type = "source") # Adjust file path location
}

# Load the httk package
library(httk)

# Clear the environment
rm(list=ls(all=TRUE))

# Function to run the HTTK model with passed parameters
run_httk_model <- function(chem_name, species, daily_dose, doses_per_day, days) {
  BPA <- solve_pbtk(chem.name = chem_name,
                    species = species,
                    daily.dose = daily_dose,
                    doses.per.day = doses_per_day,
                    days = days,
                    plots = FALSE)  # Set plots to FALSE
                    
  BPA_df <- as.data.frame(BPA)
  
  return(BPA_df)
}

# Example usage (remove this from the script if used as a module)
# results <- run_httk_model("Bisphenol A", "human", 1, 1, 15)
