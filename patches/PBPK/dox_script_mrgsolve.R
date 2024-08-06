# Ensure the mrgsolve package is installed
if (!requireNamespace("mrgsolve", quietly = TRUE)) {
    install.packages("mrgsolve")
}

# Load the mrgsolve package
library(mrgsolve)

# Clear the environment
rm(list=ls(all=TRUE))

# Function to run the doxorubicin model with passed parameters
run_doxorubicin_model <- function(dose_mg, age, weight, height) {
  # Read in the model
  m1 <- mread("models/PBPK/dox_mod_2ic")
  
  # Dose regimen calculation
  MW <- 543.52 # g/mol doxorubicin
  BSA <- weight ^ 0.425 * height ^ 0.725 * 0.007184 # [m2] Body surface area calculation
  inf_dose_mg <- dose_mg * BSA # [mg * m^2]
  inf_dose_umol <- (inf_dose_mg * 0.001 / MW ) * 1000000 # conversion to umol
  Vve <- m1$Vve
  inf_dose_uM <- inf_dose_umol / Vve # Dosing to Vve, C = A / V
  regimen <- ev(amt = inf_dose_uM, cmt="Cve", tinf=0.05)
  
  # Simulate the model, all output in dataframe
  out <- mrgsim_df(m1, events=regimen)
  
  # Return the result
  return(out)
}

# Example usage (remove this from the script if used as a module)
# dose_mg <- as.numeric(Sys.getenv("dose_mg"))
# age <- as.numeric(Sys.getenv("age"))
# weight <- as.numeric(Sys.getenv("weight"))
# height <- as.numeric(Sys.getenv("height"))
# doxorubicin_model_output <- run_doxorubicin_model(dose_mg, age, weight, height)
