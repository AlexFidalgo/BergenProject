# Load necessary libraries
library(MASS) # for stepwise regression
library(car)  # for VIF, optional for diagnostics
library(readxl)
library(stringr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(partykit)
library(caret)

library(ggplot2)
library(SmartEDA)
library(psych)


# install.packages('psych')


Scale_Transform = function(dfbase) {

  
  continuous_vars <- names(data_cleaned)[sapply(data_cleaned, is.numeric)]  # Only numeric columns
  
  # Create a new dataframe and keep track of variable name changes
  data_transformed <- as.data.frame(data_cleaned)
  new_names <- names(data_transformed)
  
  for (var in continuous_vars) {
    result <- log_transform(data_cleaned[[var]], var)
    data_transformed[[var]] <- result$data
    new_names[which(names(data_transformed) == var)] <- result$name
  }
  
  # Update the column names after transformation
  names(data_transformed) <- new_names
  
  Sumarios = describe(dfbase, skew = TRUE, trim=.01,
                      type=3,check=TRUE,fast=NULL,quant=NULL,IQR=FALSE,omit=FALSE)
  VarSummary = data.frame(Var=rownames(Sumarios), min=Sumarios$min, 
                          skew=Sumarios$skew, kurtosis=Sumarios$kurtosis)
  
  NumericVar = VarSummary$Var[which(!str_detect(VarSummary$Var, '\\*'))]
  
  
  #
  # Box-Cox transformation
  #
  dfbasetrans = dfbase
  idxselec = Reduce(intersect,
                    list(which(!str_detect(VarSummary$Var, '\\*')),
                         which(VarSummary$min>0)))
  
  df_lambdas = data.frame()
  
  for (idvar in 1:length(idxselec)) {
    currvar = VarSummary$Var[idxselec[idvar]]
    
    boxcoxres = BoxCoxTrans(dfbasetrans[,currvar], lambda=seq(-20,20,by=0.1), fudge=0.1, na.rm=T)
    
    dfbasetrans[,currvar] = predict(boxcoxres, dfbasetrans[,currvar])
    
    df_lambdas = rbind(df_lambdas, 
                       data.frame(Var=currvar, lambda=boxcoxres$lambda,
                                  p.value=agostino.test(dfbasetrans[,currvar])$p.value))
  }
  
  return(list(boxcoxres=boxcoxres, df_lambdas=df_lambdas, 
              dfbasetrans=dfbasetrans))
  
}



# Load your dataset
data <- read_xlsx("../20241103b_TratIndic_valores.xlsx", sheet='Valores_nitrito_CAbrev')
data$Data = NULL
data$Classe = NULL
data$Anonimiz = NULL
#data$Anonimiz = factor(data$Anonimiz, levels = sort(unique(data$Anonimiz)))


column.names = colnames(data)
new.column.names = paste('v', str_replace(column.names, '\\_', ''), sep='')
colnames(data) = new.column.names

# Step a: Remove predictors with more than 50% missing values (mvp=0.5)
mvp <- 0.6
missing_prop <- colMeans(is.na(data))
data_cleaned <- data[, missing_prop <= mvp]


# saida = Scale_Transform(data_cleaned)


# Step b: Shapiro-Wilk test and log transformation
log_transform <- function(x, var_name) {
  sumconst =  0  #  Não precisamos adicionar nenhum valor, pois não há zeros na base
  
  p_original <- shapiro.test(x[!is.na(x)])$p.value
  p_log <- shapiro.test(log10(x[!is.na(x)] + sumconst))$p.value # log10 + 1 to avoid log(0)
  
  if (p_log > p_original) {
    new_var_name <- paste0(var_name, "lg")  # Add "lg" suffix if log is chosen
    return(list(data = log10(x + sumconst), name = new_var_name))
  } else {
    return(list(data = x, name = var_name))
  }
}

# Apply log transformation to continuous predictors
continuous_vars <- names(data_cleaned)[sapply(data_cleaned, is.numeric)]  # Only numeric columns

# Create a new dataframe and keep track of variable name changes
data_transformed <- as.data.frame(data_cleaned)
new_names <- names(data_transformed)

for (var in continuous_vars) {
  result <- log_transform(data_cleaned[[var]], var)
  data_transformed[[var]] <- result$data
  new_names[which(names(data_transformed) == var)] <- result$name
}

# Update the column names after transformation
names(data_transformed) <- new_names

# Separate response value and predictors
response_var <- intersect(names(data_transformed), c("v09Nitrito", "v09Nitritolg"))  # Response variable
predictors <- setdiff(names(data_transformed), response_var)  # All other variables as predictors


# Aplicação do CART
formulamodel = as.formula(paste(response_var, "~", paste(predictors, collapse = " + ")))
dtmodel = rpart(formulamodel, data=data_transformed, cp=0.01, minsplit=10)
rpart.plot(dtmodel)

dtpred = as.vector(predict(dtmodel, data_transformed))

plot(as.vector(data_transformed[, response_var]), dtpred)
abline(a=0, b=1, col='red')

summ_dtmodel = summary(dtmodel)


sel_predictors = predictors
formulamodel = as.formula(paste(response_var, "~", paste(sel_predictors, collapse = " + ")))
sel_data_transformed = data_transformed[, c(sel_predictors, response_var)]
sel_data_completed = rfImpute(formulamodel, sel_data_transformed, iter=20, ntree=500)

while(T) {

  rfmodel = randomForest(formulamodel, data=sel_data_completed, 
                         cp=0.01, minsplit=10, ntree=1000, importance=T)
  
  rfpred = predict(rfmodel, sel_data_transformed)
  
  plot(sel_data_transformed[, response_var], rfpred)
  abline(a=0, b=1, col='red')

  # Importâncias das variáveis
  var.importance = sort(rfmodel$importance[,"%IncMSE"], decreasing = F)
  # Create data
  dfimport <- data.frame(
    Variable=factor(names(var.importance), levels=names(var.importance)),  
    Importance=var.importance
  )
  
  # Barplot
  p1 = ggplot(dfimport, aes(x=Variable, y=Importance)) + 
    geom_bar(stat = "identity") +
    coord_flip()
  print(p1)
  
  importance_thr = 0.015
  if (min(var.importance)<importance_thr) {
    idxsel = which(var.importance>=importance_thr)
    sel_predictors = names(var.importance)[idxsel]
    formulamodel = as.formula(paste(response_var, "~", paste(sel_predictors, collapse = " + ")))
    sel_data_transformed = data_transformed[, c(sel_predictors, response_var)]
    sel_data_completed = rfImpute(formulamodel, sel_data_transformed, iter=20, ntree=500)
    
  } else {
    break
  }
    
}

browser()

ExpNumViz(sel_data_completed, nlim=3, Page=c(3,2))

#
# Step c: Run a GLM with pairwise interactions
#

# Create formula with interactions
formula <- as.formula(paste(response_var, "~ (", paste(sel_predictors, collapse = " + "), ")^2"))

# Fit GLM model
glm_model <- glm(formula, data = sel_data_completed, family = gaussian())  # or other appropriate family

# Step d: Perform stepwise regression (bidirectional)
stepwise_model <- stepAIC(glm_model, direction = "both")

# Summary of the best model
summary(stepwise_model)


#
# Step c: Forward stepwise regression
#

# # Start with a null model (intercept only)
null_model <- glm(as.formula(paste(response_var, "~ 1")), data = sel_data_completed, family = gaussian())

# Full model (with all predictors and interactions)
full_model <- glm(as.formula(paste(response_var, "~ (", paste(sel_predictors, collapse = " + "), ")^2")),
                  data = sel_data_completed, family = gaussian())

# Perform forward stepwise regression
forward_stepwise_model <- step(null_model,
                               scope = list(lower = null_model, upper = full_model),
                               direction = "both")

# Summary of the best forward stepwise model
summary(forward_stepwise_model)

pred_compl = predict(forward_stepwise_model, sel_data_completed)

plot(sel_data_completed[, response_var], pred_compl)
abline(a=0, b=1, col='red')

pred_nomissing = predict(forward_stepwise_model, sel_data_transformed)
plot(sel_data_transformed[, response_var], pred_nomissing)
abline(a=0, b=1, col='red')

selected_terms = names(forward_stepwise_model$coefficients[-1])
formulamodel = as.formula(paste(response_var, "~", paste(selected_terms, collapse = " + ")))


model_nomiss = glm(formulamodel, data = sel_data_transformed, family = gaussian())
vif(model_nomiss)

stepwise_model_nomiss <- stepAIC(model_nomiss, direction = "both")

pred_nomissing = predict(model_nomiss, sel_data_transformed)
plot(sel_data_transformed[, response_var], pred_nomissing)
abline(a=0, b=1, col='red')



# # Step 1: Forward stepwise regression to find best main effects (individual predictors)
# 
# # Start with a null model (intercept only)
# null_model <- glm(as.formula(paste(response_var, "~ 1")), data = data_transformed, family = gaussian())
# 
# # Full model (only main effects, no interactions)
# full_model <- glm(as.formula(paste(response_var, "~", paste(predictors, collapse = " + "))),
#                   data = data_transformed, family = gaussian())
# 
# # Forward stepwise to select best individual predictors
# forward_stepwise_model <- stepAIC(null_model, 
#                                scope = list(lower = null_model, upper = full_model), 
#                                direction = "forward", trace=2)
# 
# # Extract the selected predictors from the forward stepwise model
# selected_predictors <- names(coef(forward_stepwise_model))[-1]  # Exclude intercept
# 
# # Step 2: Bidirectional stepwise with interactions
# # Create a formula with selected predictors and their pairwise interactions
# interaction_formula <- as.formula(paste(response_var, "~ (", paste(selected_predictors, collapse = " + "), ")^2"))
# 
# # Fit the full model with main effects and pairwise interactions
# interaction_model <- glm(interaction_formula, data = data_transformed, family = gaussian())
# 
# # Perform bidirectional stepwise regression to refine the model
# bidirectional_stepwise_model <- step(interaction_model, direction = "both")
