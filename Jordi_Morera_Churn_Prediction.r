###########################################################################
#                                                                         #  
#             Computing Lab Project: Churn prediction project             #
#                                                                         #
###########################################################################

#-------------------------------------------------------------------------#
#
# December 18
# Jordi 
#
#-------------------------------------------------------------------------#

# Load libraries.
library("caret")
library("randomForest")
library("pROC")
library("doMC")

# Load source files.
source("./Scripts/io.R")
source("./Scripts/classDistribution.R")


# Set seed.
set.seed(321)

# Set path to data set.
path <- "./Data/churn.csv"

# Set column names
column.names <- c("Cookie", "Time_Stamp",
                  "Active_d1","Active_d2","Active_d3","Active_d4","Active_d5", "Active_d6", "Active_d7",
                  "Dwell_d1", "Dwell_d2","Dwell_d3","Dwell_d4","Dwell_d5","Dwell_d6","Dwell_d7",
                  "Sessions_d1","Sessions_d2","Sessions_d3","Sessions_d4","Sessions_d5","Sessions_d6","Sessions_d7",
                  "Views_d1","Views_d2","Views_d3","Views_d4","Views_d5","Views_d6","Views_d7",
                  "Clicks_d1", "Clicks_d2", "Clicks_d3","Clicks_d4","Clicks_d5","Clicks_d6","Clicks_d7",
                  "Cluster")

# Count number of lines in input file.
lines <- readChar(path, file.info(path)$size)
total.rows <- length(gregexpr("\n",lines)[[1L]])
rm(lines)

# Load data.
df <- load.data(p.path = path,
                p.header = TRUE,
                p.dec = ".",
                p.sep = ",",
                p.blank.lines.skip = TRUE,
                p.stringsAsFactors = FALSE,
                p.comment.char = "",
                p.initial.rows = 100,
                p.total.nrows = total.rows,
                p.column.names = column.names,
                p.id = FALSE)

# Start stop-watch
start.time <- as.numeric(as.POSIXct(Sys.time()))

# Remove missing cases.
df <- df[complete.cases(df),]

# Tranform the class values to factors.
df$Cluster <- as.factor(df$Cluster)

# Feature engineering.

# Get mean and standard deviation for each user.
# I tried to engineer the features in order to check if the user is a heavy/light user via mean
# and i he/she is a consistent/sporadic user via the standard deviation. The reason behind is that
# a heavy user with consistent use will be less likely to churn.

# -----------------------------------------------------------------------------------------------------#
# I also tried to capture the trend of the user over the 7 observed days fitting a line to value vs day and
# returning the line slope. However the results were worse than the ones obtained with the finally
# selected features. See an example below

#lin_reg <- function(d1,d2,d3,d4,d5,d6,d7){
#  x <- c(1:7)
#  y <- c(d1,d2,d3,d4,d5,d6,d7)
#  model <- lm(y~x)
#  
#  return(summary(model)$coefficients[2])
#}
#
#df$dwell_trend <- pmap_dbl(select(df,Dwell_d1:Dwell_d7),~lin_reg(..1,..2,..3,..4,..5,..6,..7))
# -----------------------------------------------------------------------------------------------------#


df$active_sd <- pmap_dbl(select(df,Active_d1:Active_d7),~sd(c(..1,..2,..3,..4,..5,..6,..7)))
df$active_m <- pmap_dbl(select(df,Active_d1:Active_d7),~mean(c(..1,..2,..3,..4,..5,..6,..7)))

df$dwell_m <- pmap_dbl(select(df,Dwell_d1:Dwell_d7),~mean(c(..1,..2,..3,..4,..5,..6,..7)))
df$dwell_sd <- pmap_dbl(select(df,Dwell_d1:Dwell_d7),~sd(c(..1,..2,..3,..4,..5,..6,..7)))

df$sessions_m <- pmap_dbl(select(df,Sessions_d1:Sessions_d7),~mean(c(..1,..2,..3,..4,..5,..6,..7)))
df$sessions_sd <- pmap_dbl(select(df,Sessions_d1:Sessions_d7),~sd(c(..1,..2,..3,..4,..5,..6,..7)))

df$views_m <- pmap_dbl(select(df,Views_d1:Views_d7),~mean(c(..1,..2,..3,..4,..5,..6,..7)))
df$views_sd <- pmap_dbl(select(df,Views_d1:Views_d7),~sd(c(..1,..2,..3,..4,..5,..6,..7)))

df$clicks_m <- pmap_dbl(select(df,Clicks_d1:Clicks_d7),~mean(c(..1,..2,..3,..4,..5,..6,..7)))
df$clicks_sd <- pmap_dbl(select(df,Clicks_d1:Clicks_d7),~sd(c(..1,..2,..3,..4,..5,..6,..7)))



# Set the class.
class <- length(df)

# Perform stratified bootstrapping (keep 70% of observations for training and 30% for testing).
indices.training <- createDataPartition(df[,class], 
                                        times = 1, 
                                        p = .70, 
                                        list = FALSE)

# Get training and test set.
training <- df[indices.training[,1],]
test  <- df[-indices.training[,1],]

# Print class distribution.
cat("\n\n")
classDistribution(dataset.name = "df",
                  table = df,
                  class = class-10)

classDistribution(dataset.name = "training",
                  table = training,
                  class = class-10)

classDistribution(dataset.name = "test",
                  table = test,
                  class = class-10)

# Setting the formula to introduce to the xgBoost.
formula <- as.formula(paste("Cluster ~", paste(names(df)[(ncol(df)-8):ncol(df)], collapse = '+')))

# Tuned parameters (Tunning grid commented)
xgboostGrid <- expand.grid(nrounds = 6,
                           #nrounds = seq(5,10,1),
                           eta = 0.2,
                           #eta = c(0.1,0.2),
                           gamma = 1,
                           #gamma = c(0.8,0.9,1),
                           colsample_bytree = 0.7,
                           #colsample_bytree = c(0.5,0.7,1.0),
                           max_depth = 4,
                           #max_depth = c(2,4,6),
                           min_child_weight = 5,
                           #min_child_weight = seq(1,8,1),
                           subsample = 1)

xgboostControl = trainControl(method = "cv",
                              number = 10,
                              classProbs = TRUE,
                              search = "grid",
                              allowParallel = TRUE)

#Number of threads paralellised computing
i = 4

# Model training
model.training <- train(formula,
                        data = training,
                        method = "xgbTree",
                        trControl = xgboostControl,
                        tuneGrid = xgboostGrid,
                        verbose = TRUE,
                        metric = "Accuracy",
                        nthread = i)

# Stop stop-watch
end.time <- as.numeric(as.POSIXct(Sys.time()))
print(c("Elapsed time: ",round(end.time-start.time,4), "seconds"),quote=FALSE)

# Print training results
model.training
model.training$results

# Predicting test fold class (30% remaining data)
model.test.pred <- predict(model.training, 
                           test, 
                           type = "raw",
                           norm.votes = TRUE)

# Predicting test fold probability (30% remaining data)
model.test.prob <- predict(model.training, 
                           test, 
                           type = "prob",
                           norm.votes = TRUE)

# Print confusion matrix
performance <- confusionMatrix(model.test.pred, test$Cluster)
print(performance)
print(performance$byClass)


# Compute AUC for the model.
model.roc <- plot.roc(predictor = model.test.prob[,2],  
                      test$Cluster,
                      levels = rev(levels(test$Cluster)),
                      legacy.axes = FALSE,
                      percent = TRUE,
                      mar = c(4.1,4.1,0.2,0.3),
                      identity.col = "red",
                      identity.lwd = 2,
                      smooth = FALSE,
                      ci = TRUE, 
                      print.auc = TRUE,
                      auc.polygon.border=NULL,
                      lwd = 2,
                      cex.lab = 2.0, 
                      cex.axis = 1.6, 
                      font.lab = 2,
                      font.axis = 2,
                      col = "blue")

# Compute and plot confidence interval for ROC curve
ciobj <- ci.se(model.roc, specificities = seq(0, 100, 5))
plot(ciobj, type = "shape", col = "#1c61b6AA")
plot(ci(model.roc, of = "thresholds", thresholds = "best"))
