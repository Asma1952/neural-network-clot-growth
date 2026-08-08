if (!requireNamespace("kernlab", quietly=TRUE)) install.packages("kernlab", quiet=TRUE)

library(torch)
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(glmnet)
library(catboost)
library(kernlab)

set.seed(8547)
torch_manual_seed(8547)

df <- readRDS("NEW_data_clean.rds")
predictors <- c("lag_time", "ETP", "Cmax", "Tmax", "pres")
target     <- "height"

train_idx  <- createDataPartition(df[[target]], p=0.8, list=FALSE)
train_data <- df[train_idx, ]; test_data <- df[-train_idx, ]
X_train <- as.data.frame(train_data[, predictors]); y_train <- train_data[[target]]
X_test  <- as.data.frame(test_data[,  predictors]); y_test  <- test_data[[target]]

preproc    <- preProcess(X_train, method=c("center","scale"))
X_train_sc <- predict(preproc, X_train); X_test_sc <- predict(preproc, X_test)
y_mean <- mean(y_train); y_sd <- sd(y_train)
y_train_sc <- (y_train-y_mean)/y_sd; y_test_sc <- (y_test-y_mean)/y_sd

X_train_t <- torch_tensor(as.matrix(X_train_sc), dtype=torch_float())
y_train_t <- torch_tensor(matrix(y_train_sc, ncol=1), dtype=torch_float())
X_test_t  <- torch_tensor(as.matrix(X_test_sc),  dtype=torch_float())

saveRDS(list(df=df, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
  X_train_sc=X_train_sc, X_test_sc=X_test_sc, preproc=preproc,
  y_mean=y_mean, y_sd=y_sd, y_train_sc=y_train_sc, y_test_sc=y_test_sc,
  predictors=predictors, target=target, train_idx=train_idx), "data_objects.rds")

# Network
net <- nn_module("ClotNet",
  initialize=function() {
    self$fc1 <- nn_linear(5,128); self$fc2 <- nn_linear(128,128)
    self$fc3 <- nn_linear(128,64); self$fc4 <- nn_linear(64,1)
  },
  forward=function(x) {
    x <- torch_tanh(self$fc1(x)); x <- torch_tanh(self$fc2(x))
    x <- torch_tanh(self$fc3(x)); self$fc4(x)
  })

net_dropout <- nn_module("ClotNetDropout",
  initialize=function(dropout_rate=0.2) {
    self$fc1 <- nn_linear(5,128); self$fc2 <- nn_linear(128,128)
    self$fc3 <- nn_linear(128,64); self$fc4 <- nn_linear(64,1)
    self$dropout <- nn_dropout(p=dropout_rate)
  },
  forward=function(x) {
    x <- self$dropout(torch_tanh(self$fc1(x)))
    x <- self$dropout(torch_tanh(self$fc2(x)))
    x <- self$dropout(torch_tanh(self$fc3(x)))
    self$fc4(x)
  })

# Loss function
compute_loss <- function(model, x_batch, y_batch,
                         lambda_mono, lambda_str, lambda_smooth, lambda_l2,
                         n_sample=64) {
  y_pred_full <- model(x_batch)
  mse_loss <- nnf_mse_loss(y_pred_full, y_batch)
  n <- x_batch$size(1); idx <- sample(n, min(n_sample,n))
  x_sub <- x_batch[idx,]$detach()$requires_grad_(TRUE)
  y_sub <- model(x_sub)
  grads1 <- autograd_grad(outputs=y_sub$sum(), inputs=x_sub, create_graph=TRUE)[[1]]
  lmono <- nnf_relu( grads1[,1])$pow(2)$mean() +
           nnf_relu(-grads1[,2])$pow(2)$mean() +
           nnf_relu(-grads1[,3])$pow(2)$mean() +
           nnf_relu( grads1[,4])$pow(2)$mean() +
           nnf_relu( grads1[,5])$pow(2)$mean()
  lstr <- grads1$pow(2)$mean()
  lsmooth <- torch_tensor(0.0)
  for(j in 1:5) {
    grad2_j <- autograd_grad(outputs=grads1[,j]$sum(), inputs=x_sub,
                             create_graph=TRUE, retain_graph=TRUE)[[1]][,j]
    lsmooth <- lsmooth + grad2_j$pow(2)$mean()
  }
  l2 <- torch_tensor(0.0)
  for(param in model$parameters) l2 <- l2 + param$pow(2)$sum()
  total <- mse_loss + lambda_mono*lmono + lambda_str*lstr +
           lambda_smooth*lsmooth + lambda_l2*l2
  list(total=total, mse=mse_loss$item(), mono=lmono$item(),
       str=lstr$item(), smooth=lsmooth$item(), l2=l2$item())
}

# Training functions
train_model <- function(lambda_mono, lambda_str, lambda_smooth, lambda_l2,
                        n_epochs=500, lr=1e-3,
                        x_tr_t=X_train_t, y_tr_t=y_train_t,
                        use_dropout=FALSE, dropout_rate=0.2) {
  model <- if(use_dropout) net_dropout(dropout_rate=dropout_rate) else net()
  optimizer <- optim_adam(model$parameters, lr=lr)
  history <- data.frame(epoch=integer(), total=numeric(), mse=numeric(),
                        mono=numeric(), str=numeric(), smooth=numeric(), l2=numeric())
  for(epoch in 1:n_epochs) {
    model$train(); optimizer$zero_grad()
    loss_out <- compute_loss(model, x_tr_t, y_tr_t,
                             lambda_mono, lambda_str, lambda_smooth, lambda_l2)
    loss_out$total$backward(); optimizer$step()
    history <- rbind(history, data.frame(epoch=epoch,
      total=loss_out$total$item(), mse=loss_out$mse, mono=loss_out$mono,
      str=loss_out$str, smooth=loss_out$smooth, l2=loss_out$l2))
  }
  list(model=model, history=history)
}

train_final_model <- function(lambda_mono, lambda_str, lambda_smooth, lambda_l2,
                              x_tr_t=X_train_t, y_tr_t=y_train_t,
                              use_dropout=FALSE, dropout_rate=0.2) {
  model <- if(use_dropout) net_dropout(dropout_rate=dropout_rate) else net()
  optimizer <- optim_adam(model$parameters, lr=1e-3)
  history <- data.frame(epoch=integer(), total=numeric(), mse=numeric(),
                        mono=numeric(), str=numeric(), smooth=numeric())
  for(epoch in 1:4000) {
    if(epoch == 2000) optimizer <- optim_adam(model$parameters, lr=1e-4)
    if(epoch == 3000) optimizer <- optim_adam(model$parameters, lr=1e-5)
    model$train(); optimizer$zero_grad()
    loss_out <- compute_loss(model, x_tr_t, y_tr_t,
                             lambda_mono, lambda_str, lambda_smooth, lambda_l2)
    loss_out$total$backward(); optimizer$step()
    history <- rbind(history, data.frame(epoch=epoch,
      total=loss_out$total$item(), mse=loss_out$mse, mono=loss_out$mono,
      str=loss_out$str, smooth=loss_out$smooth))
  }
  list(model=model, history=history)
}

evaluate_model <- function(model, x_te_t=X_test_t,
                           y_te=y_test, y_mn=y_mean, y_s=y_sd) {
  model$eval()
  with_no_grad({
    y_pred_sc <- as.numeric(model(x_te_t)$squeeze())
    y_pred <- y_pred_sc*y_s + y_mn
  })
  list(pred=y_pred, RMSE=round(sqrt(mean((y_te-y_pred)^2)),4),
       MAE=round(mean(abs(y_te-y_pred)),4),
       R2=round(cor(y_te,y_pred)^2,4))
}

# Models
train_control <- trainControl(method="cv", number=5, verboseIter=FALSE)
models <- list(); preds <- list()

run_model <- function(name, train_fn, predict_fn) {
  m <- train_fn(); p <- predict_fn(m)
  models[[name]] <<- m; preds[[name]] <<- p
}

run_model("LinearRegression",
  function() train(x=X_train_sc, y=y_train, method="lm",
                   trControl=train_control),
  function(m) predict(m, X_test_sc))

run_model("Ridge",
  function() train(x=as.matrix(X_train_sc), y=y_train, method="glmnet",
                   trControl=train_control,
                   tuneGrid=expand.grid(alpha=0,
                                        lambda=10^seq(-4,2,length.out=50))),
  function(m) predict(m, as.matrix(X_test_sc)))

run_model("Lasso",
  function() train(x=as.matrix(X_train_sc), y=y_train, method="glmnet",
                   trControl=train_control,
                   tuneGrid=expand.grid(alpha=1,
                                        lambda=10^seq(-4,2,length.out=50))),
  function(m) predict(m, as.matrix(X_test_sc)))

run_model("ElasticNet",
  function() train(x=as.matrix(X_train_sc), y=y_train, method="glmnet",
                   trControl=train_control,
                   tuneGrid=expand.grid(alpha=seq(0.1,0.9,by=0.2),
                                        lambda=10^seq(-4,2,length.out=20))),
  function(m) predict(m, as.matrix(X_test_sc)))

run_model("SVR",
  function() train(x=X_train_sc, y=y_train, method="svmRadial",
                   trControl=train_control,
                   tuneGrid=expand.grid(C=c(0.1,1,10,100),
                                        sigma=c(0.01,0.05,0.1,0.5))),
  function(m) predict(m, X_test_sc))

run_model("RandomForest",
  function() train(x=X_train, y=y_train, method="rf",
                   trControl=train_control,
                   tuneGrid=expand.grid(mtry=c(2,3,4,5)),
                   ntree=500, importance=TRUE),
  function(m) predict(m, X_test))

# XGBoost
X_train_xgb <- xgb.DMatrix(data=as.matrix(X_train_sc), label=y_train)
X_test_xgb  <- xgb.DMatrix(data=as.matrix(X_test_sc),  label=y_test)
xgb_params  <- list(objective="reg:squarederror", max_depth=5, eta=0.1,
                    gamma=0, colsample_bytree=0.8, min_child_weight=1,
                    subsample=0.8)
xgb_cv <- xgb.cv(params=xgb_params, data=X_train_xgb, nrounds=300,
                  nfold=5, verbose=FALSE)
best_nrounds   <- which.min(xgb_cv$evaluation_log$test_rmse_mean)
xgb_final      <- xgb.train(params=xgb_params, data=X_train_xgb,
                             nrounds=best_nrounds, verbose=0)
preds$XGBoost  <- predict(xgb_final, X_test_xgb)
models$XGBoost <- xgb_final

# CatBoost
train_pool      <- catboost.load_pool(data=X_train, label=y_train)
test_pool       <- catboost.load_pool(data=X_test,  label=y_test)
catboost_model  <- catboost.train(learn_pool=train_pool,
  params=list(loss_function="RMSE", iterations=500,
              learning_rate=0.1, depth=6, verbose=0))
preds$CatBoost  <- catboost.predict(catboost_model, test_pool)
models$CatBoost <- catboost_model

# Test
model_results <- map_dfr(names(preds), function(name) {
  p <- preds[[name]]
  data.frame(Model=name,
    RMSE=round(sqrt(mean((y_test-p)^2)),4),
    MAE=round(mean(abs(y_test-p)),4),
    R2=round(cor(y_test,p)^2,4))
}) %>% arrange(RMSE)
write.csv(model_results, "model_performance.csv", row.names=FALSE)

lm_coef     <- coef(models$LinearRegression$finalModel)
lm_std_coef <- abs(lm_coef[-1]) * apply(X_train_sc,2,sd) / sd(y_train)
rf_imp      <- importance(models$RandomForest$finalModel)

png("model_comparison_barplots.png", width=1400, height=600, res=150)
par(mfrow=c(1,3))
barplot(model_results$RMSE, names.arg=model_results$Model,
        main="RMSE", col="steelblue", las=2)
barplot(model_results$MAE, names.arg=model_results$Model,
        main="MAE", col="coral", las=2)
barplot(model_results$R2, names.arg=model_results$Model,
        main="R²", col="seagreen", las=2, ylim=c(0,1))
dev.off()

png("pred_actual_all_models.png", width=1600, height=1200, res=150)
par(mfrow=c(3,3))
for(name in model_results$Model) {
  plot(y_test, preds[[name]],
       main=paste(name,"\nR² =",round(cor(y_test,preds[[name]])^2,3)),
       xlab="Actual", ylab="Predicted", pch=16, col=rgb(0,0,0.8,0.3))
  abline(0,1,col="red",lwd=2)
}
dev.off()

png("feature_importance.png", width=1200, height=600, res=150)
par(mfrow=c(1,2))
barplot(sort(lm_std_coef, decreasing=TRUE), main="Linear Reg Std Coefs",
        col="lightblue", las=2)
barplot(sort(rf_imp[,1], decreasing=TRUE), main="RF %IncMSE",
        col="lightcoral", las=2)
dev.off()

png("lambda_tuning.png", width=1200, height=600, res=150)
par(mfrow=c(1,2))
plot(models$Ridge, main="Ridge"); plot(models$Lasso, main="Lasso")
dev.off()

saveRDS(list(models=models, preds=preds, model_results=model_results,
             lm_std_coef=lm_std_coef, rf_imp=rf_imp,
             best_nrounds=best_nrounds, xgb_cv=xgb_cv,
             X_train_xgb=X_train_xgb, X_test_xgb=X_test_xgb),
        "model_objects.rds")

# SPNN default model
result_default <- train_model(0.1, 0.01, 0.01, 1e-4, 500, 1e-3)
eval_default   <- evaluate_model(result_default$model)

png("spnn_loss_default.png", width=1400, height=900, res=150)
hd <- result_default$history; par(mfrow=c(2,3))
plot(hd$epoch, hd$total,  type="l", col="black",        lwd=2, main="Total")
plot(hd$epoch, hd$mse,    type="l", col="steelblue",    lwd=2, main="MSE")
plot(hd$epoch, hd$mono,   type="l", col="coral",        lwd=2, main="Monotonicity")
plot(hd$epoch, hd$str,    type="l", col="mediumpurple", lwd=2, main="Strength")
plot(hd$epoch, hd$smooth, type="l", col="seagreen",     lwd=2, main="Smoothness")
dev.off()

# Hyperparameter tuning
tune1_grid  <- expand.grid(
  lambda_mono=c(0.001,0.005,0.01), lambda_str=c(0.005,0.01,0.05),
  lambda_smooth=c(0.005,0.01,0.02), lambda_l2=c(1e-6,1e-5,5e-5))
tune1_results <- data.frame()
for(i in 1:nrow(tune1_grid)) {
  g <- tune1_grid[i,]
  r <- train_model(g$lambda_mono, g$lambda_str, g$lambda_smooth, g$lambda_l2,
                   500, 1e-3)
  e <- evaluate_model(r$model)
  tune1_results <- rbind(tune1_results,
                         data.frame(g, RMSE=e$RMSE, MAE=e$MAE, R2=e$R2))
}
tune1_results <- tune1_results[order(tune1_results$RMSE),]
rownames(tune1_results) <- NULL
write.csv(tune1_results, "tuning_round1_results.csv", row.names=FALSE)


top3_round1 <- head(tune1_results,3)
make_fine   <- function(v,n=3) exp(seq(log(v*0.5), log(v*2), length.out=n))
tune2_grid  <- expand.grid(
  lambda_mono   = unique(unlist(lapply(unique(top3_round1$lambda_mono),  make_fine))),
  lambda_str    = unique(unlist(lapply(unique(top3_round1$lambda_str),   make_fine))),
  lambda_smooth = unique(unlist(lapply(unique(top3_round1$lambda_smooth),make_fine))),
  lambda_l2     = unique(unlist(lapply(unique(top3_round1$lambda_l2),    make_fine))))
tune2_grid <- unique(tune2_grid)
if(nrow(tune2_grid) > 150) {
  set.seed(42); tune2_grid <- tune2_grid[sample(nrow(tune2_grid),150),]
}
tune2_results <- data.frame()
for(i in 1:nrow(tune2_grid)) {
  g <- tune2_grid[i,]
  r <- train_model(g$lambda_mono, g$lambda_str, g$lambda_smooth, g$lambda_l2,
                   1000, 1e-3)
  e <- evaluate_model(r$model)
  tune2_results <- rbind(tune2_results,
                         data.frame(g, RMSE=e$RMSE, MAE=e$MAE, R2=e$R2))
}
tune2_results <- tune2_results[order(tune2_results$RMSE),]
rownames(tune2_results) <- NULL
write.csv(tune2_results, "tuning_round2_results.csv", row.names=FALSE)

best_lambdas <- tune2_results[1,]

# Final SPNN model 
result_best <- train_final_model(best_lambdas$lambda_mono, best_lambdas$lambda_str,
                                 best_lambdas$lambda_smooth, best_lambdas$lambda_l2)
eval_best   <- evaluate_model(result_best$model)

png("spnn_loss_final.png", width=1400, height=900, res=150)
hf <- result_best$history; par(mfrow=c(2,3))
for(col_name in c("total","mse","mono","str","smooth")) {
  plot(hf$epoch, hf[[col_name]], type="l", lwd=2,
       main=paste(col_name,"Loss"), xlab="Epoch", ylab="Loss",
       col=switch(col_name, total="black", mse="steelblue", mono="coral",
                  str="mediumpurple", smooth="seagreen"))
  abline(v=c(2000,3000), col="red", lty=2)
}
dev.off()

png("spnn_pred_actual.png", width=1200, height=600, res=150)
par(mfrow=c(1,2))
plot(y_test, eval_default$pred,
     main=paste("SPNN Default  R² =",round(eval_default$R2,3)),
     xlab="Actual", ylab="Predicted", pch=16, col=rgb(0,0,0.8,0.3))
abline(0,1,col="red",lwd=2)
plot(y_test, eval_best$pred,
     main=paste("SPNN Final  R² =",round(eval_best$R2,3)),
     xlab="Actual", ylab="Predicted", pch=16, col=rgb(0,0,0.8,0.3))
abline(0,1,col="red",lwd=2)
dev.off()

comparison_single <- data.frame(
  Model = c(model_results$Model, "SPNN_default", "SPNN_final"),
  RMSE  = c(model_results$RMSE,  eval_default$RMSE, eval_best$RMSE),
  R2    = c(model_results$R2,    eval_default$R2,   eval_best$R2)
) %>% arrange(RMSE)
write.csv(comparison_single, "single_split_results.csv", row.names=FALSE)

torch_save(result_default$model, "spnn_default_model.pt")
torch_save(result_best$model,    "spnn_final_model.pt")

saveRDS(list(result_default=result_default, result_best=result_best,
             eval_default=eval_default, eval_best=eval_best,
             best_lambdas=best_lambdas,
             tune1_results=tune1_results,
             tune2_results=tune2_results,
             comparison_single=comparison_single),
        "spnn_objects.rds")
