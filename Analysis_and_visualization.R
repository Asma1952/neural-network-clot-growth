
library(torch)
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(glmnet)
library(catboost)

if(!require("fastshap", quietly=TRUE)) devtools::install_github("bgreenwell/fastshap")
library(fastshap)

set.seed(8547)
torch_manual_seed(8547)

data_obj  <- readRDS("data_objects.rds")
model_obj <- readRDS("model_objects.rds")
spnn_obj  <- readRDS("spnn_objects.rds")

df         <- data_obj$df
X_train    <- data_obj$X_train; y_train <- data_obj$y_train
X_test     <- data_obj$X_test;  y_test  <- data_obj$y_test
X_train_sc <- data_obj$X_train_sc; X_test_sc <- data_obj$X_test_sc
preproc    <- data_obj$preproc; y_mean <- data_obj$y_mean; y_sd <- data_obj$y_sd
y_train_sc <- data_obj$y_train_sc; y_test_sc <- data_obj$y_test_sc
predictors <- data_obj$predictors; target <- data_obj$target

models        <- model_obj$models; preds <- model_obj$preds
model_results <- model_obj$model_results
lm_std_coef   <- model_obj$lm_std_coef; rf_imp <- model_obj$rf_imp
best_nrounds  <- model_obj$best_nrounds; xgb_cv <- model_obj$xgb_cv

result_default    <- spnn_obj$result_default
result_best       <- spnn_obj$result_best
eval_default      <- spnn_obj$eval_default
eval_best         <- spnn_obj$eval_best
best_lambdas      <- spnn_obj$best_lambdas
tune1_results     <- spnn_obj$tune1_results
tune2_results     <- spnn_obj$tune2_results
comparison_single <- spnn_obj$comparison_single

X_train_t <- torch_tensor(as.matrix(X_train_sc), dtype=torch_float())
y_train_t <- torch_tensor(matrix(y_train_sc, ncol=1), dtype=torch_float())
X_test_t  <- torch_tensor(as.matrix(X_test_sc),  dtype=torch_float())

result_default$model <- torch_load("spnn_default_model.pt")
result_best$model    <- torch_load("spnn_final_model.pt")

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

compute_loss_no_str <- function(model, x_batch, y_batch,
                                lambda_mono, lambda_smooth, lambda_l2,
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
  lsmooth <- torch_tensor(0.0)
  for(j in 1:5) {
    grad2_j <- autograd_grad(outputs=grads1[,j]$sum(), inputs=x_sub,
                             create_graph=TRUE, retain_graph=TRUE)[[1]][,j]
    lsmooth <- lsmooth + grad2_j$pow(2)$mean()
  }
  l2 <- torch_tensor(0.0)
  for(param in model$parameters) l2 <- l2 + param$pow(2)$sum()
  total <- mse_loss + lambda_mono*lmono + lambda_smooth*lsmooth + lambda_l2*l2
  list(total=total, mse=mse_loss$item(), mono=lmono$item(),
       smooth=lsmooth$item(), l2=l2$item())
}

train_model <- function(lambda_mono, lambda_str, lambda_smooth, lambda_l2,
                        n_epochs=1000, lr=1e-3,
                        x_tr_t=X_train_t, y_tr_t=y_train_t,
                        use_dropout=FALSE, dropout_rate=0.2) {
  model <- if(use_dropout) net_dropout(dropout_rate=dropout_rate) else net()
  optimizer <- optim_adam(model$parameters, lr=lr)
  for(epoch in 1:n_epochs) {
    model$train(); optimizer$zero_grad()
    loss_out <- compute_loss(model, x_tr_t, y_tr_t,
                             lambda_mono, lambda_str, lambda_smooth, lambda_l2)
    loss_out$total$backward(); optimizer$step()
  }
  model
}

train_model_no_str <- function(lambda_mono, lambda_smooth, lambda_l2,
                               n_epochs=1000, lr=1e-3,
                               x_tr_t=X_train_t, y_tr_t=y_train_t) {
  model <- net()
  optimizer <- optim_adam(model$parameters, lr=lr)
  for(epoch in 1:n_epochs) {
    model$train(); optimizer$zero_grad()
    loss_out <- compute_loss_no_str(model, x_tr_t, y_tr_t,
                                    lambda_mono, lambda_smooth, lambda_l2)
    loss_out$total$backward(); optimizer$step()
  }
  model
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

# Partial nested cv
nested_start <- Sys.time()
set.seed(8547)
outer_folds <- createFolds(df[[target]], k=5, list=TRUE)

tune1_grid_inner <- expand.grid(
  lambda_mono=c(0.001,0.005,0.01), lambda_str=c(0.005,0.01,0.05),
  lambda_smooth=c(0.005,0.01,0.02), lambda_l2=c(1e-6,1e-5,5e-5))

nested_results <- data.frame()

for(fold_i in 1:5) {
  test_idx <- outer_folds[[fold_i]]; train_idx <- unlist(outer_folds[-fold_i])
  fold_train <- df[train_idx,]; fold_test <- df[test_idx,]
  X_tr <- as.data.frame(fold_train[, predictors]); y_tr <- fold_train[[target]]
  X_te <- as.data.frame(fold_test[,  predictors]); y_te <- fold_test[[target]]
  pp <- preProcess(X_tr, method=c("center","scale"))
  X_tr_s <- predict(pp, X_tr); X_te_s <- predict(pp, X_te)
  y_mn_f <- mean(y_tr); y_sd_f <- sd(y_tr); y_tr_sc <- (y_tr-y_mn_f)/y_sd_f
  X_tr_t_f <- torch_tensor(as.matrix(X_tr_s), dtype=torch_float())
  y_tr_t_f <- torch_tensor(matrix(y_tr_sc, ncol=1), dtype=torch_float())
  X_te_t_f <- torch_tensor(as.matrix(X_te_s), dtype=torch_float())
  
  inner_results <- data.frame()
  for(j in 1:nrow(tune1_grid_inner)) {
    g <- tune1_grid_inner[j,]
    torch_manual_seed(8547 + fold_i*100 + j)
    m <- train_model(g$lambda_mono, g$lambda_str, g$lambda_smooth, g$lambda_l2,
                     n_epochs=500, lr=1e-3,
                     x_tr_t=X_tr_t_f, y_tr_t=y_tr_t_f)
    e <- evaluate_model(m, X_te_t_f, y_te, y_mn_f, y_sd_f)
    inner_results <- rbind(inner_results, data.frame(g, RMSE=e$RMSE))
  }
  inner_results <- inner_results[order(inner_results$RMSE),]
  best_inner <- inner_results[1,]
  
  nested_results <- rbind(nested_results, data.frame(
    Fold=fold_i, lambda_mono=best_inner$lambda_mono,
    lambda_str=best_inner$lambda_str, lambda_smooth=best_inner$lambda_smooth,
    lambda_l2=best_inner$lambda_l2, RMSE=best_inner$RMSE))
}

nested_time <- as.numeric(difftime(Sys.time(),nested_start,units="hours"))
write.csv(nested_results, "nested_cv_results.csv", row.names=FALSE)

repcv_start    <- Sys.time()
n_repeats      <- 5
all_cv_results <- data.frame()

for(rep_i in 1:n_repeats) {
  set.seed(8547 + rep_i)
  folds_rep <- createFolds(df[[target]], k=5, list=TRUE)
  
  for(fold_i in 1:5) {
    test_idx <- folds_rep[[fold_i]]; train_idx <- unlist(folds_rep[-fold_i])
    fold_train <- df[train_idx,]; fold_test <- df[test_idx,]
    X_tr <- as.data.frame(fold_train[, predictors]); y_tr <- fold_train[[target]]
    X_te <- as.data.frame(fold_test[,  predictors]); y_te <- fold_test[[target]]
    pp <- preProcess(X_tr, method=c("center","scale"))
    X_tr_s <- predict(pp, X_tr); X_te_s <- predict(pp, X_te)
    y_mn_f <- mean(y_tr); y_sd_f <- sd(y_tr); y_tr_sc <- (y_tr-y_mn_f)/y_sd_f
    fold_seed <- 8547 + rep_i*1000 + fold_i
    
    add <- function(name, pred) {
      all_cv_results <<- rbind(all_cv_results, data.frame(
        Repeat=rep_i, Fold=fold_i, Model=name,
        RMSE=sqrt(mean((y_te-pred)^2)), R2=cor(y_te,pred)^2))
    }
    
    add("LinearRegression", predict(lm(height~., data=cbind(X_tr_s, height=y_tr)), newdata=X_te_s))
    add("Ridge", as.numeric(predict(glmnet(as.matrix(X_tr_s), y_tr, alpha=0, lambda=10.48), as.matrix(X_te_s))))
    add("Lasso", as.numeric(predict(glmnet(as.matrix(X_tr_s), y_tr, alpha=1, lambda=0.049), as.matrix(X_te_s))))
    add("ElasticNet", as.numeric(predict(glmnet(as.matrix(X_tr_s), y_tr, alpha=0.3, lambda=0.070), as.matrix(X_te_s))))
    add("SVR", predict(svm(x=X_tr_s, y=y_tr, kernel="radial", cost=100, gamma=0.01), newdata=X_te_s))
    add("RandomForest", predict(randomForest(x=X_tr_s, y=y_tr, ntree=500, mtry=3), newdata=X_te_s))
    xgb_tr2 <- xgb.DMatrix(data=as.matrix(X_tr_s), label=y_tr)
    xgb_te2 <- xgb.DMatrix(data=as.matrix(X_te_s), label=y_te)
    xgbf <- xgb.train(params=list(objective="reg:squarederror", max_depth=5, eta=0.1,
                                  colsample_bytree=0.8, subsample=0.8),
                      data=xgb_tr2, nrounds=best_nrounds, verbose=0)
    add("XGBoost", predict(xgbf, xgb_te2))
    cb <- catboost.train(catboost.load_pool(data=X_tr_s, label=y_tr),
                         params=list(loss_function="RMSE", iterations=500,
                                     learning_rate=0.1, depth=6, verbose=0))
    add("CatBoost", catboost.predict(cb, catboost.load_pool(data=X_te_s, label=y_te)))
    
    X_tr_t_f <- torch_tensor(as.matrix(X_tr_s), dtype=torch_float())
    y_tr_t_f <- torch_tensor(matrix(y_tr_sc, ncol=1), dtype=torch_float())
    X_te_t_f <- torch_tensor(as.matrix(X_te_s), dtype=torch_float())
    
    add_spnn <- function(name, model) {
      e <- evaluate_model(model, X_te_t_f, y_te, y_mn_f, y_sd_f)
      all_cv_results <<- rbind(all_cv_results, data.frame(
        Repeat=rep_i, Fold=fold_i, Model=name, RMSE=e$RMSE, R2=e$R2))
    }
    
    torch_manual_seed(fold_seed)
    add_spnn("SPNN_default",
             train_model(0.1, 0.01, 0.01, 1e-4, n_epochs=1000, lr=1e-3,
                         x_tr_t=X_tr_t_f, y_tr_t=y_tr_t_f))
    
    torch_manual_seed(fold_seed + 50)
    add_spnn("SPNN_tuned",
             train_model(best_lambdas$lambda_mono, best_lambdas$lambda_str,
                         best_lambdas$lambda_smooth, best_lambdas$lambda_l2,
                         n_epochs=1000, lr=1e-3,
                         x_tr_t=X_tr_t_f, y_tr_t=y_tr_t_f))
    
    torch_manual_seed(fold_seed + 100)
    add_spnn("SPNN_tuned_noLstr",
             train_model_no_str(best_lambdas$lambda_mono, best_lambdas$lambda_smooth,
                                best_lambdas$lambda_l2, 1000, 1e-3, X_tr_t_f, y_tr_t_f))
    
    torch_manual_seed(fold_seed + 150)
    add_spnn("SPNN_tuned_dropout01",
             train_model(best_lambdas$lambda_mono, best_lambdas$lambda_str,
                         best_lambdas$lambda_smooth, best_lambdas$lambda_l2,
                         n_epochs=1000, lr=1e-3,
                         x_tr_t=X_tr_t_f, y_tr_t=y_tr_t_f,
                         use_dropout=TRUE, dropout_rate=0.1))
    
    torch_manual_seed(fold_seed + 200)
    add_spnn("SPNN_tuned_dropout02",
             train_model(best_lambdas$lambda_mono, best_lambdas$lambda_str,
                         best_lambdas$lambda_smooth, best_lambdas$lambda_l2,
                         n_epochs=1000, lr=1e-3,
                         x_tr_t=X_tr_t_f, y_tr_t=y_tr_t_f,
                         use_dropout=TRUE, dropout_rate=0.2))
    
    torch_manual_seed(fold_seed + 250)
    add_spnn("SPNN_tuned_dropout03",
             train_model(best_lambdas$lambda_mono, best_lambdas$lambda_str,
                         best_lambdas$lambda_smooth, best_lambdas$lambda_l2,
                         n_epochs=1000, lr=1e-3,
                         x_tr_t=X_tr_t_f, y_tr_t=y_tr_t_f,
                         use_dropout=TRUE, dropout_rate=0.3))
  }
}

repcv_time <- as.numeric(difftime(Sys.time(),repcv_start,units="hours"))

cv_summary <- all_cv_results %>%
  group_by(Model) %>%
  summarise(Mean_RMSE=round(mean(RMSE),4), SD_RMSE=round(sd(RMSE),4),
            Median_RMSE=round(median(RMSE),4),
            Mean_R2=round(mean(R2),4), SD_R2=round(sd(R2),4)) %>%
  arrange(Mean_RMSE)
write.csv(cv_summary, "repeated_cv_summary.csv", row.names=FALSE)
write.csv(all_cv_results, "repeated_cv_all_results.csv", row.names=FALSE)

png("repeated_cv_boxplot.png", width=2000, height=900, res=150)
par(mar=c(12,4,3,1))
mo <- cv_summary$Model
cp <- all_cv_results %>% mutate(Model=factor(Model, levels=mo))
boxplot(RMSE~Model, data=cp, main="Repeated 5-Fold CV (5 reps)",
        xlab="", ylab="RMSE", col="lightblue", las=2)
dev.off()

png("cv_lstr_comparison.png", width=900, height=600, res=150)
ld <- all_cv_results %>% filter(Model %in% c("SPNN_tuned","SPNN_tuned_noLstr")) %>%
  mutate(Model=factor(Model, levels=c("SPNN_tuned_noLstr","SPNN_tuned")))
boxplot(RMSE~Model, data=ld, main="Lstr Impact (tuned lambdas)",
        names=c("Without Lstr","With Lstr"),
        col=c("lightcoral","steelblue"), ylab="RMSE", las=1)
dev.off()

png("cv_dropout_comparison.png", width=1200, height=600, res=150)
dd <- all_cv_results %>%
  filter(Model %in% c("SPNN_tuned","SPNN_tuned_dropout01",
                      "SPNN_tuned_dropout02","SPNN_tuned_dropout03")) %>%
  mutate(Model=factor(Model, levels=c("SPNN_tuned","SPNN_tuned_dropout01",
                                      "SPNN_tuned_dropout02","SPNN_tuned_dropout03")))
boxplot(RMSE~Model, data=dd, main="Dropout Comparison (tuned lambdas)",
        names=c("No drop","Drop=0.1","Drop=0.2","Drop=0.3"),
        col=c("steelblue","lightgreen","lightyellow","lightcoral"),
        ylab="RMSE", las=1)
dev.off()

lstr_summary <- all_cv_results %>%
  filter(Model %in% c("SPNN_tuned","SPNN_tuned_noLstr")) %>%
  group_by(Model) %>%
  summarise(Mean_RMSE=round(mean(RMSE),4), SD_RMSE=round(sd(RMSE),4))
write.csv(lstr_summary, "lstr_comparison.csv", row.names=FALSE)

drop_summary <- all_cv_results %>%
  filter(Model %in% c("SPNN_tuned","SPNN_tuned_dropout01",
                      "SPNN_tuned_dropout02","SPNN_tuned_dropout03")) %>%
  group_by(Model) %>%
  summarise(Mean_RMSE=round(mean(RMSE),4), SD_RMSE=round(sd(RMSE),4)) %>%
  arrange(Mean_RMSE)
write.csv(drop_summary, "dropout_comparison.csv", row.names=FALSE)



ttest_results <- data.frame()

run_ttest <- function(name1, name2, label) {
  v1 <- all_cv_results$RMSE[all_cv_results$Model==name1]
  v2 <- all_cv_results$RMSE[all_cv_results$Model==name2]
  if(length(v1) == length(v2) && length(v1) > 1) {
    tt <- t.test(v1, v2, paired=TRUE)
    ttest_results <<- rbind(ttest_results, data.frame(
      Comparison=label, Mean_diff=round(mean(v1)-mean(v2),4),
      t_stat=round(tt$statistic,4), df=tt$parameter,
      p_value=round(tt$p.value,4),
      significant=ifelse(tt$p.value<0.05,"YES","NO")))
  }
}

run_ttest("SPNN_tuned", "SVR",               "SPNN_tuned vs SVR")
run_ttest("SPNN_tuned", "SPNN_default",      "SPNN_tuned vs SPNN_default")
run_ttest("SPNN_tuned", "SPNN_tuned_noLstr", "With Lstr vs Without")
run_ttest("SPNN_tuned", "SPNN_tuned_dropout01","No dropout vs Dropout 0.1")
run_ttest("SPNN_tuned", "SPNN_tuned_dropout02","No dropout vs Dropout 0.2")
run_ttest("SPNN_tuned", "SPNN_tuned_dropout03","No dropout vs Dropout 0.3")
run_ttest("SPNN_tuned", "RandomForest",      "SPNN_tuned vs RandomForest")
run_ttest("SPNN_tuned", "XGBoost",           "SPNN_tuned vs XGBoost")
run_ttest("SPNN_tuned", "CatBoost",          "SPNN_tuned vs CatBoost")
run_ttest("SPNN_tuned", "LinearRegression",  "SPNN_tuned vs LinearRegression")
write.csv(ttest_results, "paired_ttest_results.csv", row.names=FALSE)

# Subgroup error analysis
tertiles     <- quantile(y_test, probs=c(0,1/3,2/3,1))
height_group <- cut(y_test, breaks=tertiles, include.lowest=TRUE,
                    labels=c("Low","Medium","High"))

subgroup_results <- data.frame()
all_preds_list   <- preds
all_preds_list$SPNN_default <- eval_default$pred
all_preds_list$SPNN_final   <- eval_best$pred

for(name in names(all_preds_list)) {
  p <- all_preds_list[[name]]
  for(grp in levels(height_group)) {
    idx <- height_group == grp
    if(sum(idx) > 0) {
      subgroup_results <- rbind(subgroup_results, data.frame(
        Model=name, Subgroup=grp, N=sum(idx),
        Height_range=sprintf("%.0f-%.0f", min(y_test[idx]), max(y_test[idx])),
        RMSE=round(sqrt(mean((y_test[idx]-p[idx])^2)),4),
        MAE=round(mean(abs(y_test[idx]-p[idx])),4)))
    }
  }
}
write.csv(subgroup_results, "subgroup_analysis.csv", row.names=FALSE)

png("subgroup_rmse.png", width=1400, height=700, res=150)
par(mar=c(10,4,3,1))
sg_wide <- subgroup_results %>%
  select(Model, Subgroup, RMSE) %>%
  pivot_wider(names_from=Subgroup, values_from=RMSE) %>%
  arrange(Low + Medium + High)
sg_mat <- as.matrix(sg_wide[,-1])
rownames(sg_mat) <- sg_wide$Model
barplot(t(sg_mat), beside=TRUE, las=2,
        col=c("lightgreen","lightyellow","lightcoral"),
        legend.text=c("Low","Medium","High"),
        args.legend=list(x="topleft"),
        main="RMSE by Height Subgroup", ylab="RMSE")
dev.off()

# Partial dependence plots
X_train_df <- as.data.frame(X_train)

make_pdp <- function(pred_fn, model_name) {
  png(paste0("pdp_", model_name, ".png"), width=1400, height=900, res=150)
  par(mfrow=c(2,3))
  for(var in predictors) {
    var_seq <- seq(min(X_train_df[[var]]), max(X_train_df[[var]]), length.out=50)
    pdp_vals <- sapply(var_seq, function(val) {
      temp <- X_train_df; temp[[var]] <- val
      mean(pred_fn(temp))
    })
    plot(var_seq, pdp_vals, type="l", lwd=2, col="steelblue",
         main=paste("PDP:",var), xlab=var, ylab="Predicted Height")
  }
  dev.off()
}

make_pdp(function(d) predict(models$LinearRegression, predict(preproc,d)), "LinearRegression")
make_pdp(function(d) as.numeric(predict(models$Ridge, as.matrix(predict(preproc,d)))), "Ridge")
make_pdp(function(d) as.numeric(predict(models$Lasso, as.matrix(predict(preproc,d)))), "Lasso")
make_pdp(function(d) as.numeric(predict(models$ElasticNet, as.matrix(predict(preproc,d)))), "ElasticNet")
make_pdp(function(d) predict(models$SVR, predict(preproc,d)), "SVR")
make_pdp(function(d) predict(models$RandomForest, d), "RandomForest")
make_pdp(function(d) predict(models$XGBoost, xgb.DMatrix(data=as.matrix(predict(preproc,d)))), "XGBoost")
make_pdp(function(d) catboost.predict(models$CatBoost, catboost.load_pool(data=d)), "CatBoost")
make_pdp(function(d) {
  X_t <- torch_tensor(as.matrix(predict(preproc,d)), dtype=torch_float())
  result_default$model$eval()
  with_no_grad({ p <- as.numeric(result_default$model(X_t)$squeeze()) })
  p*y_sd + y_mean
}, "SPNN_default")
make_pdp(function(d) {
  X_t <- torch_tensor(as.matrix(predict(preproc,d)), dtype=torch_float())
  result_best$model$eval()
  with_no_grad({ p <- as.numeric(result_best$model(X_t)$squeeze()) })
  p*y_sd + y_mean
}, "SPNN_tuned")

# Residual analysis
make_residuals <- function(y_actual, y_pred, name) {
  res  <- y_actual - y_pred
  r2   <- round(cor(y_actual,y_pred)^2,4)
  rmse <- round(sqrt(mean(res^2)),4)
  png(paste0("residuals_",name,".png"), width=1400, height=1000, res=150)
  par(mfrow=c(2,2))
  plot(y_actual, y_pred,
       main=paste(name,"Pred vs Actual\nR²=",r2,"RMSE=",rmse),
       pch=16, col=rgb(0,0,1,0.3), xlab="Actual", ylab="Predicted")
  abline(0,1,col="red",lwd=2)
  plot(y_pred, res, main="Residuals vs Predicted",
       pch=16, col=rgb(0,0,1,0.3), xlab="Predicted", ylab="Residuals")
  abline(h=0, col="red", lwd=2, lty=2)
  hist(res, breaks=30, main="Residual Distribution",
       col="lightblue", xlab="Residuals")
  qqnorm(res, pch=16, col=rgb(0,0,1,0.3), main="Q-Q Plot")
  qqline(res, col="red", lwd=2)
  dev.off()
}

for(name in names(preds)) make_residuals(y_test, preds[[name]], name)
make_residuals(y_test, eval_default$pred, "SPNN_default")
make_residuals(y_test, eval_best$pred,    "SPNN_tuned")

# SHAP
tryCatch({
  pred_rf <- function(object, newdata) predict(object, newdata=newdata)
  shap_values <- fastshap::explain(models$RandomForest$finalModel,
                                   X=X_train_df, pred_wrapper=pred_rf, nsim=50,
                                   newdata=as.data.frame(X_test))
  shap_imp <- colMeans(abs(shap_values))
  
  png("shap_importance.png", width=900, height=600, res=150)
  barplot(sort(shap_imp, decreasing=TRUE), col="steelblue", las=2,
          main="SHAP Global Feature Importance (RF)", ylab="Mean |SHAP|")
  dev.off()
  
  low_idx  <- which.min(y_test)
  med_idx  <- order(y_test)[length(y_test)/2]
  high_idx <- which.max(y_test)
  png("shap_patients.png", width=1500, height=500, res=150)
  par(mfrow=c(1,3))
  for(i in c(low_idx, med_idx, high_idx)) {
    label <- if(i==low_idx) "Low" else if(i==med_idx) "Median" else "High"
    barplot(as.numeric(shap_values[i,]), names.arg=predictors,
            main=paste(label, "patient: actual =", round(y_test[i],1)),
            col=ifelse(as.numeric(shap_values[i,])>0,"coral","steelblue"),
            las=2, ylab="SHAP value")
    abline(h=0, lwd=2)
  }
  dev.off()
  
  shap_df <- data.frame(Feature=predictors,
                        Mean_abs_SHAP=round(as.numeric(shap_imp),4))
  shap_df <- shap_df[order(-shap_df$Mean_abs_SHAP),]
  write.csv(shap_df, "shap_importance.csv", row.names=FALSE)
}, error=function(e) message("SHAP failed: ", e$message))

# ICE plots
low_idx      <- which.min(y_test)
med_idx      <- order(y_test)[length(y_test)/2]
high_idx     <- which.max(y_test)
patient_idx    <- c(low_idx, med_idx, high_idx)
patient_labels <- c("Low","Median","High")
patient_colors <- c("steelblue","seagreen","coral")

png("ice_plots.png", width=1400, height=900, res=150)
par(mfrow=c(2,3))
for(var in predictors) {
  var_seq  <- seq(min(X_test[[var]]), max(X_test[[var]]), length.out=50)
  ice_vals <- sapply(patient_idx, function(p) {
    sapply(var_seq, function(val) {
      temp <- X_test[p,, drop=FALSE]; temp[[var]] <- val
      predict(models$RandomForest, temp)
    })
  })
  matplot(var_seq, ice_vals, type="l", lty=1, lwd=2,
          col=patient_colors, main=paste("ICE:", var),
          xlab=var, ylab="Predicted Height")
  legend("topright", legend=patient_labels,
         col=patient_colors, lty=1, lwd=2, cex=0.8)
}
dev.off()

# Additional visualizations
png("coefficient_plots.png", width=1400, height=900, res=150)
par(mfrow=c(2,2))
for(name in c("LinearRegression","Ridge","Lasso","ElasticNet")) {
  if(name == "LinearRegression") {
    coefs <- coef(models[[name]]$finalModel)[-1]
  } else {
    coefs <- as.numeric(coef(models[[name]]$finalModel,
                             s=models[[name]]$bestTune$lambda))[-1]
    names(coefs) <- predictors
  }
  barplot(sort(coefs), main=paste(name,"Coefficients"),
          col=ifelse(sort(coefs)>0,"coral","steelblue"),
          horiz=TRUE, las=2)
  abline(v=0, lwd=2)
}
dev.off()

png("regularization_paths.png", width=1400, height=600, res=150)
par(mfrow=c(1,2), mar=c(5,4,6,2))
plot(glmnet(as.matrix(X_train_sc), y_train, alpha=0), xvar="lambda",
     main="Ridge Path", label=TRUE)
plot(glmnet(as.matrix(X_train_sc), y_train, alpha=1), xvar="lambda",
     main="Lasso Path", label=TRUE)
dev.off()

png("rf_oob_error.png", width=900, height=600, res=150)
plot(models$RandomForest$finalModel, main="RF OOB Error vs Trees")
dev.off()

png("rf_importance_detailed.png", width=1000, height=600, res=150)
varImpPlot(models$RandomForest$finalModel, main="RF Variable Importance",
           pch=16, col="steelblue")
dev.off()

png("xgboost_importance.png", width=900, height=600, res=150)
xgb.plot.importance(xgb.importance(model=models$XGBoost),
                    main="XGBoost Feature Importance")
dev.off()

png("xgboost_learning_curve.png", width=900, height=600, res=150)
plot(xgb_cv$evaluation_log$iter, xgb_cv$evaluation_log$train_rmse_mean,
     type="l", col="coral", lwd=2, main="XGBoost Learning Curve",
     xlab="Rounds", ylab="RMSE")
lines(xgb_cv$evaluation_log$iter, xgb_cv$evaluation_log$test_rmse_mean,
      col="steelblue", lwd=2)
abline(v=best_nrounds, col="black", lty=2)
legend("topright", c("Train","CV","Best"),
       col=c("coral","steelblue","black"), lty=c(1,1,2), lwd=2)
dev.off()


X_grad <- torch_tensor(as.matrix(X_train_sc), dtype=torch_float())$requires_grad_(TRUE)
result_default$model$eval()
y_grad    <- result_default$model(X_grad)
grads_all <- autograd_grad(outputs=y_grad$sum(), inputs=X_grad,
                           create_graph=FALSE)[[1]]
grads_mean <- as.numeric(colMeans(as.matrix(grads_all$detach())))
names(grads_mean) <- predictors

mono_check <- data.frame(
  Feature   = predictors,
  Expected  = c("negative","positive","positive","negative","negative"),
  Mean_Grad = round(grads_mean,4),
  Compliant = c(grads_mean[1]<0, grads_mean[2]>0, grads_mean[3]>0,
                grads_mean[4]<0, grads_mean[5]<0))
write.csv(mono_check, "monotonicity_compliance.csv", row.names=FALSE)

png("spnn_gradient_visualization.png", width=1100, height=600, res=150)
par(mfrow=c(1,2))
barplot(grads_mean, col=ifelse(grads_mean>0,"coral","steelblue"),
        main="SPNN Mean Learned Gradients", las=2, ylab="Mean ∂f/∂x")
abline(h=0, lwd=2)
barplot(abs(grads_mean), col="mediumpurple",
        main="SPNN Gradient Magnitudes", las=2, ylab="|Mean ∂f/∂x|")
dev.off()

while(dev.cur() > 1) dev.off()