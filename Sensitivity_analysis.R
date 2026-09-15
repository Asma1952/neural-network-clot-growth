
library(torch)
library(tidyverse)
library(caret)
library(xgboost)
library(e1071)

data_obj  <- readRDS("data_objects.rds")
model_obj <- readRDS("model_objects.rds")
spnn_obj  <- readRDS("spnn_objects.rds")

df         <- data_obj$df
X_train    <- data_obj$X_train; y_train <- data_obj$y_train
X_test     <- data_obj$X_test;  y_test  <- data_obj$y_test
X_train_sc <- data_obj$X_train_sc; X_test_sc <- data_obj$X_test_sc
preproc    <- data_obj$preproc
y_mean     <- data_obj$y_mean; y_sd <- data_obj$y_sd
y_train_sc <- data_obj$y_train_sc
predictors <- data_obj$predictors

models       <- model_obj$models; preds <- model_obj$preds
best_nrounds <- model_obj$best_nrounds

result_default <- spnn_obj$result_default
result_best    <- spnn_obj$result_best
eval_default   <- spnn_obj$eval_default
eval_best      <- spnn_obj$eval_best
best_lambdas   <- spnn_obj$best_lambdas

X_train_t <- torch_tensor(as.matrix(X_train_sc), dtype=torch_float())
y_train_t <- torch_tensor(matrix(y_train_sc, ncol=1), dtype=torch_float())
X_test_t  <- torch_tensor(as.matrix(X_test_sc),  dtype=torch_float())


result_best$model    <- torch_load("spnn_final_model.pt")
result_default$model <- torch_load("spnn_default_model.pt")
result_best$model$eval()
result_default$model$eval()


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
  grads1 <- autograd_grad(outputs=y_sub$sum(), inputs=x_sub,
                          create_graph=TRUE)[[1]]
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


lm <- best_lambdas$lambda_mono
ls <- best_lambdas$lambda_str
lk <- best_lambdas$lambda_smooth
ll <- best_lambdas$lambda_l2

train_eval_spnn <- function(lambda_mono, lambda_str, lambda_smooth, lambda_l2,
                            seed=8547) {
  torch_manual_seed(seed)
  m <- train_model(lambda_mono, lambda_str, lambda_smooth, lambda_l2,
                   n_epochs=1000, lr=1e-3,
                   x_tr_t=X_train_t, y_tr_t=y_train_t)
  e <- evaluate_model(m, X_test_t, y_test, y_mean, y_sd)
  list(model=m, RMSE=e$RMSE, R2=e$R2, pred=e$pred)
}


lambda_multipliers  <- c(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
lambda_names        <- c("lambda_mono","lambda_str","lambda_smooth","lambda_l2")
lambda_tuned        <- c(lm, ls, lk, ll)
lambda_sens_results <- data.frame()

for(li in seq_along(lambda_names)) {
  for(mult in lambda_multipliers) {
    lambdas <- lambda_tuned
    lambdas[li] <- lambda_tuned[li] * mult
    e <- train_eval_spnn(lambdas[1], lambdas[2], lambdas[3], lambdas[4],
                         seed=8547 + li*100 + which(lambda_multipliers==mult))
    lambda_sens_results <- rbind(lambda_sens_results, data.frame(
      Lambda=lambda_names[li], Tuned_value=lambda_tuned[li],
      Multiplier=mult, Actual_value=lambdas[li],
      RMSE=e$RMSE, R2=e$R2))
  }
}

write.csv(lambda_sens_results, "lambda_sensitivity.csv", row.names=FALSE)

png("lambda_sensitivity.png", width=1600, height=1200, res=150)
par(mfrow=c(2,2), mar=c(5,4,4,2))
for(ln in lambda_names) {
  sub        <- lambda_sens_results[lambda_sens_results$Lambda==ln,]
  tuned_rmse <- sub$RMSE[sub$Multiplier==1.0]
  plot(sub$Multiplier, sub$RMSE, type="b", pch=16, lwd=2, col="steelblue",
       log="x", main=paste("Lambda sensitivity:", ln),
       xlab="Multiplier (log scale)", ylab="Test RMSE", xaxt="n")
  axis(1, at=lambda_multipliers, labels=lambda_multipliers)
  abline(h=tuned_rmse, col="red", lty=2, lwd=1.5)
  abline(v=1, col="black", lty=3, lwd=1.5)
  legend("topright", legend=c("RMSE","Tuned RMSE"),
         col=c("steelblue","red"), lty=c(1,2), lwd=2, cex=0.8)
}
dev.off()

lambda_sens_summary <- lambda_sens_results %>%
  group_by(Lambda) %>%
  summarise(Min_RMSE=round(min(RMSE),4), Max_RMSE=round(max(RMSE),4),
            Range_RMSE=round(max(RMSE)-min(RMSE),4),
            Tuned_RMSE=round(RMSE[Multiplier==1.0],4), .groups="drop") %>%
  arrange(desc(Range_RMSE))
write.csv(lambda_sens_summary, "lambda_sensitivity_summary.csv", row.names=FALSE)


noise_fractions <- c(0, 0.05, 0.10, 0.20, 0.30, 0.50)
n_noise_reps    <- 10
feature_sds     <- apply(X_train, 2, sd)
perturb_results <- data.frame()

for(frac in noise_fractions) {
  rep_rmse <- matrix(NA, nrow=n_noise_reps, ncol=5,
                     dimnames=list(NULL,
                       c("SPNN_tuned","SPNN_default","SVR","RF","XGBoost")))
  for(r in 1:n_noise_reps) {
    set.seed(8547 + r)
    if(frac == 0) {
      X_noisy_orig <- X_test
    } else {
      noise_orig <- sapply(seq_along(predictors), function(j)
        rnorm(nrow(X_test), mean=0, sd=feature_sds[j]*frac))
      colnames(noise_orig) <- predictors
      X_noisy_orig <- X_test + noise_orig
    }
    X_noisy_orig <- as.data.frame(X_noisy_orig)
    colnames(X_noisy_orig) <- predictors
    X_noisy_sc <- predict(preproc, X_noisy_orig)
    X_noisy_t  <- torch_tensor(as.matrix(X_noisy_sc), dtype=torch_float())

    result_best$model$eval()
    with_no_grad({
      p_spnn_tuned <- as.numeric(result_best$model(X_noisy_t)$squeeze())*y_sd+y_mean
    })
    result_default$model$eval()
    with_no_grad({
      p_spnn_def <- as.numeric(result_default$model(X_noisy_t)$squeeze())*y_sd+y_mean
    })
    p_svr <- predict(models$SVR, X_noisy_sc)
    p_rf  <- predict(models$RandomForest, X_noisy_orig)
    p_xgb <- predict(models$XGBoost, xgb.DMatrix(data=as.matrix(X_noisy_sc)))

    rep_rmse[r,"SPNN_tuned"]   <- sqrt(mean((y_test-p_spnn_tuned)^2))
    rep_rmse[r,"SPNN_default"] <- sqrt(mean((y_test-p_spnn_def)^2))
    rep_rmse[r,"SVR"]          <- sqrt(mean((y_test-p_svr)^2))
    rep_rmse[r,"RF"]           <- sqrt(mean((y_test-p_rf)^2))
    rep_rmse[r,"XGBoost"]      <- sqrt(mean((y_test-p_xgb)^2))
  }
  mean_rmse <- colMeans(rep_rmse); sd_rmse <- apply(rep_rmse,2,sd)
  for(model_name in colnames(rep_rmse)) {
    perturb_results <- rbind(perturb_results, data.frame(
      Noise_fraction=frac, Model=model_name,
      Mean_RMSE=round(mean_rmse[model_name],4),
      SD_RMSE=round(sd_rmse[model_name],4)))
  }
}

write.csv(perturb_results, "perturbation_stability.csv", row.names=FALSE)

perturb_wide <- perturb_results %>%
  select(Noise_fraction, Model, Mean_RMSE) %>%
  pivot_wider(names_from=Model, values_from=Mean_RMSE)

model_cols <- c("SPNN_tuned"="steelblue", "SPNN_default"="dodgerblue",
                "SVR"="coral", "RF"="seagreen", "XGBoost"="mediumpurple")
model_lty  <- c(1,2,1,1,1)

png("perturbation_stability.png", width=1000, height=700, res=150)
y_lim <- range(perturb_results$Mean_RMSE, na.rm=TRUE)*c(0.98,1.05)
plot(noise_fractions, perturb_wide$SPNN_tuned,
     type="b", pch=16, lwd=2.5, col=model_cols["SPNN_tuned"], ylim=y_lim,
     main="Input Perturbation Stability\n(mean RMSE over 10 noise draws)",
     xlab="Noise as fraction of feature SD", ylab="Mean Test RMSE")
for(i in 2:length(model_cols)) {
  mn <- names(model_cols)[i]
  lines(noise_fractions, perturb_wide[[mn]], type="b", pch=16, lwd=2,
        col=model_cols[mn], lty=model_lty[i])
}
legend("topleft", legend=names(model_cols),
       col=model_cols, lty=model_lty, lwd=2, cex=0.85)
dev.off()


mono_dir <- c(-1, 1, 1, -1, -1)
names(mono_dir) <- predictors

X_test_grad <- torch_tensor(as.matrix(X_test_sc),
                            dtype=torch_float())$requires_grad_(TRUE)
result_best$model$eval()
y_test_grad <- result_best$model(X_test_grad)
grads_test  <- autograd_grad(outputs=y_test_grad$sum(), inputs=X_test_grad,
                             create_graph=FALSE)[[1]]
grads_mat   <- as.matrix(grads_test$detach())
colnames(grads_mat) <- predictors

grad_summary <- data.frame(
  Feature       = predictors,
  Expected      = ifelse(mono_dir>0,"positive","negative"),
  Mean_grad     = round(colMeans(grads_mat),4),
  SD_grad       = round(apply(grads_mat,2,sd),4),
  P5_grad       = round(apply(grads_mat,2,quantile,0.05),4),
  P95_grad      = round(apply(grads_mat,2,quantile,0.95),4),
  Pct_compliant = round(100*sapply(seq_along(predictors), function(j) {
    if(mono_dir[j]>0) mean(grads_mat[,j]>0) else mean(grads_mat[,j]<0)
  }),1))
write.csv(grad_summary, "gradient_sensitivity.csv", row.names=FALSE)

png("gradient_distributions.png", width=1000, height=600, res=150)
par(mar=c(6,4,4,2))
boxplot(as.data.frame(grads_mat),
        main="SPNN Gradient Distribution by Feature (test set)",
        ylab=expression(partialdiff*hat(y)/partialdiff*x[j]),
        col="lightsteelblue", las=2, outline=FALSE)
abline(h=0, col="red", lty=2, lwd=1.5)
dev.off()

set.seed(8547)
hm_idx    <- sort(sample(nrow(grads_mat), min(200,nrow(grads_mat))))
hm_data   <- grads_mat[hm_idx,]
hm_scaled <- apply(hm_data, 2, function(x) x/max(abs(x)+1e-8))

png("gradient_heatmap.png", width=1100, height=800, res=150)
par(mar=c(6,5,4,5))
rng <- max(abs(hm_scaled))
image(t(hm_scaled),
      col=colorRampPalette(c("steelblue","white","coral"))(101),
      zlim=c(-rng,rng), axes=FALSE,
      main="Gradient Heatmap: scaled gradients (SPNN tuned)\n(rows = test observations, columns = features)")
axis(1, at=seq(0,1,length.out=ncol(hm_scaled)),
     labels=predictors, las=2, cex.axis=0.85)
axis(2, at=c(0,0.5,1), labels=c("1","midpoint",nrow(hm_scaled)), las=1)
box()
dev.off()

noise_fractions_mono <- c(0, 0.05, 0.10, 0.20)
n_mono_reps          <- 20
mono_robust_results  <- data.frame()

for(frac in noise_fractions_mono) {
  rep_compliance <- matrix(NA, nrow=n_mono_reps, ncol=5)
  colnames(rep_compliance) <- predictors
  for(r in 1:n_mono_reps) {
    set.seed(8547 + r*7)
    if(frac == 0) {
      X_noisy_orig_m <- X_test
    } else {
      noise_om <- sapply(seq_along(predictors), function(j)
        rnorm(nrow(X_test), mean=0, sd=feature_sds[j]*frac))
      colnames(noise_om) <- predictors
      X_noisy_orig_m <- X_test + noise_om
    }
    X_noisy_orig_m <- as.data.frame(X_noisy_orig_m)
    colnames(X_noisy_orig_m) <- predictors
    X_noisy_sc_m <- predict(preproc, X_noisy_orig_m)
    X_ng <- torch_tensor(as.matrix(X_noisy_sc_m),
                         dtype=torch_float())$requires_grad_(TRUE)
    result_best$model$eval()
    y_ng <- result_best$model(X_ng)
    g_ng <- autograd_grad(outputs=y_ng$sum(), inputs=X_ng,
                          create_graph=FALSE)[[1]]
    g_mat <- as.matrix(g_ng$detach())
    rep_compliance[r,] <- sapply(seq_along(predictors), function(j) {
      if(mono_dir[j]>0) mean(g_mat[,j]>0) else mean(g_mat[,j]<0)
    })
  }
  mean_comp <- colMeans(rep_compliance); sd_comp <- apply(rep_compliance,2,sd)
  for(j in seq_along(predictors)) {
    mono_robust_results <- rbind(mono_robust_results, data.frame(
      Noise_fraction=frac, Feature=predictors[j],
      Expected=ifelse(mono_dir[j]>0,"positive","negative"),
      Mean_compliance=round(mean_comp[j]*100,2),
      SD_compliance=round(sd_comp[j]*100,2)))
  }
}
write.csv(mono_robust_results, "monotonicity_robustness.csv", row.names=FALSE)


                         
png("monotonicity_robustness.png", width=1000, height=650, res=150)
feat_cols <- c("steelblue","coral","seagreen","mediumpurple","goldenrod")
names(feat_cols) <- predictors
plot(noise_fractions_mono, rep(100,length(noise_fractions_mono)),
     type="n", ylim=c(0,105),
     main="Monotonicity Compliance under Input Noise\n(% observations with correct gradient sign)",
     xlab="Noise as fraction of feature SD", ylab="Sign compliance (%)")
abline(h=100, col="grey80", lty=2); abline(h=50, col="grey80", lty=3)
for(j in seq_along(predictors)) {
  feat <- predictors[j]
  sub  <- mono_robust_results[mono_robust_results$Feature==feat,]
  lines(sub$Noise_fraction, sub$Mean_compliance,
        type="b", pch=16, lwd=2, col=feat_cols[feat])
}
legend("right", legend=predictors, col=feat_cols, lty=1, lwd=2, pch=16, cex=0.85)
dev.off()

while(dev.cur() > 1) dev.off()
