library(tidyverse)
library(caret)
library(recipes)
library(glmnet)
library(gbm)

airbnb <- read_csv('austin_listings.csv')

#step 1:

set.seed(20260731)

train_idx <- createDataPartition(
    airbnb$price,
    p = 0.80,
    list = FALSE
)

airbnb_train <- airbnb[train_idx, ]
airbnb_test <- airbnb[-train_idx, ]

nrow(airbnb_train)
nrow(airbnb_test)


#step 2:

raw_lm <- lm(
    price ~ . - id,
    data = airbnb_train
)

ggplot(airbnb_train, aes(x = price)) +
  geom_histogram(bins = 30) +
  labs(
    x = "Nightly price ($)",
    y = "Count"
  )

raw_diagnostics <- tibble(
    fitted = fitted(raw_lm),
    residual = residuals(raw_lm)
)

ggplot(raw_diagnostics, aes(x = fitted, y = residual)) +
  geom_point(alpha = 0.3) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(
    x = "Fitted nightly price",
    y = "Residual"
  )

airbnb_train <- airbnb_train |>
  mutate(log_price = log(price))

log_lm <- lm(
  log_price ~ . - id - price,
  data = airbnb_train
)

log_diagnostics <- tibble(
  fitted = fitted(log_lm),
  residual = residuals(log_lm)
)

ggplot(log_diagnostics, aes(x = fitted, y = residual)) +
  geom_point(alpha = 0.3) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(
    x = 'Fitted log(price)',
    y = 'Residual'
  )

diagnostic_data <- airbnb_train |>
  mutate(
    residual = residuals(log_lm)
  )

ggplot(diagnostic_data, aes(x = accommodates, y = residual)) +
  geom_point(alpha = 0.3) +
  geom_smooth() +
  labs(
    x = 'Accommodates',
    y = 'Residual'
  )

ggplot(diagnostic_data, aes(x = bedrooms, y = residual)) +
  geom_point(alpha = 0.3) +
  geom_smooth() +
  labs(
    x = "Bedrooms",
    y = "Residual"
  )

ggplot(diagnostic_data, aes(x = dist_center_km, y = residual)) +
  geom_point(alpha = 0.3) +
  geom_smooth() +
  labs(
    x = 'Distance from city center in km',
    y = 'Residual'
  )



#step 3:

linear_recipe <- recipe(
  log_price ~ .,
  data = airbnb_train
) |>
  update_role(id, price, new_role = 'ID') |>
  step_dummy(all_nominal_predictors()) |>
  step_poly(
    accommodates,
    bedrooms,
    dist_center_km,
    degree = 3
  ) |>
  step_zv(all_predictors())

regularized_recipe <- linear_recipe |>
  step_normalize(all_numeric_predictors())

tree_recipe <- recipe(
  log_price ~ .,
  data = airbnb_train
) |>
  update_role(id, price, new_role = 'ID') |>
  step_dummy(
    all_nominal_predictors(),
    one_hot = TRUE
  ) |>
  step_zv(all_predictors())




#step 4:

set.seed(20260731)

cv_holdouts <- createFolds(
  airbnb_train$log_price,
  k = 5,
  returnTrain = FALSE
)

cv_train <- vector('list', length(cv_holdouts))

for (fold in seq_along(cv_holdouts)) {
  cv_train[[fold]] <- setdiff(
    seq_len(nrow(airbnb_train)),
    cv_holdouts[[fold]]
  )
}

cv_control <- trainControl(
  method = 'cv',
  index = cv_train,
  indexOut = cv_holdouts
)

#R candidate settings

ridge_grid <- expand.grid(
alpha = 0,
lambda = exp(seq(log(0.000442), log(44.2), length.out = 40))
)
lasso_grid <- expand.grid(
alpha = 1,
lambda = exp(seq(log(0.0000442), log(0.0442), length.out = 40))
)
tree_grid <- expand.grid(
cp = c(
0.00006, 0.00015, 0.0004, 0.001,
0.0025, 0.006, 0.015, 0.04
)
)
forest_grid <- expand.grid(
mtry = c(5, 10, 15, 20, 30, 40, 52),
splitrule = "variance",
min.node.size = 5
)

boost_grid <- expand.grid(
n.trees = c(400, 700),
interaction.depth = c(4, 6, 8),
shrinkage = c(0.05, 0.10),
n.minobsinnode = 10
)

set.seed(1)

ols_fit <- train(
  linear_recipe,
  data = airbnb_train,
  method = 'lm',
  trControl = cv_control
)

set.seed(2)

ridge_fit <- train(
  regularized_recipe,
  data = airbnb_train,
  method = 'glmnet',
  tuneGrid = ridge_grid,
  trControl = cv_control
)

set.seed(3)

lasso_fit <- train(
  regularized_recipe,
  data = airbnb_train,
  method = 'glmnet',
  tuneGrid = lasso_grid,
  trControl = cv_control
)

set.seed(4)

tree_fit <- train(
  tree_recipe,
  data = airbnb_train,
  method = 'rpart',
  tuneGrid = tree_grid,
  trControl = cv_control
)

set.seed(5)

forest_fit <- train(
  tree_recipe,
  data = airbnb_train,
  method = 'ranger',
  num.trees = 500,
  tuneGrid = forest_grid,
  trControl = cv_control
)

set.seed(6)

boost_fit <- train(
  tree_recipe,
  data = airbnb_train,
  method = 'gbm',
  tuneGrid = boost_grid,
  trControl = cv_control,
  verbose = FALSE
)


ols_fit$results
ridge_fit$bestTune
lasso_fit$bestTune
tree_fit$bestTune
forest_fit$bestTune
boost_fit$bestTune

tree_fit$results |>
  select(cp, RMSE)

forest_fit$results |>
  select(mtry, RMSE)

ridge_fit$results |>
  select(lambda, RMSE)

lasso_fit$results |>
  select(lambda, RMSE)


forest_fit$results |>
  arrange(RMSE)

boost_fit$results |>
  arrange(RMSE)


forest_fit$results |>
  arrange(RMSE) |>
  select(mtry, splitrule, min.node.size, RMSE)

boost_fit$results |>
  arrange(RMSE) |>
  select(
    n.trees,
    interaction.depth,
    shrinkage,
    n.minobsinnode,
    RMSE
  )


#step 5:

airbnb_test <- airbnb_test |>
  mutate(log_price = log(price))

ols_test_pred <- predict(
  ols_fit,
  newdata = airbnb_test
)

ridge_test_pred <- predict(
  ridge_fit,
  newdata = airbnb_test
)

lasso_test_pred <- predict(
  lasso_fit,
  newdata = airbnb_test
)

tree_test_pred <- predict(
  tree_fit,
  newdata = airbnb_test
)

forest_test_pred <- predict(
  forest_fit,
  newdata = airbnb_test
)

boost_test_pred <- predict(
  boost_fit,
  newdata = airbnb_test
)

comparison <- tibble(
  model = c(
    'OLS',
    'Ridge',
    'Lasso',
    'Single tree',
    'Random forest',
    'Boosting'
  ),

  selected_settings = c(
    'none',
    paste0(
      'lambda = ',
      signif(ridge_fit$bestTune$lambda, 3)
    ),

    paste0(
      'lambda = ',
      signif(lasso_fit$bestTune$lambda, 3)
    ),

    paste0(
      'cp = ',
      signif(tree_fit$bestTune$cp, 3)
    ),

    paste0(
      'mtry = ',
      forest_fit$bestTune$mtry,
      ', min.node.size = ',
      forest_fit$bestTune$min.node.size
    ),

    paste0(
      'trees = ',
      boost_fit$bestTune$n.trees,
      ', depth = ',
      boost_fit$bestTune$interaction.depth,
      ', rate = ',
      boost_fit$bestTune$shrinkage
    )
  ),

  cv_rmse = c(
    getTrainPerf(ols_fit)$TrainRMSE,
    getTrainPerf(ridge_fit)$TrainRMSE,
    getTrainPerf(lasso_fit)$TrainRMSE,
    getTrainPerf(tree_fit)$TrainRMSE,
    getTrainPerf(forest_fit)$TrainRMSE,
    getTrainPerf(boost_fit)$TrainRMSE
  ),

  test_rmse = c(
    sqrt(mean(
      (airbnb_test$log_price - ols_test_pred)^2
    )),

    sqrt(mean(
      (airbnb_test$log_price - ridge_test_pred)^2
    )),

    sqrt(mean(
      (airbnb_test$log_price - lasso_test_pred)^2
    )),

    sqrt(mean(
      (airbnb_test$log_price - tree_test_pred)^2
    )),

    sqrt(mean(
      (airbnb_test$log_price - forest_test_pred)^2
    )),

    sqrt(mean(
      (airbnb_test$log_price - boost_test_pred)^2
    ))
  )
)

comparison
