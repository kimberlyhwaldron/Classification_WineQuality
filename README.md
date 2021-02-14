---
# Classification Modeling <br> Wine Quality Data <br> Kimberly Healy  |  healy.kim@gmx.us 

## Background
The data ([WineQuality.csv](https://archive.ics.uci.edu/ml/datasets/wine+quality)) includes the results of physicochemical and sensory tests for different types of wine. The attributes from the physicochemical tests include: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, wine type. The attribute from the sensory test includes: quality, a score between 0 and 10.   
    
The purpose of this investigation is to find the optimal classification model to predict wine quality.  The variables in the raw data set include:   
   
Variable       |   Type      
---------------|-------------
fixed.acidity    |    numeric 
volatile.acidity          |    numeric  
citric.acid  |    numeric  
residual.sugar |    numeric  
chlorides    |    numeric  
free.sulfur.dioxide     |    numeric     
total.sulfer.dioxide   |    numeric   
density   |    numeric
pH    |    numeric  
sulphates    |    numeric  
alcohol    |    numeric (%)  
winetype   |    string, red or white  
quality   |    integer, ordinal rating between 0 and 10  
    
The following predictive models are used for a categorical output:        
**Model 1: Ordered Logistic Regression**: A regression model with an ordinal dependent variable.   
**Model 2: CART (tree)**: Classification & regression tree that takes a either categorical or continuous response variable. At an inner node, it allows a binary split.    
**Model 3: C5.0 (tree)**: Classification tree that takes only a categorical response variable. At an inner node, it allows a multiple split. Each node is split using the best split by the Gini Index among all input variables.   
**Model 4: Random Forest (tree)**: Combines multiple models (i.e. hundreds of decision trees) into a single ensemble. Each decision tree is independent from another tree. Each node is split using the best split among a random subset of all input predictor variables chosen at the node.   

         
         
The following questions are answered:     
  - **Which attributes best predict wine quality?**   
  - **Which model(s) best predicts wine quality?**   
  
  
 ***
## I. Load required libraries 
```{r lib, message = FALSE}
library(tidyverse)
library(rpart)
library(caret)
library(e1071)
library(C50)
library(randomForest)
```
***
## II. Load raw data from csv file
``` {r load}
wine_raw <- read.csv("WineQuality.csv", sep=",")
wine <- wine_raw
```

***

## III. Define Useful Functions
### (i) Significance Tests 
 $H_{o}: \rho = 0$   
 $H_{a}: \rho ≠ 0$

```{r ada}
# correlation test  
cor.mtest <- function(mat, ...) {
    mat <- as.matrix(mat)
    n <- ncol(mat)
    p.mat<- matrix(NA, n, n)
    diag(p.mat) <- 0
    for (i in 1:(n - 1)) {
        for (j in (i + 1):n) {
            tmp <- cor.test(mat[, i], mat[, j], ...)
            p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
        }
    }
    colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
    p.mat
}
```

***

## IV. Data Preprocessing & Exploration

### (i) Preprocessing {.tabset}
    
#### Summary
The data is pretty clean in its raw state.  
```{r aa}
dim(wine)
summary(wine)
str(wine)
length(which(is.na(wine)))

# change to appropriate data types
wine$free.sulfur.dioxide<-as.numeric(wine$free.sulfur.dioxide)
wine$total.sulfur.dioxide<-as.numeric(wine$total.sulfur.dioxide)
wine$quality<-as.factor(wine$quality)

# convert winetype (red, white) attribute to numeric binary
wine$winetype<-ifelse(wine$winetype == "red", 0, 1)
```
***
#### Create wine.bin
Bin wine quality ratings for classification models
```{r skladaddd}
wine$quality.bin<-as.character(wine$quality)
wine$quality.bin[wine$quality.bin %in% c("0","1","2","3","4","5")] <- "OK"
wine$quality.bin[wine$quality.bin %in% c("6","7")] <- "Above Average"
wine$quality.bin[wine$quality.bin %in% c("8", "9", "10")] <- "Superior"
wine$quality.bin<-as.factor(wine$quality.bin)
```

***
### (ii) Exploration {.tabset}
    
#### Distributions
The data does not include any wines with quality rating 1, 2, or 10.    
Wine quality is approx. normally distributed.   
```{r skladdd}
table(wine$quality)
ggplot(wine, aes(factor(quality), fill = factor(quality.bin))) + 
  labs(title = "Quality Ratings Binned", x = "quality rating") + geom_bar()
```

Distributions of numeric variables.   
```{r dklsa, message=FALSE, error=FALSE}
wine %>%
  purrr::keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
    facet_wrap(~ key, scales = "free") +
    geom_histogram()
```








***
#### Correlations
Correlations between numerical variables.   
Notable correlations are between:   
- free.sulfur.dioxide and total.sulfur.dioxide (r = 0.721)  
- density and alcohol (r = -0.6867)  
- volatile.acidity and winetype (r = -0.653)   
- quality is correlated the most with alcohol (r = 0.444), although moderately so   
- quality and pH level correlation is statistically insignificant (p-val = 0.116)   
```{r dfkall}
# calculate p-values of correlations
p.mat <- cor.mtest(select_if(wine, is.numeric))

# plot correlations. X's indicate statistically insignificant correlation
corrplot::corrplot(cor(select_if(wine, is.numeric)), 
                   method="number",
                   type="upper", order="hclust", tl.col="black", tl.srt=45,
                   p.mat = p.mat, sig.level = 0.05
)

```

***

#### Chi Square
Determine the relationship between the two categorical variables, winetype and quality.    
   
p-val < 0.05, there is enough evidence to reject $H_{o}:$, the variables are dependent 
```{r lksaklda}
chi.test <- chisq.test(table(wine$winetype, wine$quality))
chi.test
```



***  

## V. Partition the data into training and testing sets
Training set = 70% data   
Testing set = 30% data   
```{r bf}
samplesize = 0.70*nrow(wine)
set.seed(100)

index = sample(seq_len(nrow(wine)), size = samplesize)
wine_train = wine[index,]
wine_test = wine[-index,]
```

***  
## VI. Modelling

### (i) Ordered Logistic Regression  {.tabset}

#### Model
All attributes are included in the model. P-Values are calculated and included in the summary.   
  
- The p-values show that all inputs into the model are significant in predicting wine quality.      
- **Residual Deviance**: 6288.248   
- **AIC**: 6316.248  
    
The coefficients are interpreted in log odds.  
- Positive log odds = more likely to be in a lower bin  
- Negative log odds = more likely to be in a higher bin  
   
| NOTE: Comparison of coefficients would be more efficient if the input variables are 
| scaled over the same interval. I did not scale the input variables, so each
| coefficient in this model should be interpreted in the context of its units and 
| distribution shown in (ii) Exploration: Distributions of numeric variables.   
   
- **Coefficient interpretation example 1**: for a one unit increase in volatile.acidity, we expect the expected value of the log odds of quality.bin to increase by 3.98 (log odds of the wine being of lesser or equal quality than the current quality bin), given that all of the other variables in the model remain constant.   
- **Coefficient interpretation example 2**: for a one unit increase in fixed.acidity, we expect the expected value of the log odds of quality.bin to decrease by 0.057 (log odds of the wine being of lesser or equal quality than the current quality bin), given that all of the other variables in the model remain constant.   

```{r fd}
# model to predict quality bin from all other factors
wine_logit = MASS::polr(quality.bin~.,data=wine_train[,-(ncol(wine_train)-1)], Hess=TRUE)
summary(wine_logit)
# p-values for variables
ctable <- coef(summary(wine_logit))
 p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
(ctable <- cbind(ctable, "p value" = p))
```
***

#### Evaluation
The confusion matrix shows the performance of the ordered logistic regression model. The numbers show how many times the quality bin was identified correctly in the test data set.  
- **Misclassification Error**: 29.44%  
- **Odds Ratio (OR) interpretation 1**: For every one unit increase in citric.acid,  the odds of being in a lower quality.bin is increased by 84.8% (multiplied 1.848 times), given that all of the other variables in the model remain constant.     
- **Odds Ratio (OR) interpretation 2**: For every one unit increase in pH,  the odds of being in a lower quality.bin is multiplied 0.5428 times, given that all of the other variables in the model remain constant.   
```{r 2e}
# confusion matrix
predict_logit<-predict(wine_logit, wine_test, type="class")
confusionMatrix(wine_test$quality.bin, predict_logit)
 
# misclassification error
mean(as.character(wine_test$quality.bin) != as.character(predict_logit))

# convert the coefficients into odds ratios (OR)
# add the confidence intervals for parameter estimates
exp(cbind(OR = coef(wine_logit), confint.default(wine_logit)))

```


***

### (ii) CART  {.tabset}

#### Model
All attributes are inputted in the model. The initial model with standard parameter cp=0.001 used all quantitative predictors in tree construction. The pruned model changes the parameter to cp=0.002024664 as it results in the lowest error.      
   
Alcohol and volatile.acidity are important node splitters.    
```{r eed}
# model to predict quality bin from all other factors. start with cp = 0.001 
wine_CART<-rpart(quality.bin~.,data=wine_train[,-(ncol(wine_train)-1)],method="class", control=rpart.control(cp=0.001)) 
printcp(wine_CART) # display results
plotcp(wine_CART) # visualize cross validation results
bestcp <- wine_CART$cptable[which.min(wine_CART$cptable[,"xerror"]),"CP"] 
bestcp # print best cp based on model
```

Prune tree to avoid minimize overfitting 
```{r ldksldas}
# prune tree using bestcp from above
wine_CART_pruned <- prune(wine_CART, cp = bestcp)
printcp(wine_CART_pruned) # display results
plot(wine_CART_pruned) # visualize cross validation results
text(wine_CART_pruned, use.n = TRUE, xpd = TRUE)
```
***

#### Evaluation
The confusion matrix shows the performance of the CART model. The numbers show how many times the quality bin was identified correctly in the test data set.  
- **Misclassification Error**: 28.26%  
 
```{r se}
# confusion matrix
predict_CART<-predict(wine_CART_pruned, wine_test, type="class")
confusionMatrix(predict_CART,wine_test$quality.bin)
 
# misclassification error
mean(as.character(wine_test$quality.bin) != as.character(predict_CART))
```

***

### (iii) C5.0  {.tabset}

#### Model
All attributes are included in the model.    
Like the CART model, **alcohol** and **volatile.acidity** are important node splitters, with 100% use in the model. This means those two variables give the greatest loss of model prediction performance if omitted.  
  
The next important variables are free.sulfur.dioxide and sulphates, with 79.59% and 64.37% use in the model respectively. The least important varaibles are density (9.57%), total.sulfur.dioxide (7.59%), and winetype (3.78%).   

```{r dr}
# model to predict quality bin from all other factors
wine_C5<-C5.0(quality.bin~., data=wine_train[,-(ncol(wine_train)-1)]) #
summary(wine_C5)
plot(wine_C5, main = "C5.0 Classification Tree for Quality Bin")
```

***
#### Evaluation
The confusion matrix shows the performance of the C5.0 model. The numbers show how many times the quality bin was identified correctly in the test data set.  
- **Misclassification Error**: 28.05%  

```{r re}
# confusion matrix
predict_C5 <- predict(wine_C5,wine_test, type="class")
confusionMatrix(wine_test$quality.bin, predict_C5)
 
# misclassification error
mean(as.character(wine_test$quality.bin) != as.character(predict_C5))
```
***

### (iv) Random Forest  {.tabset}


#### Model
The goal is to minimize the complexity of the random forest (i.e. fewest trees in the model) and minimize the OOB estimate of error rate.   
   
ntree | OOB estimate of error rate   
------|---------------------------   
10 | 26%   
25 | 23.03%  
45 | 20.8%  
50 | 20.45%  
75 | 19.99%  
100 | 20.21%  
200 | 19.4%  
500 | 19.64%  
  
The differences in error rate are negotiable after 45 trees.  
  
Like the two tree models above, **alcohol** and **volatile.acidity** are the most important variables in the model. According to MeanDecreaseAccuracy, residual.sugar, total.sulfur.dioxide, and sulphates are also important to the model.  
  
The model has a particular hard time predicting "Superior" quality.bin. The classification error for the bin is 69.12%, meaning the model predicted "OK" or "Above Average" 69.12% of the time, when in reality the wine is "Superior".   

```{r d33}
# model to predict quality bin from all other factors
wine_rf<-randomForest(quality.bin~., data=wine_train[,-(ncol(wine_train)-1)], ntree=100, importance=TRUE)
wine_rf
varImpPlot(wine_rf)
```

***
#### Evaluation 
The confusion matrix shows the performance of the random forest model when ntree=45. The numbers show how many times the quality bin was identified correctly in the test data set.  
- **Misclassification Error**: 20% 
  
```{r e3d3}
# confusion matrix
predict_rf<-predict(wine_rf,wine_test)
confusionMatrix(predict_rf,wine_test$quality.bin)

# misclassification error
mean(as.character(wine_test$quality.bin) != as.character(predict_rf))
```

***

## VII. Conclusion & Recommendations  
  
We evaluated four models to predict the quality of wine ("OK", "Above Average" and "Superior" groupings) using 12 input variables. The accuracy of the models are shown below:  
  
Model | Accuracy (1-Misclassification Error)  
------|---------------------------------  
Ordered Logistic Regression | 70.56%  
CART | 71.74%  
C5.0 | 71.95%  
Random Forest | 80%   
  
   
**Based on our model selections, the best model to predict wine quality is Random Forest with 80% accuracy.**   
  
This investigation could be improved in the following ways:  
**(R1)** Bin the wine quality ratings into two or four more equal groups so that the differences between the groups are not as drastic. The group Superior is decreasing the model's accuracy in this report.       
```{r kdlsjla}
table(wine$quality.bin)
```
**(R2)** Bin the wine quality rating into two groups to utilize an ROC curve for performance.   
**(R3)** Consider other model types like AdaBoost and SVM.    
**(R4)** Consider doing a deeper analysis into decreasing the amount of predictor inputs. From this report, alcohol and volatile.acidity should be retained. However, winetype is a candidate to be cut from the predictors.  Consider the correlations between the variables. During the exploration, free.sulfur.dioxide and total.sulfur.dioxide are moderately/highly correlated (r = 0.721). Perhaps our model would have been better if we left one "sulfur.dioxide" out of the analysis.   
    
       
       
        



