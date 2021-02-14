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

