# Working with Real Data

# Look at the Big Picture

这里以预测房价为例

## Frame the Problem

明确模型的输出数据内容及用途
是一个监督学习，回归任务，单变量回归，batch learning

## Select a Performance Measure

对于回归问题，典型的衡量标准是Root Mean Square Error (RMSE)，它给出了预测会包含多少误差（对于大误差有更大的权重）
![EQU 2-1](PIC/EQU_2_1.png)

尽管RMSE很常用，也可以选择其他方法来评价。比如有离群点较多，可能会选用MAE(mean absolute error)

![EQU 2-2](PIC/EQU_2_2.png)

> The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.

## Check the Assumptions


# Get the Data


## Create a Test Set

[Refs.  ](EXAMPLE/2-1.py "")

## Discover and Visualize the Data to Gain Insights

### Visualizing Geographical Data
```python
print(housing.head())
```

>           
            longitude  latitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  median_house_value ocean_proximity
        0    -122.23     37.88                41.0        880.0           129.0       322.0       126.0         8.3252            452600.0        NEAR BAY
        1    -122.22     37.86                21.0       7099.0          1106.0      2401.0      1138.0         8.3014            358500.0        NEAR BAY
        2    -122.24     37.85                52.0       1467.0           190.0       496.0       177.0         7.2574            352100.0        NEAR BAY
        3    -122.25     37.85                52.0       1274.0           235.0       558.0       219.0         5.6431            341300.0        NEAR BAY
        4    -122.25     37.85                52.0       1627.0           280.0       565.0       259.0         3.8462            342200.0        NEAR BAY

```python
print(housing.info())
```
>   
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
    longitude             20640 non-null float64
    latitude              20640 non-null float64
    housing_median_age    20640 non-null float64
    total_rooms           20640 non-null float64
    total_bedrooms        20433 non-null float64
    population            20640 non-null float64
    households            20640 non-null float64
    median_income         20640 non-null float64
    median_house_value    20640 non-null float64
    ocean_proximity       20640 non-null object
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB
    None

```python
print(housing["ocean_proximity"].value_counts())
```
>
    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64


```python
print(housing.describe())
```
>
             longitude      latitude  housing_median_age   total_rooms  total_bedrooms    population    households  median_income  median_house_value
    count  20640.000000  20640.000000        20640.000000  20640.000000    20433.000000  20640.000000  20640.000000   20640.000000        20640.000000
    mean    -119.569704     35.631861           28.639486   2635.763081      537.870553   1425.476744    499.539680       3.870671       206855.816909
    std        2.003532      2.135952           12.585558   2181.615252      421.385070   1132.462122    382.329753       1.899822       115395.615874
    min     -124.350000     32.540000            1.000000      2.000000        1.000000      3.000000      1.000000       0.499900        14999.000000
    25%     -121.800000     33.930000           18.000000   1447.750000      296.000000    787.000000    280.000000       2.563400       119600.000000
    50%     -118.490000     34.260000           29.000000   2127.000000      435.000000   1166.000000    409.000000       3.534800       179700.000000
    75%     -118.010000     37.710000           37.000000   3148.000000      647.000000   1725.000000    605.000000       4.743250       264725.000000
    max     -114.310000     41.950000           52.000000  39320.000000     6445.000000  35682.000000   6082.000000      15.000100       500001.000000


```python
housing.hist(bins=50, figsize=(20,15))
plt.show()
```
![attribute_histogram_plots](images/end_to_end_project/attribute_histogram_plots.png)
    
```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
```

![housing_prices_scatterplot](images/end_to_end_project/housing_prices_scatterplot.png)

### Looking for Correlations

```python
print("corr_matrix = housing.corr()")
corr_matrix = housing.corr()
print(corr_matrix)
```
>
                         longitude  latitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  median_house_value
    longitude            1.000000 -0.924478           -0.105848     0.048871        0.076598    0.108030    0.063070      -0.019583           -0.047432
    latitude            -0.924478  1.000000            0.005766    -0.039184       -0.072419   -0.115222   -0.077647      -0.075205           -0.142724
    housing_median_age  -0.105848  0.005766            1.000000    -0.364509       -0.325047   -0.298710   -0.306428      -0.111360            0.114110
    total_rooms          0.048871 -0.039184           -0.364509     1.000000        0.929379    0.855109    0.918392       0.200087            0.135097
    total_bedrooms       0.076598 -0.072419           -0.325047     0.929379        1.000000    0.876320    0.980170      -0.009740            0.047689
    population           0.108030 -0.115222           -0.298710     0.855109        0.876320    1.000000    0.904637       0.002380           -0.026920
    households           0.063070 -0.077647           -0.306428     0.918392        0.980170    0.904637    1.000000       0.010781            0.064506
    median_income       -0.019583 -0.075205           -0.111360     0.200087       -0.009740    0.002380    0.010781       1.000000            0.687160
    median_house_value  -0.047432 -0.142724            0.114110     0.135097        0.047689   -0.026920    0.064506       0.687160            1.000000

```python
print(corr_matrix["median_house_value"].sort_values(ascending=False))
```
>
    median_house_value    1.000000
    median_income         0.687160
    total_rooms           0.135097
    housing_median_age    0.114110
    households            0.064506
    total_bedrooms        0.047689
    population           -0.026920
    longitude            -0.047432
    latitude             -0.142724

```python
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```
![scatter_matrix_4_attributes](images/end_to_end_project/scatter_matrix_4_attributes.png)

```python
housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)
```
![scatter_median_income_median_house_value](images/end_to_end_project/scatter_median_income_median_house_value.png)

### Experimenting with Attribute Combinations
```python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
```
>
    median_house_value          1.000000
    median_income               0.687160
    rooms_per_household         0.146285
    total_rooms                 0.135097
    housing_median_age          0.114110
    households                  0.064506
    total_bedrooms              0.047689
    population_per_household   -0.021985
    population                 -0.026920
    longitude                  -0.047432
    latitude                   -0.142724
    bedrooms_per_room          -0.259984
    Name: median_house_value, dtype: float64


## Prepare the Data for Machine Learning Algorithms

```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```

### Data Cleaning
某些数据特征缺失时，一般使用以下三种方法进行处理：
1. 丢弃该条数据
2. 丢弃该特征
3. 补充该特征数据（0，平均数，中位数等）

分别对应以下实现：
>
    housing.dropna(subset=["total_bedrooms"]) # option 1
    housing.drop("total_bedrooms", axis=1) # option 2
    median = housing["total_bedrooms"].median() # option 3
    housing["total_bedrooms"].fillna(median, inplace=True)

Scikit-Learn提供了一个方法来解决这个问题

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
```

Since the median can only be computed on numerical attributes, you need to create a
copy of the data without the text attribute ocean_proximity:
```python
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
```

```python
print(imputer.statistics_)
# array([ -118.51 , 34.26 , 29. , 2119.5 , 433. , 1164. , 408. , 3.5409])
```

```python
housing_num.median().values
# array([ -118.51 , 34.26 , 29. , 2119.5 , 433. , 1164. , 408. , 3.5409])
```
```python
X = imputer.transform(housing_num)

```

The result is a plain NumPy array containing the transformed features. If you want to put it back into a pandas DataFrame, it’s simple:
```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
```

#### Scikit-Learn Design
>
    <center>Scikit-Learn Design</center>
    Scikit-Learn’s API is remarkably well designed. These are the main design principles:

    Consistency
        All objects share a consistent and simple interface:

        Estimators
            Any object that can estimate some parameters based on a dataset is called an
            estimator (e.g., an imputer is an estimator). The estimation itself is per‐
            formed by the fit() method, and it takes only a dataset as a parameter (or
            two for supervised learning algorithms; the second dataset contains the
            labels). Any other parameter needed to guide the estimation process is con‐
            sidered a hyperparameter (such as an imputer’s strategy), and it must be
            set as an instance variable (generally via a constructor parameter).

        Transformers
            Some estimators (such as an imputer) can also transform a dataset; these are
            called transformers. Once again, the API is simple: the transformation is
            performed by the transform() method with the dataset to transform as a
            parameter. It returns the transformed dataset. This transformation generally
            relies on the learned parameters, as is the case for an imputer. All transform‐
            ers also have a convenience method called fit_transform() that is equiva‐
            lent to calling fit() and then transform() (but sometimes
            fit_transform() is optimized and runs much faster).

        Predictors
            Finally, some estimators, given a dataset, are capable of making predictions;
            they are called predictors. For example, the LinearRegression model in the
            previous chapter was a predictor: given a country’s GDP per capita, it pre‐
            dicted life satisfaction. A predictor has a predict() method that takes a
            dataset of new instances and returns a dataset of corresponding predictions.
            It also has a score() method that measures the quality of the predictions,
            given a test set (and the corresponding labels, in the case of supervised learn‐
            ing algorithms).

    Inspection
        All the estimator’s hyperparameters are accessible directly via public instance
        variables (e.g., imputer.strategy), and all the estimator’s learned parameters are
        accessible via public instance variables with an underscore suffix (e.g.,
        imputer.statistics_)            

    Nonproliferation of classes
        Datasets are represented as NumPy arrays or SciPy sparse matrices, instead of
        homemade classes. Hyperparameters are just regular Python strings or numbers.
    Composition
        Existing building blocks are reused as much as possible. For example, it is easy to
        create a Pipeline estimator from an arbitrary sequence of transformers followed
        by a final estimator, as we will see.
    Sensible defaults
        Scikit-Learn provides reasonable default values for most parameters, making it
        easy to quickly create a baseline working system

### Handling Text and Categorical Attributes
        