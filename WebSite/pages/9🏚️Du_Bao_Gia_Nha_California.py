import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
st.title("Dự báo giá nhà California")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    housing = pd.read_csv(uploaded_file)
    def PhanNhomMedianIncome(housing):
            st.header("PhanNhomMedianIncome")
            st.write(housing)
            housing["income_cat"] = pd.cut(housing["median_income"],
            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
            labels=[1, 2, 3, 4, 5])
            housing["income_cat"].hist()
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(housing.info())
            st.download_button("Download new housing.csv",housing.to_csv() , file_name="housing_new.csv")
    def Decision_Tree_Regressor(housing):
        st.header("DecisionTreeRegressor")
        col1, col2 = st.columns(2)
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
            def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
                self.add_bedrooms_per_room = add_bedrooms_per_room
            def fit(self, X, y=None):
                return self # nothing else to do
            def transform(self, X, y=None):
                rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                population_per_household = X[:, population_ix] / X[:, households_ix]
                if self.add_bedrooms_per_room:
                    bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                    return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
                else:
                    return np.c_[X, rooms_per_household, population_per_household]
        def display_scores(scores):
            st.markdown("#### - Mean: %.2f" % (scores.mean()))
            st.markdown("#### - Standard deviation: %.2f" % (scores.std()))
        # Them column income_cat dung de chia data
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]
        # Chia xong thi delete column income_cat
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()
        housing_num = housing.drop("ocean_proximity", axis=1)
        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])
        housing_prepared = full_pipeline.fit_transform(housing)
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(housing_prepared, housing_labels)
        # Prediction
        some_data = housing.iloc[:5]
        some_labels = housing_labels.iloc[:5]
        some_data_prepared = full_pipeline.transform(some_data)
        # Prediction 5 samples 
        with col1:
            st.write('Predictions:  ' ,tree_reg.predict(some_data_prepared).T)
            st.write('Labels: ' ,list(some_labels))
            st.write('\n')
        with col2:
            # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
            housing_predictions = tree_reg.predict(housing_prepared)
            mse_train = mean_squared_error(housing_labels, housing_predictions)
            rmse_train = np.sqrt(mse_train)
            st.markdown('### 1.Sai so binh phuong trung binh - train: %.2f' % rmse_train )
            # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
            scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
            st.markdown('### 2.Sai so binh phuong trung binh - cross-validation:')
            rmse_cross_validation = np.sqrt(-scores)
            display_scores(rmse_cross_validation)
            # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
            X_test = strat_test_set.drop("median_house_value", axis=1)
            y_test = strat_test_set["median_house_value"].copy()
            X_test_prepared = full_pipeline.transform(X_test)
            y_predictions = tree_reg.predict(X_test_prepared)
            mse_test = mean_squared_error(y_test, y_predictions)
            rmse_test = np.sqrt(mse_test)
            st.markdown('### 3.Sai so binh phuong trung binh - test:  %.2f' % rmse_test)
    def Linear_Regression(housing):
            st.header('Linear regression')
            col1, col2 = st.columns(2)
            rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
            class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
                def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
                    self.add_bedrooms_per_room = add_bedrooms_per_room
                def fit(self, X, y=None):
                    return self # nothing else to do
                def transform(self, X, y=None):
                    rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                    population_per_household = X[:, population_ix] / X[:, households_ix]
                    if self.add_bedrooms_per_room:
                        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
                    else:
                        return np.c_[X, rooms_per_household, population_per_household]


            def display_scores(scores):
                
                st.markdown("#### - Mean: %.2f" % (scores.mean()))
                st.markdown("#### - Standard deviation: %.2f" % (scores.std()))
                # Them column income_cat dung de chia data
            housing["income_cat"] = pd.cut(housing["median_income"],
                                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                        labels=[1, 2, 3, 4, 5])
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in split.split(housing, housing["income_cat"]):
                strat_train_set = housing.loc[train_index]
                strat_test_set = housing.loc[test_index]
            # Chia xong thi delete column income_cat
            for set_ in (strat_train_set, strat_test_set):
                set_.drop("income_cat", axis=1, inplace=True)
            housing = strat_train_set.drop("median_house_value", axis=1)
            housing_labels = strat_train_set["median_house_value"].copy()
            housing_num = housing.drop("ocean_proximity", axis=1)
            num_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy="median")),
                    ('attribs_adder', CombinedAttributesAdder()),
                    ('std_scaler', StandardScaler()),
                ])
            num_attribs = list(housing_num)
            cat_attribs = ["ocean_proximity"]
            full_pipeline = ColumnTransformer([
                    ("num", num_pipeline, num_attribs),
                    ("cat", OneHotEncoder(), cat_attribs),
                ])
            housing_prepared = full_pipeline.fit_transform(housing)
            # Training
            lin_reg = LinearRegression()
            lin_reg.fit(housing_prepared, housing_labels)
            # Prediction
            some_data = housing.iloc[:5]
            some_labels = housing_labels.iloc[:5]
            some_data_prepared = full_pipeline.transform(some_data)
            # Prediction 5 samples 
            with col1:
                st.write("Predictions:", lin_reg.predict(some_data_prepared))
                st.write("Labels:", list(some_labels))
            # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
            housing_predictions = lin_reg.predict(housing_prepared)
            mse_train = mean_squared_error(housing_labels, housing_predictions)
            with col2:
                rmse_train = np.sqrt(mse_train)
                st.markdown('### 1.Sai so binh phuong trung binh - train: %.2f' % rmse_train)
                # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
                scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
                st.markdown('### 2.Sai so binh phuong trung binh - cross-validation:')
                rmse_cross_validation = np.sqrt(-scores)
                display_scores(rmse_cross_validation)
                # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
                X_test = strat_test_set.drop("median_house_value", axis=1)
                y_test = strat_test_set["median_house_value"].copy()
                X_test_prepared = full_pipeline.transform(X_test)
                y_predictions = lin_reg.predict(X_test_prepared)
                mse_test = mean_squared_error(y_test, y_predictions)
                rmse_test = np.sqrt(mse_test)
                st.markdown('### 3.Sai so binh phuong trung binh - test:%.2f' % rmse_test)
    def Random_Forest_Regression_Grid_Search_CV(housing):
        st.header('Random Forest Regression Grid Search CV')
        col1, col2 = st.columns(2)
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
            def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
                self.add_bedrooms_per_room = add_bedrooms_per_room
            def fit(self, X, y=None):
                return self # nothing else to do
            def transform(self, X, y=None):
                rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                population_per_household = X[:, population_ix] / X[:, households_ix]
                if self.add_bedrooms_per_room:
                    bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                    return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
                else:
                    return np.c_[X, rooms_per_household, population_per_household]

        def display_scores(scores):
            st.markdown("#### -Mean: %.2f" % (scores.mean()))
            st.markdown("#### -Standard deviation: %.2f" % (scores.std()))
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # Chia xong thi delete column income_cat
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        housing_num = housing.drop("ocean_proximity", axis=1)

        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])

        housing_prepared = full_pipeline.fit_transform(housing)

        param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
                    ]
        # Training
        forest_reg = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                                scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(housing_prepared, housing_labels)

        final_model = grid_search.best_estimator_

        # Prediction
        some_data = housing.iloc[:5]
        some_labels = housing_labels.iloc[:5]
        some_data_prepared = full_pipeline.transform(some_data)
        # Prediction 5 samples 
        with col1:
            st.write("Predictions:", final_model.predict(some_data_prepared))
            st.write("Labels:", list(some_labels))
        with col2:
        # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
            housing_predictions = final_model.predict(housing_prepared)
            mse_train = mean_squared_error(housing_labels, housing_predictions)
            rmse_train = np.sqrt(mse_train)
            st.markdown('### 1.Sai so binh phuong trung binh - train:%.2f' % rmse_train)


            # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
            scores = cross_val_score(final_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

            st.markdown('### 2.Sai so binh phuong trung binh - cross-validation:')
            rmse_cross_validation = np.sqrt(-scores)
            display_scores(rmse_cross_validation)

            # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
            X_test = strat_test_set.drop("median_house_value", axis=1)
            y_test = strat_test_set["median_house_value"].copy()
            X_test_prepared = full_pipeline.transform(X_test)
            y_predictions = final_model.predict(X_test_prepared)

            mse_test = mean_squared_error(y_test, y_predictions)
            rmse_test = np.sqrt(mse_test)
            st.markdown('### 3.Sai so binh phuong trung binh - test:%.2f' % rmse_test)
    def Random_Forest_Regression_Random_Search_CV(housing):
        st.header('Random_Forest_Regression_Random_Search_CV')
        col1, col2 = st.columns(2)
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
            def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
                self.add_bedrooms_per_room = add_bedrooms_per_room
            def fit(self, X, y=None):
                return self # nothing else to do
            def transform(self, X, y=None):
                rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                population_per_household = X[:, population_ix] / X[:, households_ix]
                if self.add_bedrooms_per_room:
                    bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                    return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
                else:
                    return np.c_[X, rooms_per_household, population_per_household]

        def display_scores(scores):
            st.markdown("#### -Mean: %.2f" % (scores.mean()))
            st.markdown("#### -Standard deviation: %.2f" % (scores.std()))
        # Them column income_cat dung de chia data
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # Chia xong thi delete column income_cat
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        housing_num = housing.drop("ocean_proximity", axis=1)

        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])

        housing_prepared = full_pipeline.fit_transform(housing)

        param_distribs = {
                'n_estimators': randint(low=1, high=200),
                'max_features': randint(low=1, high=8),
            }

        # Training
        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                        n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
        rnd_search.fit(housing_prepared, housing_labels)

        final_model = rnd_search.best_estimator_

        # Prediction
        some_data = housing.iloc[:5]
        some_labels = housing_labels.iloc[:5]
        some_data_prepared = full_pipeline.transform(some_data)
        # Prediction 5 samples 
        with col1:
            st.write("Predictions: %d", final_model.predict(some_data_prepared))
            st.write("Labels: %d", list(some_labels))
        with col2:
        # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
            housing_predictions = final_model.predict(housing_prepared)
            mse_train = mean_squared_error(housing_labels, housing_predictions)
            rmse_train = np.sqrt(mse_train)
            st.markdown('### 1.Sai so binh phuong trung binh - train:%.2f' % rmse_train)

        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
            scores = cross_val_score(final_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

            st.markdown('### 2.Sai so binh phuong trung binh - cross-validation:')
            rmse_cross_validation = np.sqrt(-scores)
            display_scores(rmse_cross_validation)

            # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
            X_test = strat_test_set.drop("median_house_value", axis=1)
            y_test = strat_test_set["median_house_value"].copy()
            X_test_prepared = full_pipeline.transform(X_test)
            y_predictions = final_model.predict(X_test_prepared)

            mse_test = mean_squared_error(y_test, y_predictions)
            rmse_test = np.sqrt(mse_test)
            st.markdown('### 3.Sai so binh phuong trung binh - test:%.2f' % rmse_test)

    def Random_Forest_Regression(housing):
        st.header('Regression Forest Regression')
        col1, col2 = st.columns(2)
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
            def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
                self.add_bedrooms_per_room = add_bedrooms_per_room
            def fit(self, X, y=None):
                return self # nothing else to do
            def transform(self, X, y=None):
                rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                population_per_household = X[:, population_ix] / X[:, households_ix]
                if self.add_bedrooms_per_room:
                    bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                    return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
                else:
                    return np.c_[X, rooms_per_household, population_per_household]
        def display_scores(scores):
            st.markdown("#### -Mean: %.2f" % (scores.mean()))
            st.markdown("#### -Standard deviation: %.2f" % (scores.std()))
        housing["income_cat"] = pd.cut(housing["median_income"],
                                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                        labels=[1, 2, 3, 4, 5])
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
                strat_train_set = housing.loc[train_index]
                strat_test_set = housing.loc[test_index]

            # Chia xong thi delete column income_cat
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()
        housing_num = housing.drop("ocean_proximity", axis=1)
        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])
        housing_prepared = full_pipeline.fit_transform(housing)
        # Training
        forest_reg = RandomForestRegressor()
        forest_reg.fit(housing_prepared, housing_labels)
        # Prediction
        some_data = housing.iloc[:5]
        some_labels = housing_labels.iloc[:5]
        some_data_prepared = full_pipeline.transform(some_data)
        # Prediction 5 samples 
        with col1:
            st.write("Predictions:", forest_reg.predict(some_data_prepared))
            st.write("Labels:", list(some_labels))
        with col2:
        # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
            housing_predictions = forest_reg.predict(housing_prepared)
            mse_train = mean_squared_error(housing_labels, housing_predictions)
            rmse_train = np.sqrt(mse_train)
            st.markdown('### 1.Sai so binh phuong trung binh - train:%.2f' % rmse_train)
            # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
            scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
            st.markdown('### 2.Sai so binh phuong trung binh - cross-validation:')
            rmse_cross_validation = np.sqrt(-scores)
            display_scores(rmse_cross_validation)
            # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
            X_test = strat_test_set.drop("median_house_value", axis=1)
            y_test = strat_test_set["median_house_value"].copy()
            X_test_prepared = full_pipeline.transform(X_test)
            y_predictions = forest_reg.predict(X_test_prepared)
            mse_test = mean_squared_error(y_test, y_predictions)
            rmse_test = np.sqrt(mse_test)
            st.markdown('### 3.Sai so binh phuong trung binh - test:%.2f' % rmse_test)

    page = st.sidebar.selectbox('Select page',['PhannhomMedianIncome','DecisionTreeRegressor','LinearRegression','RandomForestRegressionGridSearchCV','RandomForestRegressionRandomSearchCV', 'RandomForestRegression',]) 
    if page == 'DecisionTreeRegressor':
        Decision_Tree_Regressor(housing)
    elif page == 'LinearRegression':
        Linear_Regression(housing)
    elif page == 'RandomForestRegressionGridSearchCV':
        Random_Forest_Regression_Grid_Search_CV(housing)
    elif page == 'RandomForestRegressionRandomSearchCV':
        Random_Forest_Regression_Random_Search_CV(housing)
    elif page == 'RandomForestRegression':
        Random_Forest_Regression(housing)
    else :
        PhanNhomMedianIncome(housing)
