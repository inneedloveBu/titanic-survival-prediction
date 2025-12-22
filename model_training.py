# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 保存测试集的乘客ID，用于最终提交文件
test_passenger_id = test_df['PassengerId']

# 2. 定义特征和目标变量

# 在加载数据后，创建新特征
for df in [train_df, test_df]:
    # 家庭大小
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # 是否独自一人
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    # 从姓名中提取头衔（如 Mr, Miss, Master）
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # 将罕见的头衔归类为 'Rare'
    rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')
    # 根据客舱号判断是否有客舱信息
    df['HasCabin'] = df['Cabin'].notna().astype(int)

## 更新特征列表
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone", "Title", "HasCabin"]
# 我们选择一些有潜力的特征，可以后续增减 features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train_df[features]
y = train_df["Survived"]
X_test = test_df[features]


# 3. 创建预处理管道
# 3.1 区分数字和分类特征
numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]


# 3.2 分别对它们进行预处理
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # 用中位数填充缺失值
    ('scaler', StandardScaler()) # 标准化
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # 用众数填充缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 独热编码
])

# 3.3 合并
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. 划分训练集和验证集，用于本地评估模型
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 创建完整的机器学习管道（预处理 + 模型）
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# 6. 划分训练集和验证集，用于本地评估模型
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 使用网格搜索进行模型训练与调优
# 定义要搜索的参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5]
}

# 创建GridSearchCV对象，使用5折交叉验证
search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# 在训练集上进行网格搜索
search.fit(X_train, y_train)

print(f"最佳参数: {search.best_params_}")
print(f"最佳交叉验证平均分数: {search.best_score_:.4f}")

# 8. 使用最佳模型在验证集上进行评估
best_model = search.best_estimator_
y_val_pred = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"调优后模型在验证集上的准确率为: {val_accuracy:.4f}")

# 9. 使用最佳参数重新训练最终模型（使用全部训练数据）
# 从最佳参数字典中移除 'classifier__' 前缀
rf_best_params = {key.replace('classifier__', ''): value for key, value in search.best_params_.items()}

final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**rf_best_params, random_state=42, n_jobs=-1))
])
final_model.fit(X, y)
test_predictions = final_model.predict(X_test)


# 2. 进行预测
test_predictions = final_model.predict(X_test)

# 3. 生成提交文件
output = pd.DataFrame({
    'PassengerId': test_passenger_id,  # 这个变量应该在文件开头附近定义过
    'Survived': test_predictions
})
output.to_csv('submission_tuned.csv', index=False)
print("调优后的竞赛提交文件已保存至 ‘submission_tuned.csv’")


# 新增：保存训练好的最终模型
import joblib
joblib.dump(final_model, 'titanic_model.pkl')
print("模型已保存至 ‘titanic_model.pkl’")