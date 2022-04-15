from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


iris_dataset = load_iris()
pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 5)
df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
df['species'] = pd.Categorical.from_codes(iris_dataset.target, iris_dataset.target_names).astype("category")
print(df.head())
print("Dataset Shape: ", df.shape)
print("\nNumber of species in each class:")
print(df['species'].value_counts())
print("More information: \n", df.describe())
print("Correlation with target: \n", df.corrwith(df["species"].cat.codes))
fig = plt.figure()
corr_mat = df.corr()
sns.heatmap(corr_mat, annot=True, square=True)
plt.xticks(rotation=0)
plt.show()

sns.pairplot(df, hue="species", height=2, palette='colorblind')
plt.show()

