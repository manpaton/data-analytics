import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

df = load_data()

print(df.describe().round(2))

sample = df.sample(2000, random_state=42)

# scatter
plt.figure(figsize=(8,5))
plt.scatter(sample['MedHouseVal'], sample['AveRooms'], alpha=0.5)
plt.title("House Value vs Rooms")
plt.show()

# hist
df.hist(bins=30, figsize=(12,10))
plt.show()

# correlation
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()