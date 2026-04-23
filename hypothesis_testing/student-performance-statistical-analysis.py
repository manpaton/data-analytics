import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

df = pd.read_csv('StudentsPerformance 1.csv')

#1
math_scores = df['math score']
sample_mean = math_scores.mean()
t_stat, p_value = stats.ttest_1samp(math_scores, 65)
alpha = 0.05
print(p_value)
if p_value < alpha:
    print("Reject H0: our sample mean IS significantly different")
else:
    print("Fail to reject H0: our sample mean IS NOT significantly different")


#2
yesTestPrep = df[df['test preparation course'] == 'completed']['math score']
noTestPrep = df[df['test preparation course'] == 'no']['math score']
t_stat, p_value = stats.ttest_ind(yesTestPrep, noTestPrep, equal_var=False)
p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value /2
print(p_value_one_tailed)
if p_value_one_tailed < alpha:
    print("Reject H0, prep course improves math score")
else:
    print("Fail to reject H0: prep course does not improve math score")

#3
readingScore = df['reading score']
writingScore = df['writing score']

t_stat, p_value = stats.ttest_rel(readingScore, writingScore)
print(p_value)
if p_value < alpha:
    print("Reject H0: writing and reading score are not correlated")
else:
    print("Fail to reject H0")

#4


success_female = ((df['gender'] == 'female') & (df['math score'] > 80))
failure_female = ((df['gender'] == 'female') & (df['math score'] <= 80))

success_male = ((df['gender'] == 'male') & (df['math score'] > 80))
failure_male = ((df['gender'] == 'male') & (df['math score'] <= 80))

if success_female.sum() < 10 or failure_female.sum() < 10 or \
   success_male.sum() < 10 or failure_male.sum() < 10:
    print("Warning: sample sizes too small for reliable z-test")
else:
    x_female = success_female.sum()
    n_female = success_female.sum() + failure_female.sum()
    x_male = success_male.sum()
    n_male = success_male.sum() + failure_male.sum()

    successes = [x_female, x_male]
    nobs = [n_female, n_male]

    z_stat, p_value = proportions_ztest(successes, nobs)

    print("p-value:", p_value)

    alpha = 0.05
    if p_value < alpha:
        print("Reject H0: there is a gender gap in Excellence (Math > 80)")
    else:
        print("Fail to reject H0: no gender gap in Excellence")