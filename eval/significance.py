import pandas as pd
import scipy.stats as stats

prebuilt = pd.read_csv("data_processing/data/results/crossval.csv", index_col=0)

M = 5

stanza = prebuilt[prebuilt["name"] == "stanza"]["f1"].values
distil = prebuilt[prebuilt["name"] == "distil"]["f1"].values
bert = prebuilt[prebuilt["name"] == "bert"]["f1"].values
roberta = prebuilt[prebuilt["name"] == "roberta"]["f1"].values
crf = prebuilt[prebuilt["name"] == "crf"]["f1"].values
crf_basic = prebuilt[prebuilt["name"] == "crf_basic"]["f1"].values

# stanza vs fine-tuned
# crf basic > 0.05
models = [distil, bert, roberta, crf, crf_basic]
for i in range(M):
    print(stats.ttest_ind(a=stanza, b=models[i], equal_var=False))


# distil vs fine-tuned
# crf + crf basic < 0.05
for model in [bert, roberta, crf, crf_basic]:
    print(stats.ttest_ind(a=distil, b=model, equal_var=False))
