# %% can type in the python console `help(name of function)` to get the documentation
from pydoc import help
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from IPython.display import display, HTML

# figures inline in notebook
# %matplotlib inline

np.set_printoptions(suppress=True)

DISPLAY_MAX_ROWS = 20  # number of max rows to print for a DataFrame
pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)

# plot style
plt.style.use('ggplot')

# screeplot for PCA


def screeplot(pca, standardised_values, set):
    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    axis.plot(x, y, "o-")
    axis.set_xticks(x)
    axis.set_xticklabels(["Comp."+str(i) for i in x], rotation=60)
    axis.set_ylabel("Variance")
    axis.set_title("Scree Plot for "+set)

# scatterplot for PCA


def pca_scatter(pca, standardised_values, classifs, set):
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(zip(foo[:, 0], foo[:, 1], classifs), columns=["PC1", "PC2", "Class"])
    sns.lmplot(x="PC1", y="PC2", data=bar, hue="Class", fit_reg=False)
    ax = plt.gca()
    ax.set_title(set)


# %% PRELIMINARY STEPS

# Olink data has been edited as follows:
# NPX normalized data (per chemical) is used in this analysis
# Control Data has been duplicated for each dataset with different sampleID
# All red-filled cells are below LQL or above ULOQ have been replaced with 'NA'
# (for descriptions see Olink's certificated of analysis)
# Samples that didn't pass Olink's quality test (red numbers) and Olink's
# quality test samples have been deleted
# Remaining rows and columns only used for description have been deleted

# A file containing SampleID, Dataset, Day (on which sample has been taken),
# Progress (of disease), and PatientID has been created manually


# %% LOAD DATA
data = pd.read_csv(
    "C:/umea_immunology/experiments/corona/olink_data/formatted_data/20201969_Forsell_NPX_edit.csv")
dataattributes = pd.read_csv(
    "C:/umea_immunology/experiments/corona\olink_data/formatted_data/olink_sample_names.csv")

# save data columns (without Sample) for the plotting later
proteins = data.drop("Sample", axis=1).columns.str.strip()

print(data.head())
print(dataattributes.head())

# %% Append sample attributes to the right sample ID  (find efficient solution)

# Create empty dataframe to append
additionaldata = pd.DataFrame(
    columns=["Sample", "Dataset", "Day", "Progress", "PatientID", "Phase"])

# Build a dataframe out of matching rows from dataattributes for each row data
for ind in data.index:
    currentattribute = dataattributes[dataattributes.Sample == data.Sample[ind]]
    phase = ""

    if (currentattribute["Day"].item() <= 10):
        phase = pd.Series(["Early"], name="Phase")
    elif (currentattribute["Day"].item() >= 11 and currentattribute["Day"].item() < 90):
        phase = pd.Series(["Mid"], name="Phase")
    elif (currentattribute["Day"].item() >= 90):
        phase = pd.Series(["Late"], name="Phase")

    currentattribute.reset_index(drop=True, inplace=True)
    currentattribute = pd.concat([currentattribute, phase], axis=1)
    additionaldata = additionaldata.append(currentattribute, ignore_index=True)

    if dataattributes[dataattributes.Sample == data.Sample[ind]].empty:
        print(data.Sample[ind])

# Append additionaldata to data, in order to add attributes to the corresponding samples
print("TAIL:\n", additionaldata.tail())
print(additionaldata["Phase"].unique())
additionaldata = additionaldata.reset_index(drop=True)

data = pd.concat([data, additionaldata.drop("Sample", axis=1)],
                 axis=1)

# %% Save complete data to new csv file
data.to_csv("C:/umea_immunology/experiments/corona/olink_data/formatted_data/20201969_Forsell_NPX_edit_complete.csv")

data = data.drop("Sample", axis=1)
data.columns = data.columns.str.strip()  # get rid of whitespace in first column

# %% Create basic plots

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data.pdf")

# divide up data by dataset (one figure per dataset)
datasets_unique = data["Dataset"].unique()
for oneset in datasets_unique:
    data_oneset = data[data["Dataset"] == oneset]

    groups = data_oneset.groupby("Progress")
    patients = data_oneset.groupby("PatientID")

    patient_unique = data_oneset["Progress"].unique()
    color_values = sns.color_palette("colorblind", len(patient_unique))
    color_map = dict(zip(patient_unique, color_values))

    # begin figure
    fig, ax = plt.subplots(nrows=9, ncols=5, figsize=(
        30, 40))  # for 45 proteins
    fig.tight_layout(pad=3)
    # plot one graph per protein
    for i, axis in enumerate(ax.flat):

        # setup for plot
        for name, group in groups:
            axis.scatter(group["Day"], group[proteins[i]], label=name,
                         c=group["Progress"].map(color_map))

        axis.set_xlabel("Day")
        axis.set_ylabel("Normalized Protein Amount")
        axis.set_title(proteins[i]+" amounts in "+oneset+" samples")
        axis.legend(loc="upper right", title="Progress")

        for name, patient in patients:
            axis.plot(patient["Day"], patient[proteins[i]],
                      c=patient["Progress"].map(color_map).iloc[0])

    # save plot
    plotpages.savefig(fig)

plotpages.close()

# %% Principal Component analysis

datasets_unique = data["Dataset"].unique()
proteinlist = [str(x) for x in proteins.tolist()]

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data_PCA.pdf")

# split data by dataset
for oneset in datasets_unique:
    fig, axis = plt.subplots(figsize=(
        10, 5))
    fig.tight_layout(pad=3)

    data_oneset = data[data["Dataset"] == oneset]
    only_concentration_data = data_oneset[proteinlist]
    only_concentration_data = only_concentration_data.dropna(axis=1)
    pcaproteins = only_concentration_data.columns.str.strip()
    pcaproteinlist = [str(x) for x in pcaproteins.tolist()]

    standardised_oneset = scale(only_concentration_data)
    standardised_oneset = pd.DataFrame(
        standardised_oneset, index=only_concentration_data.index, columns=only_concentration_data.columns)
    print(standardised_oneset.apply(np.nanmean))
    print(standardised_oneset.apply(np.nanstd))

    pca = PCA().fit(standardised_oneset)
    screeplot(pca, standardised_oneset, oneset)
    fig2 = pca_scatter(pca, standardised_oneset, data_oneset["Progress"], oneset)

    # save plot
    plotpages.savefig(fig)
    plotpages.savefig(fig2)

    # correlate pricincipal components to variables
    indices = np.arange(len(pcaproteinlist))
    pcaproteinarray = np.array(pcaproteinlist)

    # First Component
    correlation_array = np.vstack((pcaproteinarray, pca.components_[0])).T
    correlation_dataframe = pd.DataFrame(
        data=correlation_array, index=indices, columns=["Protein", "Loadings"])
    correlation_dataframe.sort_values(by="Loadings", ascending=False).to_csv(
        "C:/umea_immunology/experiments/corona/olink_data/formatted_data/20201969_Forsell_NPX_edit_PCA_loadings.csv", mode='a', header=True)

    # Second Component
    correlation_array2 = np.vstack((pcaproteinarray, pca.components_[1])).T
    correlation_dataframe2 = pd.DataFrame(
        data=correlation_array2, index=indices, columns=["Protein", "Loadings"])
    correlation_dataframe2.sort_values(by="Loadings", ascending=False).to_csv(
        "C:/umea_immunology/experiments/corona/olink_data/formatted_data/20201969_Forsell_NPX_edit_PCA_loadings.csv", mode='a', header=True)

    # generate boxplots of the first three variables correlating with either component
    # get most important variables from Loadings (check, if lowest AND highest values)

    if len(correlation_dataframe[(correlation_dataframe["Loadings"].astype(float) >= 0.2) | (correlation_dataframe["Loadings"].astype(float) <= -0.2)]) >= 3:
        chosen_corr1 = correlation_dataframe[(correlation_dataframe["Loadings"].astype(float) >= 0.2) |
                                             (correlation_dataframe["Loadings"].astype(float) <= -0.2)].iloc[0: 3]
        chosen_corr1 = chosen_corr1["Protein"].tolist()
    elif len(correlation_dataframe[(correlation_dataframe["Loadings"].astype(float) >= 0.2) | (correlation_dataframe["Loadings"].astype(float) <= -0.2)]) == 0:
        chosen_corr1 = ""
    else:
        chosen_corr1 = correlation_dataframe[(correlation_dataframe["Loadings"].astype(float) >= 0.2) |
                                             (correlation_dataframe["Loadings"].astype(float) <= -0.2)]
        chosen_corr1 = chosen_corr1["Protein"].tolist()

    if len(correlation_dataframe2[(correlation_dataframe2["Loadings"].astype(float) >= 0.2) | (correlation_dataframe2["Loadings"].astype(float) <= -0.2)]) >= 3:
        chosen_corr2 = correlation_dataframe2[(correlation_dataframe2["Loadings"].astype(float) >= 0.2) |
                                              (correlation_dataframe2["Loadings"].astype(float) <= -0.2)].iloc[0: 3]
        chosen_corr2 = chosen_corr2["Protein"].tolist()
    elif len(correlation_dataframe2[(correlation_dataframe2["Loadings"] >= 0.2) | (correlation_dataframe2["Loadings"].astype(float) <= -0.2)]) == 0:
        chosen_corr2 = ""
    else:
        chosen_corr2 = correlation_dataframe2[(correlation_dataframe2["Loadings"].astype(float) >= 0.2) |
                                              (correlation_dataframe2["Loadings"].astype(float) <= -0.2)]
        chosen_corr2 = chosen_corr2["Protein"].tolist()

    # combine both lists
    chosen_corr_total = chosen_corr1+chosen_corr2
    chosen_corr_total = list(set(chosen_corr_total))  # no duplicates

    fig3, axes = plt.subplots(len(chosen_corr_total), figsize=(
        20, 30))
    fig3.tight_layout(pad=3)
    # start plotting per protein
    for i in range(0, len(chosen_corr_total)):
        sns.boxplot(y=chosen_corr_total[i], x="Phase",
                    data=data_oneset, palette="colorblind", hue="Progress", ax=axes[i], order=["Early", "Mid", "Late"])
        sns.stripplot(y=chosen_corr_total[i], x="Phase",
                      data=data_oneset, palette="colorblind", hue="Progress", ax=axes[i], order=["Early", "Mid", "Late"], dodge=True)
        axes[i].legend(loc="upper right", title="Progress")
        axes[i].set_xlabel("Phase")
        axes[i].set_ylabel("Normalized Protein Amount")
        axes[i].set_title(proteins[i]+" amounts in "+oneset+" samples")

    plotpages.savefig(fig3)

plotpages.close()
