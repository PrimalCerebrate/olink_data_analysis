# %% can type in the python console `help(name of function)` to get the documentation
from pydoc import help
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from IPython.display import display, HTML

# figures inline in notebook
# %matplotlib inline

np.set_printoptions(suppress=True)

DISPLAY_MAX_ROWS = 20  # number of max rows to print for a DataFrame
pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)

# plot style
plt.style.use('ggplot')


# %% Define functions used
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
    sns.lmplot(x="PC1", y="PC2", data=bar, hue="Class", fit_reg=False, palette="colorblind")
    ax = plt.gca()
    ax.set_title(set)

# Build a dataframe out of matching rows from dataattributes for each row data


def append_attributes(givendata, dataattributes):
    # Create empty dataframe to append
    add_data = pd.DataFrame(
        columns=["Sample", "Dataset", "Day", "Progress", "PatientID", "Phase"])

    for ind in givendata.index:
        currentattribute = dataattributes[dataattributes.Sample == givendata.Sample[ind]]
        phase = ""

        if (currentattribute["Day"].item() <= 10):
            phase = pd.Series(["Early"], name="Phase")
        elif (currentattribute["Day"].item() >= 11 and currentattribute["Day"].item() < 90):
            phase = pd.Series(["Mid"], name="Phase")
        elif (currentattribute["Day"].item() >= 90):
            phase = pd.Series(["Late"], name="Phase")

        currentattribute.reset_index(drop=True, inplace=True)
        currentattribute = pd.concat([currentattribute, phase], axis=1)

        if dataattributes[dataattributes.Sample == givendata.Sample[ind]].empty:
            print(givendata.Sample[ind])
            add_data = pd.concat([add_data, pd.DataFrame(
                [[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]], columns=add_data.columns)], ignore_index=True)
        else:
            add_data = add_data.append(currentattribute, ignore_index=True)

    return(add_data)

# Append antibody levels


def append_antibodies(givendata, antibodydata):
    # Create empty dataframe to append
    add_data = pd.DataFrame(
        columns=["Sample", "IgG", "IgA", "IgM"])

    antibodydata.Sample = antibodydata.Sample.astype(str)

    for ind in givendata.index:
        currentattribute = antibodydata[antibodydata.Sample == givendata.Sample[ind]]

        currentattribute.reset_index(drop=True, inplace=True)

        if antibodydata[antibodydata.Sample == givendata.Sample[ind]].empty:
            print(givendata.Sample[ind])
            add_data = pd.concat([add_data, pd.DataFrame(
                [[np.NaN, np.NaN, np.NaN, np.NaN]], columns=add_data.columns)], ignore_index=True)
        else:
            add_data = add_data.append(currentattribute, ignore_index=True)

    return(add_data)

# Plot protein amounts per dataset with day and progress categorical variables


def plot_protein_amounts(one_set, chosenproteins):
    groups = one_set.groupby("Progress")
    patients = one_set.groupby("PatientID")

    patient_unique = one_set["Progress"].unique()
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
            axis.scatter(group["Day"], group[chosenproteins[i]], label=name,
                         c=group["Progress"].map(color_map))
            axis.set_xlabel("Day")
            axis.set_ylabel("Normalized Protein Amount")
            axis.set_title(chosenproteins[i]+" amounts in "+oneset+" samples")
            axis.legend(loc="upper right", title="Progress")

        for name, patient in patients:
            axis.plot(patient["Day"], patient[chosenproteins[i]],
                      c=patient["Progress"].map(color_map).iloc[0])

# correlate PCA Loadings


def correlate_pc_loadings(indix, proteinarray, pca_values, protnumbers):
    correlation_array = np.vstack((proteinarray, pca_values.components_[indix])).T
    correlation_dataframe = pd.DataFrame(
        data=correlation_array, index=protnumbers, columns=["Protein", "Loadings"])
    correlation_dataframe["Loadings"] = correlation_dataframe["Loadings"].astype(
        float)
    mask = correlation_dataframe["Loadings"].gt(0)
    correlation_dataframe = pd.concat([correlation_dataframe[mask].sort_values("Loadings", ascending=False),
                                       correlation_dataframe[~mask].sort_values("Loadings", ascending=False)], ignore_index=True)
    correlation_dataframe.to_csv(
        "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings.csv", mode='a', header=True)

    pos_filter_corr = correlation_dataframe[(
        correlation_dataframe["Loadings"] >= 0.2)].reset_index(drop=True)
    neg_filter_corr = correlation_dataframe[(
        correlation_dataframe["Loadings"] <= -0.2)].reset_index(drop=True)

    if len(pos_filter_corr) >= 5:
        pos_chosen_corr = pos_filter_corr.iloc[0: 5]
        pos_chosen_corr = pos_chosen_corr["Protein"].tolist()
    elif pos_filter_corr.empty:
        pos_chosen_corr = []
    else:
        pos_chosen_corr = pos_filter_corr
        pos_chosen_corr = pos_chosen_corr["Protein"].tolist()

    if len(neg_filter_corr) >= 5:
        neg_chosen_corr = neg_filter_corr.iloc[-5:]
        neg_chosen_corr = neg_chosen_corr["Protein"].tolist()
    elif neg_filter_corr.empty:
        neg_chosen_corr = []
    else:
        neg_chosen_corr = neg_filter_corr
        neg_chosen_corr = neg_chosen_corr["Protein"].tolist()

    chosen_corr = pos_chosen_corr + neg_chosen_corr

    return(chosen_corr)


def plot_highvar_proteins(current_pc, indix, data_of_oneset, chosenset):
    blackpal = ["black", "black", "black", "black", "black", "black", "black", "black"]
    fig, axes = plt.subplots(len(current_pc), figsize=(
        20, 5*len(current_pc)))
    fig.tight_layout(pad=3)
    # start plotting per protein
    for j in range(0, len(current_pc)):
        sns.boxplot(y=current_pc[j], x="Phase",
                    data=data_of_oneset, palette="colorblind", hue="Progress", ax=axes[j], order=["Early", "Mid", "Late"])
        sns.stripplot(y=current_pc[j], x="Phase",
                      data=data_of_oneset, hue="Progress", palette=blackpal, ax=axes[j], order=["Early", "Mid", "Late"], dodge=True)
        axes[j].legend(loc="upper right", title="Progress")
        axes[j].set_xlabel("Phase")
        axes[j].set_ylabel("Normalized Protein Amount")
        axes[j].set_title("For PC"+str(indix+1)+": "+current_pc[j] +
                          " amounts in "+chosenset+" samples")


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
attributes = pd.read_csv(
    "C:/umea_immunology/experiments/corona/olink_data/formatted_data/olink_sample_names.csv")
antibodies = pd.read_csv(
    "C:/umea_immunology/experiments/corona/olink_data/formatted_data/IgGIgAIgMELISAresults22012021_edit.csv")

# save data columns (without Sample) for the plotting later
proteins = data.drop("Sample", axis=1).columns.str.strip()

antibodproteins = data.dropna(axis=1).drop(
    "Sample", axis=1).columns.str.strip()

print(data.head())
print(attributes.head())

# %% Append sample attributes to the right sample ID  (find efficient solution)

additionaldata = append_attributes(data, attributes)

# Append additionaldata to data, in order to add attributes to the corresponding samples
print("TAIL:\n", additionaldata.tail())
print(additionaldata["Phase"].unique())
additionaldata = additionaldata.reset_index(drop=True)

data = pd.concat([data, additionaldata.drop("Sample", axis=1)],
                 axis=1)

# %% Append antibody levels to matching sample ID (DROPS COLUMNS CONTAINING NAs)

additionaldata = append_antibodies(data, antibodies)

# Append additionaldata to data, in order to add attributes to the corresponding samples
additionaldata = additionaldata.reset_index(drop=True)

data = pd.concat([data, additionaldata.drop("Sample", axis=1)],
                 axis=1)

print(data.tail())

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

    fig1 = plot_protein_amounts(data_oneset, proteins)
    # save plot
    plotpages.savefig(fig1)

plotpages.close()

# %% Principal Component analysis

datasets_unique = data["Dataset"].unique()
proteinlist = [str(x) for x in proteins.tolist()]

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data_PCA.pdf")

# open empty file for loadings (overwrite old)
df = pd.DataFrame(list())
df.to_csv("C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings.csv")

# do analysis by dataset
for oneset in datasets_unique:
    data_oneset = data[data["Dataset"] == oneset]

    only_concentration_data = data_oneset[proteinlist]
    only_concentration_data = only_concentration_data.dropna(axis=1)
    pcaproteins = only_concentration_data.columns.str.strip()
    pcaproteinlist = [str(x) for x in pcaproteins.tolist()]

    standardised_oneset = scale(only_concentration_data)
    standardised_oneset = pd.DataFrame(
        standardised_oneset, index=only_concentration_data.index, columns=only_concentration_data.columns)

    # Data QC
    # print(standardised_oneset.apply(np.nanmean))
    # print(standardised_oneset.apply(np.nanstd))

    fig, axis = plt.subplots(figsize=(
        10, 5))
    fig.tight_layout(pad=3)

    pca = PCA().fit(standardised_oneset)
    screeplot(pca, standardised_oneset, oneset)
    fig2 = pca_scatter(pca, standardised_oneset, data_oneset["Progress"], oneset)

    # save plot
    plotpages.savefig(fig)
    plotpages.savefig(fig2)

    # correlate pricincipal components to variables
    protindices = np.arange(len(pcaproteinlist))
    pcaproteinarray = np.array(pcaproteinlist)

    corr_pc1 = correlate_pc_loadings(0, pcaproteinarray, pca, protindices)
    corr_pc2 = correlate_pc_loadings(1, pcaproteinarray, pca, protindices)

    # combine both lists
    chosen_corr_total = [corr_pc1, corr_pc2]

    for i in range(0, 2):
        current_corr = chosen_corr_total[i]
        fig3 = plot_highvar_proteins(current_corr, i, data_oneset, oneset)
        plotpages.savefig(fig3)

plotpages.close()

# %% t-SNE analysis of data

datasets_unique = data["Dataset"].unique()
proteinlist = [str(x) for x in proteins.tolist()]

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data_tSNE.pdf")

# split by dataset

for oneset in datasets_unique:
    data_oneset = data[data["Dataset"] == oneset]

    progresslist = data_oneset["Progress"].unique()
    progresslist = [str(x) for x in progresslist.tolist()]

    only_concentration_data = data_oneset[proteinlist]
    only_concentration_data = only_concentration_data.dropna(axis=1)
    singleproteins = only_concentration_data.columns.str.strip()
    proteinlist = [str(x) for x in singleproteins.tolist()]

    standardised_oneset = scale(only_concentration_data)
    standardised_oneset = pd.DataFrame(
        standardised_oneset, index=only_concentration_data.index, columns=only_concentration_data.columns)

    tsne_model = TSNE(n_components=2, perplexity=20, verbose=1,
                      learning_rate=200, n_iter=1000)
    tsne_results = tsne_model.fit_transform(standardised_oneset)

    data_oneset["tsne_one"] = tsne_results[:, 0]
    data_oneset["tsne_two"] = tsne_results[:, 1]

    fig, axis = plt.subplots(figsize=(
        10, 5))
    fig.tight_layout(pad=3)

    sns.scatterplot(x="tsne_one", y="tsne_two", hue="Progress",
                    palette=sns.color_palette("hls", len(progresslist)), data=data_oneset, legend="full", ax=axis)
    axis.set_title(oneset)

    plotpages.savefig(fig)
plotpages.close()

# %% Principal Component analysis with antibody data (exclude rows with NAs after excluding olink proteins)

datasets_unique = data["Dataset"].unique()

proteinlist = [str(x) for x in antibodproteins.tolist()]  # + ["IgG", "IgA", "IgM"]

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data_PCA.pdf")

# open empty file for loadings (overwrite old)
df = pd.DataFrame(list())
df.to_csv("C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings.csv")

# do analysis by dataset
for oneset in datasets_unique:
    data_oneset = data[data["Dataset"] == oneset]

    only_concentration_data = data_oneset[proteinlist]
    only_concentration_data = only_concentration_data.dropna(axis=0)

    pcaproteins = only_concentration_data.columns.str.strip()
    pcaproteinlist = [str(x) for x in pcaproteins.tolist()]

    standardised_oneset = scale(only_concentration_data)
    standardised_oneset = pd.DataFrame(
        standardised_oneset, index=only_concentration_data.index, columns=only_concentration_data.columns)

    # Data QC
    # print(standardised_oneset.apply(np.nanmean))
    # print(standardised_oneset.apply(np.nanstd))

    fig, axis = plt.subplots(figsize=(
        10, 5))
    fig.tight_layout(pad=3)

    pca = PCA().fit(standardised_oneset)
    screeplot(pca, standardised_oneset, oneset)
    fig2 = pca_scatter(pca, standardised_oneset, data_oneset["Progress"], oneset)

    # save plot
    plotpages.savefig(fig)
    plotpages.savefig(fig2)

    # correlate pricincipal components to variables
    protindices = np.arange(len(pcaproteinlist))
    pcaproteinarray = np.array(pcaproteinlist)

    corr_pc1 = correlate_pc_loadings(0, pcaproteinarray, pca, protindices)
    corr_pc2 = correlate_pc_loadings(1, pcaproteinarray, pca, protindices)

    # combine both lists
    chosen_corr_total = [corr_pc1, corr_pc2]

    for i in range(0, 2):
        current_corr = chosen_corr_total[i]
        fig3 = plot_highvar_proteins(current_corr, i, data_oneset, oneset)
        plotpages.savefig(fig3)

plotpages.close()
