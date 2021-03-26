# %% can type in the python console `help(name of function)` to get the documentation
from pydoc import help
import pandas as pd
import numpy as np
import itertools
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
sns.set_context("notebook", rc={"font.size": 30, "axes.titlesize": 30,
                                "axes.labelsize": 30, "xtick.labelsize": 20, "ytick.labelsize": 20})

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


def pca_scatter(pca, standardised_values, classifs, set, phases):
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(zip(foo[:, 0], foo[:, 1], classifs, phases),
                       columns=["PC1", "PC2", "Group", "Phase"])
    lm = sns.lmplot(x="PC1", y="PC2", data=bar, col="Phase", hue="Group", palette="Dark2",
                    fit_reg=False, col_order=["Acute", "Convalescent", "Chronic"], )
    lm.set_titles(col_template="{col_name}"+" Phase")
    lm.fig.suptitle("PCA of "+set)
    lm.fig.subplots_adjust(top=0.9)
    plt.gca()
    #axes.legend(loc="upper right", title="Group")
    # axes.set_title(set)

# lineplot for IgX levels


def antiline(data, antibody, set):
    fig, axes = plt.subplots(2, figsize=(
        10, 7))
    fig.tight_layout(pad=3)
    sns.lineplot(data=data, x="Day", y=antibody, units="PatientID",
                 hue="Progress", palette="Dark2", estimator=None, lw=1, ax=axes[0])
    sns.boxplot(y=antibody, x="Phase", data=data, palette="Dark2",
                hue="Progress", ax=axes[1], order=["Acute", "Convalescent", "Chronic"])
    #axes = plt.gca()
    axes[1].legend(loc="upper right", title="Group", fontsize=20)
    axes[0].set_title(set+" "+antibody)
    axes[1].set_title(set+" "+antibody)

# Build a dataframe out of matching rows from dataattributes for each row data


def append_attributes(givendata, dataattributes):
    # Create empty dataframe to append
    add_data = pd.DataFrame(
        columns=["Sample", "Dataset", "Day", "Progress", "PatientID", "Phase"])

    for ind in givendata.index:
        currentattribute = dataattributes[dataattributes.Sample == givendata.Sample[ind]]
        phase = ""

        if (currentattribute["Day"].item() <= 10):
            phase = pd.Series(["Acute"], name="Phase")
        elif (currentattribute["Day"].item() >= 11 and currentattribute["Day"].item() < 90):
            phase = pd.Series(["Convalescent"], name="Phase")
        elif (currentattribute["Day"].item() >= 90):
            phase = pd.Series(["Chronic"], name="Phase")

        currentattribute.reset_index(drop=True, inplace=True)
        currentattribute = pd.concat([currentattribute, phase], axis=1)

        if dataattributes[dataattributes.Sample == givendata.Sample[ind]].empty:
            print(givendata.Sample[ind])
            add_data = pd.concat([add_data, pd.DataFrame(
                [[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]], columns=add_data.columns)], ignore_index=True)
        else:
            add_data = add_data.append(currentattribute, ignore_index=True)

    return(add_data)

# Append age and sex to matching patientIDs (after append_attributes)


def append_sexage(givendata, dataattributes):
    # Create empty dataframe to append
    add_data = pd.DataFrame(
        columns=["PatientID", "Age", "Sex", "age_group"])

    dataattributes["PatientID"] = dataattributes["PatientID"].astype("str")

    for ind in givendata.index:
        if dataattributes[dataattributes.PatientID == givendata.PatientID[ind]].empty:
            print(givendata.PatientID[ind])
            add_data = pd.concat([add_data, pd.DataFrame(
                [[np.NaN, np.NaN, np.NaN, np.NaN]], columns=add_data.columns)], ignore_index=True)
        else:
            currentattribute = dataattributes[dataattributes.PatientID == givendata.PatientID[ind]]

            agegroup = ""

            if (currentattribute["Age"].item() <= 25):
                agegroup = pd.Series(["Under_26_years"], name="age_group")
            elif (currentattribute["Age"].item() > 25 and currentattribute["Age"].item() <= 50):
                agegroup = pd.Series(["26-50_years"], name="age_group")
            elif (currentattribute["Age"].item() > 50):
                agegroup = pd.Series(["Over_50"], name="age_group")

            currentattribute.reset_index(drop=True, inplace=True)
            currentattribute["Sex"] = currentattribute["Sex"].item() + \
                "_" + givendata.Progress[ind]
            currentattribute = pd.concat([currentattribute, agegroup], axis=1)

            add_data = add_data.append(currentattribute, ignore_index=True)

    return(add_data)

# Append attributes and exchange Progress variable with highIgA and highIgM from lists


def append_highantibodiesIgA(givendata, dataattributes, highiga):
    # Create empty dataframe to append
    add_data = pd.DataFrame(
        columns=["Sample", "Dataset", "Day", "Progress", "PatientID", "Phase"])

    for ind in givendata.index:
        currentattribute = dataattributes[dataattributes.Sample == givendata.Sample[ind]]
        phase = ""

        if (currentattribute["Day"].item() <= 10):
            phase = pd.Series(["Acute"], name="Phase")
        elif (currentattribute["Day"].item() >= 11 and currentattribute["Day"].item() < 90):
            phase = pd.Series(["Convalescent"], name="Phase")
        elif (currentattribute["Day"].item() >= 90):
            phase = pd.Series(["Chronic"], name="Phase")

        currentattribute.reset_index(drop=True, inplace=True)

        highiga.study_ID = highiga.study_ID.astype(str)
        #highigm.study_ID = highigm.study_ID.astype(str)

        if (currentattribute["PatientID"].item() in highiga.values):
            currentattribute["Progress"] = "Chronic_High_IgA"
        else:
            currentattribute["Progress"] = "Chronic_Low_IgA"

        currentattribute = pd.concat([currentattribute, phase], axis=1)

        if dataattributes[dataattributes.Sample == givendata.Sample[ind]].empty:
            print(givendata.Sample[ind])
            add_data = pd.concat([add_data, pd.DataFrame(
                [[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]], columns=add_data.columns)], ignore_index=True)
        else:
            add_data = add_data.append(currentattribute, ignore_index=True)

    return(add_data)

# Append attributes and exchange Progress variable with combination of dataset and progress, exchange dataset with "complete"


def append_complete(givendata, dataattributes):
    # Create empty dataframe to append
    add_data = pd.DataFrame(
        columns=["Sample", "Dataset", "Day", "Progress", "PatientID", "Phase"])

    for ind in givendata.index:
        currentattribute = dataattributes[dataattributes.Sample == givendata.Sample[ind]]
        phase = ""

        if (currentattribute["Day"].item() <= 10):
            phase = pd.Series(["Acute"], name="Phase")
        elif (currentattribute["Day"].item() >= 11 and currentattribute["Day"].item() < 90):
            phase = pd.Series(["Convalescent"], name="Phase")
        elif (currentattribute["Day"].item() >= 90):
            phase = pd.Series(["Chronic"], name="Phase")

        currentattribute.reset_index(drop=True, inplace=True)

        currentattribute["Progress"] = currentattribute["Dataset"].item() + \
            "_" + currentattribute["Progress"].item()
        currentattribute["Dataset"] = "Complete"

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
            add_data = add_data.append(currentattribute.astype(float), ignore_index=True)

    return(add_data)


def append_stomachache(givendata, dataattributes, stomachache):
    # Create empty dataframe to append
    add_data = pd.DataFrame(
        columns=["Sample", "Dataset", "Day", "Progress", "PatientID", "Phase"])

    for ind in givendata.index:
        currentattribute = dataattributes[dataattributes.Sample == givendata.Sample[ind]]
        phase = ""

        if (currentattribute["Day"].item() <= 10):
            phase = pd.Series(["Acute"], name="Phase")
        elif (currentattribute["Day"].item() >= 11 and currentattribute["Day"].item() < 90):
            phase = pd.Series(["Convalescent"], name="Phase")
        elif (currentattribute["Day"].item() >= 90):
            phase = pd.Series(["Chronic"], name="Phase")

        currentattribute.reset_index(drop=True, inplace=True)

        stomachache.StudyID = stomachache.StudyID.astype(str)

        if (currentattribute["PatientID"].item() in stomachache.values):
            currentattribute["Progress"] = "Stomach_Ache"
        else:
            currentattribute["Progress"] = "No_Stomach_Ache"

        currentattribute = pd.concat([currentattribute, phase], axis=1)

        if dataattributes[dataattributes.Sample == givendata.Sample[ind]].empty:
            print(givendata.Sample[ind])
            add_data = pd.concat([add_data, pd.DataFrame(
                [[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]], columns=add_data.columns)], ignore_index=True)
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


def correlate_pc_loadings(indix, proteinarray, pca_values, protnumbers, filename):
    correlation_array = np.vstack((proteinarray, pca_values.components_[indix])).T
    correlation_dataframe = pd.DataFrame(
        data=correlation_array, index=protnumbers, columns=["Protein", "Loadings"])
    correlation_dataframe["Loadings"] = correlation_dataframe["Loadings"].astype(
        float)
    mask = correlation_dataframe["Loadings"].gt(0)
    correlation_dataframe = pd.concat([correlation_dataframe[mask].sort_values("Loadings", ascending=False),
                                       correlation_dataframe[~mask].sort_values("Loadings", ascending=False)], ignore_index=True)
    correlation_dataframe.to_csv(filename, mode='a', header=True)

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


def plot_highvar_proteins(current_pc, indix, data_of_oneset, chosenset, chosengroups):
    blackpal = ["black", "black", "black", "black", "black", "black", "black", "black"]
    fig, axes = plt.subplots(len(current_pc), figsize=(
        20, 5*len(current_pc)))
    fig.tight_layout(pad=3)
    # start plotting per protein
    for j in range(0, len(current_pc)):
        sns.boxplot(y=current_pc[j], x="Phase",
                    data=data_of_oneset, palette="Dark2", hue=chosengroups, ax=axes[j], order=["Acute", "Convalescent", "Chronic"])
        sns.stripplot(y=current_pc[j], x="Phase",
                      data=data_of_oneset, hue=chosengroups, palette=blackpal, ax=axes[j], order=["Acute", "Convalescent", "Chronic"], dodge=True)
        axes[j].legend(loc="upper right", title=chosengroups)
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

highIGA = pd.read_csv(
    "C:/umea_immunology/experiments/corona/olink_data/formatted_data/highIgA_olink.csv")

highIGM = pd.read_csv(
    "C:/umea_immunology/experiments/corona/olink_data/formatted_data/highIgM_olink.csv")

sexage = pd.read_csv(
    "C:/umea_immunology/experiments/corona/olink_data/formatted_data/COVID_age_sex.csv")

stomachache_list = pd.read_csv(
    "C:/umea_immunology/experiments/corona/olink_data/formatted_data/olink_stomach_ache.csv")

# save data columns (without Sample) for the plotting Chronicr
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

# %% Append sex and age info to the right patient ID  (find efficient solution) after attributes

additionaldata = append_sexage(data, sexage)

# Append additionaldata to data, in order to add attributes to the corresponding samples
print("TAIL:\n", additionaldata.tail())
additionaldata = additionaldata.reset_index(drop=True)

data = pd.concat([data, additionaldata.drop("PatientID", axis=1)],
                 axis=1)

# %% Append complete attributes for all datasets together to the right sample ID  (find efficient solution)

additionaldata = append_complete(data, attributes)

# Append additionaldata to data, in order to add attributes to the corresponding samples
print("TAIL:\n", additionaldata.tail())
print(additionaldata["Phase"].unique())
additionaldata = additionaldata.reset_index(drop=True)

data = pd.concat([data, additionaldata.drop("Sample", axis=1)],
                 axis=1)

# %% Append antibody levels to matching sample ID - after attributes (DROPS COLUMNS CONTAINING NAs)

additionaldata = append_antibodies(data, antibodies)

# Append additionaldata to data, in order to add attributes to the corresponding samples
additionaldata = additionaldata.reset_index(drop=True)

data = pd.concat([data, additionaldata.drop("Sample", axis=1)],
                 axis=1)

# %% Append sample attributes to the right sample ID and group right high IgA (find efficient solution)

additionaldata = append_highantibodiesIgA(data, attributes, highIGA)

# Append additionaldata to data, in order to add attributes to the corresponding samples
print("TAIL:\n", additionaldata.tail())
print(additionaldata["Phase"].unique())
additionaldata = additionaldata.reset_index(drop=True)

data = pd.concat([data, additionaldata.drop("Sample", axis=1)],
                 axis=1)

# %% Append sample attributes to the right sample ID and group stomach ache list (find efficient solution)

additionaldata = append_stomachache(data, attributes, stomachache_list)

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

    fig1 = plot_protein_amounts(data_oneset, proteins)
    # save plot
    plotpages.savefig(fig1)

plotpages.close()

# %% Plot antibody levels

datasets_unique = data["Dataset"].unique()
antibodylist = ["IgA", "IgM", "IgG"]

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data_antibody_levels.pdf")

# do analysis by dataset
for oneset in datasets_unique:
    data_oneset = data[data["Dataset"] == oneset]

    for onebody in antibodylist:
        data_onebody = data_oneset[["PatientID", "Day", "Progress", "Phase", onebody]]
        data_onebody = data_onebody.dropna(axis=0)
        if (not data_onebody[onebody].empty):
            fig2 = antiline(data_onebody, onebody, oneset)
            # save plot
            plotpages.savefig(fig2)

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
    # count = np.isinf(standardised_oneset).values.sum()
    # print("It contains " + str(count) + " infinite values")
    # count = np.isnan(standardised_oneset).values.sum()
    # print("It contains " + str(count) + " nan values")

    fig, axis = plt.subplots(figsize=(
        10, 5))
    fig.tight_layout(pad=3)

    pca = PCA(svd_solver='arpack').fit(standardised_oneset)
    screeplot(pca, standardised_oneset, oneset)

    fig2 = pca_scatter(pca, standardised_oneset,
                       data_oneset["Progress"], oneset, data_oneset["Phase"])

    # save plot
    plotpages.savefig(fig)
    plotpages.savefig(fig2)

    # correlate pricincipal components to variables
    protindices = np.arange(len(pcaproteinlist))
    pcaproteinarray = np.array(pcaproteinlist)

    filename = "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings.csv"
    corr_pc1 = correlate_pc_loadings(0, pcaproteinarray, pca, protindices, filename)
    corr_pc2 = correlate_pc_loadings(1, pcaproteinarray, pca, protindices, filename)

    # combine both lists
    chosen_corr_total = [corr_pc1, corr_pc2]

    for i in range(0, 2):
        current_corr = chosen_corr_total[i]
        fig3 = plot_highvar_proteins(current_corr, i, data_oneset, oneset, "Progress")
        plotpages.savefig(fig3)

plotpages.close()

# %% Principal Component analysis with Sex

datasets_unique = data["Dataset"].unique()
proteinlist = [str(x) for x in proteins.tolist()]

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data_PCA_sex.pdf")

# open empty file for loadings (overwrite old)
df = pd.DataFrame(list())
df.to_csv("C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings_sex.csv")

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
    # count = np.isinf(standardised_oneset).values.sum()
    # print("It contains " + str(count) + " infinite values")
    # count = np.isnan(standardised_oneset).values.sum()
    # print("It contains " + str(count) + " nan values")

    fig, axis = plt.subplots(figsize=(
        10, 5))
    fig.tight_layout(pad=3)

    pca = PCA(svd_solver='arpack').fit(standardised_oneset)
    screeplot(pca, standardised_oneset, oneset)
    fig2 = pca_scatter(pca, standardised_oneset, data_oneset["Sex"], oneset, data_oneset["Phase"])

    # save plot
    plotpages.savefig(fig)
    plotpages.savefig(fig2)

    # correlate pricincipal components to variables
    protindices = np.arange(len(pcaproteinlist))
    pcaproteinarray = np.array(pcaproteinlist)

    filename = "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings_sex.csv"
    corr_pc1 = correlate_pc_loadings(0, pcaproteinarray, pca, protindices, filename)
    corr_pc2 = correlate_pc_loadings(1, pcaproteinarray, pca, protindices, filename)

    # combine both lists
    chosen_corr_total = [corr_pc1, corr_pc2]

    for i in range(0, 2):
        current_corr = chosen_corr_total[i]
        fig3 = plot_highvar_proteins(current_corr, i, data_oneset, oneset, "Sex")
        plotpages.savefig(fig3)

plotpages.close()

# %% Principal Component analysis with Age

datasets_unique = data["Dataset"].unique()
proteinlist = [str(x) for x in proteins.tolist()]

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data_PCA_age.pdf")

# open empty file for loadings (overwrite old)
df = pd.DataFrame(list())
df.to_csv("C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings_age.csv")

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
    # count = np.isinf(standardised_oneset).values.sum()
    # print("It contains " + str(count) + " infinite values")
    # count = np.isnan(standardised_oneset).values.sum()
    # print("It contains " + str(count) + " nan values")

    fig, axis = plt.subplots(figsize=(
        10, 5))
    fig.tight_layout(pad=3)

    pca = PCA(svd_solver='arpack').fit(standardised_oneset)
    screeplot(pca, standardised_oneset, oneset)
    fig2 = pca_scatter(pca, standardised_oneset,
                       data_oneset["age_group"], oneset, data_oneset["Phase"])

    # save plot
    plotpages.savefig(fig)
    plotpages.savefig(fig2)

    # correlate pricincipal components to variables
    protindices = np.arange(len(pcaproteinlist))
    pcaproteinarray = np.array(pcaproteinlist)

    filename = "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings_age.csv"
    corr_pc1 = correlate_pc_loadings(0, pcaproteinarray, pca, protindices, filename)
    corr_pc2 = correlate_pc_loadings(1, pcaproteinarray, pca, protindices, filename)

    # combine both lists
    chosen_corr_total = [corr_pc1, corr_pc2]

    for i in range(0, 2):
        current_corr = chosen_corr_total[i]
        fig3 = plot_highvar_proteins(current_corr, i, data_oneset, oneset, "age_group")
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

proteinlist = [str(x) for x in antibodproteins.tolist()] + ["IgA"]

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data_PCA_with_antibody.pdf")

# open empty file for loadings (overwrite old)
df = pd.DataFrame(list())
df.to_csv("C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings_with_antibody.csv")

# do analysis by dataset
for oneset in datasets_unique:
    data_oneset = data[data["Dataset"] == oneset]

    only_concentration_data = data_oneset[proteinlist]
    only_concentration_data = only_concentration_data.dropna(axis=0)

    if not only_concentration_data.empty:
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

        pca = PCA(svd_solver='arpack').fit(standardised_oneset)
        screeplot(pca, standardised_oneset, oneset)
        fig2 = pca_scatter(pca, standardised_oneset,
                           data_oneset["Progress"], oneset, data_oneset["Phase"])

        # save plot
        plotpages.savefig(fig)
        plotpages.savefig(fig2)

        # correlate pricincipal components to variables
        protindices = np.arange(len(pcaproteinlist))
        pcaproteinarray = np.array(pcaproteinlist)

        filename = "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings_with_antibody.csv"
        corr_pc1 = correlate_pc_loadings(0, pcaproteinarray, pca, protindices, filename)
        corr_pc2 = correlate_pc_loadings(1, pcaproteinarray, pca, protindices, filename)

        # combine both lists
        chosen_corr_total = [corr_pc1, corr_pc2]

        for i in range(0, 2):
            current_corr = chosen_corr_total[i]
            fig3 = plot_highvar_proteins(current_corr, i, data_oneset, oneset, "Progress")
            plotpages.savefig(fig3)

plotpages.close()

# %% Principal Component analysis after dropping healthy group

data.drop(data[data["Progress"] == "Healthy"].index, inplace=True)
datasets_unique = data["Dataset"].unique()
proteinlist = [str(x) for x in proteins.tolist()]

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data_PCA_without_healthy.pdf")

# open empty file for loadings (overwrite old)
df = pd.DataFrame(list())
df.to_csv("C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings_without_healthy.csv")

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
    # count = np.isinf(standardised_oneset).values.sum()
    # print("It contains " + str(count) + " infinite values")
    # count = np.isnan(standardised_oneset).values.sum()
    # print("It contains " + str(count) + " nan values")

    fig, axis = plt.subplots(figsize=(
        10, 5))
    fig.tight_layout(pad=3)

    pca = PCA(svd_solver='arpack').fit(standardised_oneset)
    screeplot(pca, standardised_oneset, oneset)
    fig2 = pca_scatter(pca, standardised_oneset,
                       data_oneset["Progress"], oneset, data_oneset["Phase"])

    # save plot
    plotpages.savefig(fig)
    plotpages.savefig(fig2)

    # correlate pricincipal components to variables
    protindices = np.arange(len(pcaproteinlist))
    pcaproteinarray = np.array(pcaproteinlist)

    filename = "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/20201969_Forsell_NPX_edit_PCA_loadings_without_healthy.csv"
    corr_pc1 = correlate_pc_loadings(0, pcaproteinarray, pca, protindices, filename)
    corr_pc2 = correlate_pc_loadings(1, pcaproteinarray, pca, protindices, filename)

    # combine both lists
    chosen_corr_total = [corr_pc1, corr_pc2]

    for i in range(0, 2):
        current_corr = chosen_corr_total[i]
        fig3 = plot_highvar_proteins(current_corr, i, data_oneset, oneset, "Progress")
        plotpages.savefig(fig3)

plotpages.close()

# %% Correlation matrices of different time phases and groups based on progress

datasets_unique = data["Dataset"].unique()
proteinlist = [str(x) for x in proteins.tolist()]

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data_correlationmatrices.pdf")

# do analysis by dataset
for oneset in datasets_unique:
    data_oneset = data[data["Dataset"] == oneset]
    data_groups = data_oneset["Progress"].unique()

    fig, axes = plt.subplots(len(data_groups), 3, figsize=(
        25, 7*len(data_groups)))
    fig.tight_layout(pad=3)
    figrow = -1

    for onegroup in data_groups:
        figcol = -1
        figrow = figrow+1
        data_onegroup = data_oneset[data_oneset["Progress"] == onegroup]
        data_phases = data_onegroup["Phase"].unique()
        sorted_data_phases = []
        if "Acute" in data_phases:
            sorted_data_phases += ["Acute"]
        if "Convalescent" in data_phases:
            sorted_data_phases += ["Convalescent"]
        if "Chronic" in data_phases:
            sorted_data_phases += ["Chronic"]

        for onephase in sorted_data_phases:
            figcol = figcol+1
            data_onephase = data_onegroup[data_onegroup["Phase"] == onephase]

            only_concentration_data = data_onephase[proteinlist]
            only_concentration_data = only_concentration_data.dropna(axis=1)

            corrmat = only_concentration_data.corr()

            # start plotting
            sns.heatmap(corrmat, vmin=-1, vmax=1, center=0,
                        cmap=sns.diverging_palette(20, 220, n=100), square=True, ax=axes[figrow, figcol], xticklabels=True, yticklabels=True)
            axes[figrow, figcol].set_title(
                "Correlation Matrix of "+oneset+" "+onegroup+" "+onephase)
            # axes[figrow, figcol].set_xticklabels(
            # axes[figrow, figcol].get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    plotpages.savefig(fig)

plotpages.close()

# %% Correlation matrices of group vs group (progress) in different time phases
antibody_switch = 1
groupies = "Progress"

datasets_unique = data["Dataset"].unique()

if (antibody_switch == 1):
    proteinlist = [str(x) for x in antibodproteins.tolist()] + ["IgA"]
else:
    proteinlist = [str(x) for x in proteins.tolist()]

# setup pdf for saving plots
plotpages = PdfPages(
    "C:/umea_immunology/experiments/corona/olink_data/olinkanalysis/preliminary_olink_data_correlationmatrices_comparegroups.pdf")

# do analysis by dataset
for oneset in datasets_unique:
    data_oneset = data[data["Dataset"] == oneset]

    figrow = -1

    data_phases = data_oneset["Phase"].unique()
    sorted_data_phases = []
    if "Acute" in data_phases:
        sorted_data_phases += ["Acute"]
    if "Convalescent" in data_phases:
        sorted_data_phases += ["Convalescent"]
    if "Chronic" in data_phases:
        sorted_data_phases += ["Chronic"]

    fig, axes = plt.subplots(len(sorted_data_phases), figsize=(12, 4*len(sorted_data_phases)))
    fig.tight_layout(pad=3)

    figcol = -1

    for onephase in sorted_data_phases:
        figcol = figcol+1

        data_onephase = data_oneset[data_oneset["Phase"] == onephase]
        data_groups = data_onephase[groupies].unique()
        data_groups = [x for x in data_groups if str(x) != 'nan']

        data_grouplist = list(itertools.combinations(data_groups, r=2))

        all_groups_corr = pd.DataFrame(
            columns=proteinlist)

        for onegroup in data_grouplist:

            data_onegroupA = data_onephase[data_onephase[groupies] == onegroup[0]]
            data_onegroupB = data_onephase[data_onephase[groupies] == onegroup[1]]

            only_concentration_dataA = data_onegroupA[proteinlist]
            only_concentration_dataB = data_onegroupB[proteinlist]

            if (antibody_switch == 1):
                only_concentration_dataA = only_concentration_dataA.dropna(
                    axis=0).reset_index(drop=True)
                only_concentration_dataB = only_concentration_dataB.dropna(
                    axis=0).reset_index(drop=True)
            else:
                only_concentration_dataA = only_concentration_dataA.dropna(
                    axis=1).reset_index(drop=True)
                only_concentration_dataB = only_concentration_dataB.dropna(
                    axis=1).reset_index(drop=True)

            matchingcols = list(set(only_concentration_dataA) & set(only_concentration_dataB))
            only_concentration_dataA = only_concentration_dataA[matchingcols]
            only_concentration_dataB = only_concentration_dataB[matchingcols]

            corrmat = only_concentration_dataA.corrwith(only_concentration_dataB, axis=0)
            corrmat = pd.Series(corrmat, name=onegroup[0]+" vs "+onegroup[1])
            all_groups_corr = all_groups_corr.append(corrmat)

        # start plotting
        if (not all_groups_corr.empty and not corrmat.dropna(axis=0).empty):
            sns.heatmap(all_groups_corr, vmin=-1, vmax=1, center=0,
                        cmap=sns.diverging_palette(20, 220, n=100), square=True, ax=axes[figcol], xticklabels=True, yticklabels=True)
            axes[figcol].set_title(
                "Correlation Matrix of "+oneset+" "+onephase)
            axes[figcol].tick_params('y', labelrotation=0)
            axes[figcol].tick_params('x', top=True, bottom=False,
                                     labeltop=True, labelbottom=False)
            plt.setp(axes[figcol].get_xticklabels(), rotation=40, ha="left",
                     rotation_mode="anchor")

    plotpages.savefig(fig)

plotpages.close()
