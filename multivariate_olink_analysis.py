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

# %% PRELIMINARY STEPS

# Olink data has been edited as follows:
# NPX normalized data (per chemical) is used in this analysis
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
additionaldata = pd.DataFrame(columns=["Sample", "Dataset", "Day", "Progress", "PatientID"])

# Build a dataframe out of matching rows from dataattributes for each row data
for ind in data.index:
    additionaldata = additionaldata.append(
        dataattributes[dataattributes.Sample == data.Sample[ind]])
    if dataattributes[dataattributes.Sample == data.Sample[ind]].empty:
        print(data.Sample[ind])

# Append additionaldata to data, in order to add attributes to the corresponding samples
print("TAIL:\n", additionaldata.tail())

additionaldata = additionaldata.reset_index(drop=True)

data = pd.concat([data, additionaldata.drop("Sample", axis=1)],
                 axis=1)

# %% Save complete data to new csv file
data.to_csv("C:/umea_immunology/experiments/corona/olink_data/formatted_data/20201969_Forsell_NPX_edit_complete.csv")

data = data.drop("Sample", axis=1)

# %% Create basic plot

plt.style.use('ggplot')

# divide data in different data sets (will produce a plot for each)

data.columns = data.columns.str.strip()  # get rid of whitespace in first column

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
    color_values = sns.color_palette("Set2", len(patient_unique))
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
