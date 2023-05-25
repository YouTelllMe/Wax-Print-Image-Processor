import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from utils import CONFIG, parse_date


def format_result() -> None:
    """
    Sorts all output file into the desired format. 
    """

    # finds path of all data files, sorted
    data_paths = search_file(CONFIG.PATH, CONFIG.DATA_FILENAME)
    max_center_index = 0
    max_length = 0
    center_indecies = []
    dates = []

    for i in range(len(data_paths)):
        # read file
        df = pd.read_csv(data_paths[i])
        subdirname = os.path.basename(os.path.dirname(data_paths[i]))
        date = parse_date(subdirname)
        dates.append(date)

        # find center index
        center_tooth = df.index[df["type"] == "Tooth.CENTER_T"].to_numpy()
        if len(center_tooth) == 0:
            center_index = df.index[df["type"] == "Tooth.CENTER_G"].to_numpy()[0]
        else: 
            center_index = center_tooth[0]
        center_indecies.append(center_index)
        
        # update max length, update max center index
        if center_index > max_center_index:
            max_center_index = center_index
        if len(df) > max_length:
            max_length = len(df)

    # set dataframe shape (1 column for date + the rest for teeth indecies)
    columns = range(max_center_index + max_length - 1) - max_center_index
    columns = [str(column) for column in columns]
    df_output_arclength = pd.DataFrame(columns=["date"]+columns)
    df_output_binary = pd.DataFrame(columns=["date"]+columns)

    entry_so_far = 0
    for i in range(len(dates)):
        # read data
        df = pd.read_csv(data_paths[i])
        x = df["x"].to_numpy()
        types = df["type"]
        
        #arclength representation = x values of projection; binary data representation = 1 for teeth, 0 for gap
        arclength_data_rep = x - df["x"][center_indecies[i]]
        binary_data_rep = [1 if (types[i] == "Tooth.TOOTH" or types[i] == "Tooth.CENTER_T"
                                or types[i] == "Tooth.ERROR_T") else 0 for i in range(len(x))]

        # add front and back padding to obtain the correct shape to insert into dataframe
        arclength_data_rep_pad = padding(arclength_data_rep, center_indecies[i], max_center_index, len(columns))
        binary_data_rep_pad = padding(binary_data_rep, center_indecies[i], max_center_index, len(columns))

        # prepend date into the "date" column
        df_entry_arclength = [dates[i]] + arclength_data_rep_pad
        df_entry_binary = [dates[i]] + binary_data_rep_pad

        # set the maxlength'th entry to be the new entry; aka add new entry
        df_output_arclength.loc[entry_so_far] = df_entry_arclength
        df_output_binary.loc[entry_so_far] = df_entry_binary
        entry_so_far += 1

    # save to output folder
    df_output_binary.to_csv(os.path.join("processed", "output", "binary data.csv"))
    df_output_arclength.to_csv(os.path.join("processed", "output", "arclength data.csv"))



def plot_result() -> None:
    """
    Plot the formatted output data. 
    """

    outputPath = os.path.join("processed", "output")
    dfArclength = pd.read_csv(os.path.join(outputPath, "arclength data.csv"))
    dfBinary = pd.read_csv(os.path.join(outputPath, "binary data.csv"))

    arcTooth = []
    arcGap = []
    arcCenterT = []
    arcCenterG = []

    binTooth = []
    binGap = []
    binCenterT = []
    binCenterG = []

    toothY = []
    gapY = []
    centerTY = []
    centerGY = []

    # convert string into dates
    dates = sorted([datetime.strptime(d, '%Y-%m-%d') for d in dfArclength["date"]])
    # first two columns are: 1) index of df, 2) date
    indexColumns = dfArclength.columns.to_list()[2:] 

    for entryIndex in range(len(dates)):
        for columnIndex in indexColumns:
            arcEntry = dfArclength[columnIndex][entryIndex]
            binEntry = dfBinary[columnIndex][entryIndex]

            # if not empty
            if not pd.isna(binEntry):
                # if a tooth
                if int(binEntry) == 1:
                    # if column is the centerindex
                    if int(columnIndex) == 0:
                        arcCenterT.append(float(arcEntry))
                        binCenterT.append(float(columnIndex))
                        centerTY.append(dates[entryIndex]) 
                    # if not a centertooth
                    else:
                        arcTooth.append(float(arcEntry))
                        binTooth.append(float(columnIndex))
                        toothY.append(dates[entryIndex])
                # if a gap
                elif int(binEntry) == 0:
                    # if column is centerindex
                    if int(columnIndex) == 0:
                        arcCenterG.append(float(arcEntry))
                        binCenterG.append(float(columnIndex))
                        centerGY.append(dates[entryIndex])
                    # if not a centergap
                    else:
                        arcGap.append(float(arcEntry))
                        binGap.append(float(columnIndex))
                        gapY.append(dates[entryIndex])

    axFig, arcAx  = plt.subplots()
    binFig, binAx = plt.subplots()

    axFig.set_figwidth(CONFIG.WIDTH_SIZE)
    axFig.set_figheight(CONFIG.HEIGHT_SIZE)
    binFig.set_figwidth(CONFIG.WIDTH_SIZE)
    binFig.set_figheight(CONFIG.HEIGHT_SIZE)

    axFig.suptitle("Arclength Plot")
    binFig.suptitle("Index Plot")

    arcAx.scatter(arcTooth, toothY, c="c", s=10)
    arcAx.scatter(arcGap, gapY, c="#5A5A5A", s=10)
    arcAx.scatter(arcCenterG, centerGY, c="#FFCCCB", s=10)
    arcAx.scatter(arcCenterT, centerTY, c="r", s=10)

    binAx.scatter(binTooth, toothY, c="c", s=10)
    binAx.scatter(binGap, gapY, c="#5A5A5A", s=10)
    binAx.scatter(binCenterG, centerGY, c="#FFCCCB", s=10)
    binAx.scatter(binCenterT, centerTY, c="r", s=10)


    curr_date = dates[0]
    last_date = dates[-1]
    date_ticks = []
    while curr_date <= last_date:
        date_ticks.append(curr_date)
        curr_date += timedelta(days=3)

    teeth_index_ticks = np.linspace(-50, 50, num=101)
    teeth_arclength_ticks = np.linspace(-2500, 2500, num=101)

    arcAx.set_xticks(teeth_arclength_ticks, minor=True)
    binAx.set_xticks(teeth_index_ticks, minor=True)
    arcAx.set_yticks(date_ticks, minor=True)
    binAx.set_yticks(date_ticks, minor=True)

    arcAx.grid(which='minor', color="k", linestyle=":", alpha=0.5)
    binAx.grid(which='minor', color="k", linestyle=":", alpha=0.5)
    arcAx.grid(which='major', color="k", alpha=0.7)
    binAx.grid(which='major', color="k", alpha=0.7)
    axFig.tight_layout()
    binFig.tight_layout()

    axFig.savefig(os.path.join(outputPath,"arclength plot.png"))
    binFig.savefig(os.path.join(outputPath,"index plot.png"))


def search_file(root: str, file_name: str) -> list[str]:
    """
    Finds all instances of a file within the root folder.

    Params
    ------
    root: path of root directory of search from
    file_name: file name 

    Returns
    -------
    A list of the full path of each instance of file_name.
    """

    # returns all directories and subdirectories in root
    all_dir = [x[0] for x in os.walk(root)]
    file_instances = []
    
    # iterates through all directories
    for dir in all_dir:
        items = os.listdir(dir)
        for item in items:
            # if file name is what we're looking for, append full path
            if item == file_name:
                file_instances.append(os.path.join(dir, item))
    
    return sorted(file_instances)



def padding(unpadded_list: list, curr_center_ind: int, target_center_ind: int, num_of_cols: int) -> list:
    """
    Pad unpadded_list so the center index is aligned with the target center
    index. 

    Params
    ------
    unpadded_list: list to be padded
    curr_center_ind: current center index
    target_center_ind: target center index
    num_of_cols: number of columns of the panda dataframe to insert the result
    list; the length needed for the padded list 

    Returns
    -------
    unpadded_list with padding such that the center index is aligned with the
    target index and the length is num_of_cols

    Note: curr_center_ind is not necessarily the "center" of the list.
    """

    unpadded_list = list(unpadded_list)

    front_pad_size = target_center_ind - curr_center_ind
    front_padding = [None for _ in range(front_pad_size)]

    back_pad_size = num_of_cols - len(front_padding) - len(unpadded_list)
    back_padding = [None for _ in range(back_pad_size)]
    
    return front_padding + unpadded_list + back_padding