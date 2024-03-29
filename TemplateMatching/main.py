# library imports 
import os
import sys

# modules
from ImageProcessor import ImageProcessor
from utils import Match, CONFIG
from helper import suffix, flag_to_integer
from format_plot import analyze_result, format_result, plot_result, format_erupfall


#-----------------------------------------------------------
"PROCESSES"

def workflow_one(files: list[str]) -> None:
    """
    Processes images through "workflow one", which encompasses 
    template matching -> manual matching -> filtering
    -> fit projecting -> formating / plotting of results

    Params
    ------
    images: list of paths for images to processes
    """
    # remember to change FILTER to Filter.MANUAL in CONFIG
    match(files)
    manual(files)
    fitproj(files)
    format()
        
    

def match(files: list[str]):
    """
    Performs template matching on images

    Params
    ------
    images: list of paths for images to processes
    """
    for file_index in range(len(files)):
        process_img = ImageProcessor(files, file_index)
        process_img.template_matching(True)


def manual(files: list[str]):
    """
    Opens interface for manual editing of data

    Params
    ------
    images: list of paths for images to processes
    """
    for file_index in range(len(files)):
        process_img = ImageProcessor(files, file_index)
        process_img.manual(True)


def fitproj(files: list[str]):
    """
    Performs fitlering and projecting on images

    Params
    ------
    images: list of paths for images to processes
    """
    for file_index in range(len(files)):
        process_img = ImageProcessor(files, file_index)
        process_img.filter(True)
        process_img.fit_project(True)


def format() -> None:
    """
    Performs formating / plotting of results
    """
    format_result(True)
    format_erupfall(True)
    plot_result(True)


def analyze() -> None:
    """
    Open interface for analysis of results
    """
    analyze_result(True)




#-----------------------------------------------------------
if __name__ == "__main__":

    # obtains arguments
    args = sys.argv
    # bool structure for which processes to run
    processes = {"match": False,
                 "manual": False,
                 "fitproj": False,
                 "format": False,
                 "analyze": False}
    if "match" in args:
        processes["match"] = True
    if "manual" in args:
        processes["manual"] = True
    if "fitproj" in args:
        processes["fitproj"] = True
    if "format" in args:
        processes["format"] = True
    if "analyze" in args:
        processes["analyze"] = True
    
    # all other arguments 
    args = [arg for arg in args if (arg not in {"match", "manual", "fitproj", "format", "analyze", "main.py"})]
    # all images in image 
    files = sorted([file for file in os.listdir(os.path.join(os.getcwd(),"img")) if suffix(file) in CONFIG.FILE_TYPES])
    num_of_images = len(files)
    
    # if flags -s or -n exist 
    if "-s" in args or "-n" in args:
        start_index = flag_to_integer(args, "-s") if "-s" in args else 0
        num_to_process = flag_to_integer(args, "-n") if "-n" in args else (num_of_images - start_index)
        # get images
        files = files[start_index : min(start_index+num_to_process,num_of_images)]
        # if empty set, raise error
        if len(files) == 0:
            raise RuntimeError(f"""Empty list detected: either starting index {start_index} 
            is out of bounds (min = 0, max = {num_of_images-1}) or flag -n < 1""")
        
        print(f"""Processing images indexed {start_index} to {min(start_index+num_to_process-1,num_of_images-1)} 
        (index starts at 0): '{files[0]}' to '{files[-1]}'""")
        
    # if no flags but there's still arguments, assume those are "named images"
    elif len(args) > 0:
        files = sorted([arg for arg in args if os.path.isfile(os.path.join("img", arg))])
        print(f"Processing images {files}")
    # else process all 
    else:
        print(f"Processing all images")

    # if no processes named, run all
    if not any([processes["match"], 
                processes["manual"], 
                processes["fitproj"], 
                processes["format"],
                processes["analyze"]]):
        print("Running: all")
        workflow_one(files)
    else:
        # otherwise run them one by one if they're true
        if processes["match"]:
            print("Running: Template Matching")
            match(files)
        if processes["manual"]:
            print("Running: Manual Processing")
            manual(files)
        if processes["fitproj"]:
            print("Running: Fit Project")
            fitproj(files)
        if processes["format"]:
            print("Running: Format")
            format()
        if processes["analyze"]:
            print("Running: Analyze")
            analyze()





