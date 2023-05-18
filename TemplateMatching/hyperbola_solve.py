# library imports
import os 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import fsolve


#helper functions
from cross_prod_center import cross_prod_center
import GUI
from utils import CONFIG, Filter, Match
import project_1D
import analyze_projection

'''
This file contains:
1. Solve - fits a hyperbola to the filtered data and performs projection
2. Equidistant Set - returns a set of coordinates on the hyperbola equidistant in terms of arclength
3. Func - helper for Equidistant Set; non-linear system to solve to obtain equidistant data points
4. Plot Hyperbola Linear - returns a set of coordinates on the hyperbola equidistant in terms of x
'''

def solve(file_name, img_name, img_height):

    current_dir = os.getcwd()
    os.chdir(os.path.join(current_dir,'processed', "filter data"))
    if CONFIG.FILTER == Filter.GRADIENT:
        df = pd.read_csv(f"{img_name}_grad.csv")
    elif CONFIG.FILTER == Filter.GRADIENT_EVEN:
        df = pd.read_csv(f"{img_name}_gradeven.csv")
    elif CONFIG.FILTER == Filter.SMOOTH:
        df = pd.read_csv(f"{img_name}_smooth.csv")
    elif CONFIG.FILTER == Filter.SMOOTH_EVEN:
        df = pd.read_csv(f"{img_name}_smootheven.csv")
    else:
        df = pd.read_csv(f"{img_name}.csv")
    os.chdir(current_dir)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    matrix_t = [x**2, x*y, y**2, x, y]
    matrix = np.transpose(matrix_t)
    coeff = np.matmul(np.linalg.inv(np.matmul(matrix_t, matrix)),np.matmul(matrix_t, np.ones(np.shape(matrix)[0])))

    (A,B,C,D,E) = coeff

    error = False
    ends = np.roots([A, img_height*B+D, -1+C*img_height**2+E*img_height])

    if B**2-4*A*C < 0:
        fit = plot_hyperbola_linear(min(ends), max(ends), coeff)
        error = True
    else: 
        try: 
            fit = equidistant_set(min(ends), max(ends), coeff)
        except RuntimeError as err:
            raise RuntimeError(err)

    img_path = os.path.join('img', file_name)
    img = cv2.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=mpl.colormaps['gray'])
    ax.plot(fit[0], fit[1], '.-r', label="fit")

    target = os.path.join(current_dir,"processed", "fit visualization")
    os.chdir(target)
    fig.savefig(file_name)
    os.chdir(current_dir)

    if error:
        raise RuntimeError(f"Unable to fit a Hyperbola or Parabola; Circle or Ellipse detected.\nSee {file_name[0:len(file_name)-4]}.jpg in /processed/fit visualization for more detail.\nA={A}, B={B}, C={C}, D={D}, E={E}")

    projected_img = []
    normals_x = []
    normals_y = []
    tangents_x =[]
    tangents_y = []

    for i in range(len(fit[0])):
        projection = project_1D.project_one(fit[0][i], fit[1][i], coeff)
        temp = []
        normals_x.append(projection[2][0])
        normals_y.append(projection[2][1])
        tangents_x.append(projection[3][0])
        tangents_y.append(projection[3][1])
        for j in range(len(projection[0])):
            try:
                pixel = img[projection[1][j],projection[0][j]]
                temp.append([pixel[0], pixel[1], pixel[2]])
            except IndexError:
                temp.append([255,255,255])
        projected_img.append(temp)
        ax.plot(projection[0], projection[1], '.-y', label="projection")

    # sets up a data frame to store projection data
    df_project_data = pd.DataFrame()
    df_project_data["arclength loc"] = range(len(fit[0]))
    df_project_data["x_2D"] = fit[0]
    df_project_data["y_2D"] = fit[1]
    df_project_data["tangent_x"] = tangents_x
    df_project_data["tangent_y"] = tangents_y
    df_project_data["normal_x"] = normals_x
    df_project_data["normal_y"] = normals_y

    target = os.path.join(current_dir,"processed", "projection data", f"{img_name}.csv")
    df_project_data.to_csv(target)

    target = os.path.join(current_dir,"processed", "projection")
    os.chdir(target)
    projected_img_t = cv2.transpose(np.array(projected_img))
    cv2.imwrite(file_name, projected_img_t)
    os.chdir(current_dir)

    # intensity analysis; uncomment to perform
    # analyze_projection.avg_intensity(file_name, projected_img)

    target = os.path.join(current_dir,"processed", "projection sampling")
    os.chdir(target)
    fig.savefig(file_name)
    os.chdir(current_dir)


    if CONFIG.FILTER == Filter.MANUAL:
        
        teethdf = pd.DataFrame()
        closest_proj_indecies = []
        closest_ys = []
        side = [50 for _ in range(len(x))]

        for tooth_index in range(len(x)):
            proj_data = project_1D.proj_data(x[tooth_index], y[tooth_index],coeff) #return (x, distance)
            closest_x = proj_data[0] 
            closest_y = proj_data[1]
            closest_proj_indecies.append(np.argmin([abs(i-closest_x) for i in fit[0]]))
            closest_ys.append(CONFIG.SAMPLING_WIDTH+closest_y)

        teethdf["x"] = closest_proj_indecies
        teethdf["y"] = closest_ys
        teethdf["w"] = side
        teethdf["h"] = side
        df_manual = pd.read_csv(os.path.join("processed", "manual data", f"{img_name}.csv"))
        center_ind = cross_prod_center(coeff, df)

        t = df_manual.index[df_manual["type"]=="Tooth.CENTER_T"].to_numpy()
        g = df_manual.index[df_manual["type"]=="Tooth.CENTER_G"].to_numpy()
        assert len(t)+len(g) < 2, f"more than one center tooth or gap found in {img_name}.csv"
        # if center tooth doesn't exist 
        if len(t)+len(g) == 0:
            if df_manual["type"][center_ind] == "Tooth.TOOTH":
                df_manual["type"][center_ind] = "Tooth.CENTER_T"
            elif df_manual["type"][center_ind] == "Tooth.GAP":
                df_manual["type"][center_ind] = "Tooth.CENTER_G"
            df_manual.to_csv(os.path.join("processed", "manual data", f"{img_name}.csv"))
        else:
            if len(t) > 0 and center_ind != t[0] or len(g) > 0 and center_ind != g[0]:
                    print(f"Alternative center index found: {center_ind}; please ensure the current center index is correct")
        teethdf["type"] = df_manual["type"]
        
        GUI.plot_teeth(file_name, img_name, Match.ONE_D, teethdf)


def equidistant_set(start, end, coeff):

    # equidistant in x
    x = np.linspace(start, end, num=int(end-start)+1)
    quadratic = coeff[2]*np.ones(len(x))
    linear = coeff[1]*x+coeff[4]
    constant = coeff[0]*x**2+coeff[3]*x-1

    #conic: Ax**2+Bxy+Cy**2+Dx+Ey-1=0
    #circle parameterization: x = prev_x + cos(t); y = prev_y + sin(t)
    #intersection: plug

    start_roots = [r for r in np.roots([quadratic[0], linear[0], constant[0]]) if r >= 0]
    start_y = min(start_roots)

    result = ([],[])
    prev_x = start
    prev_y = start_y

    while prev_x < end:
        r1 = fsolve(func, np.pi/4, [prev_x, prev_y, coeff])
        r2 = fsolve(func, -np.pi/4, [prev_x, prev_y, coeff])

        if np.cos(r1[0])>0:
            curr_x, curr_y = prev_x+np.cos(r1[0]), prev_y+np.sin(r1[0])
        elif np.cos(r2[0])>0:
            curr_x, curr_y = prev_x+np.cos(r2[0]), prev_y+np.sin(r2[0])
        else:
            curr_1x, curr_1y = prev_x+np.cos(r1[0]), prev_y+np.sin(r1[0])
            curr_2x, curr_2y = prev_x+np.cos(r2[0]), prev_y+np.sin(r2[0])
            if r1[0] < np.pi/2 and r1[0] > -np.pi/2:
                raise RuntimeError(f"Equidistant Points Error: r1x_0 = {prev_x}, r1y_0 = {prev_y}, r1x_1={curr_1x}, r1x_2={curr_1y}\nr2x_0 = {prev_x}, r2y_0 = {prev_y}, r2x_1={curr_2x}, r2x_2={curr_2y}\n(A,B,C,D,E) = {coeff}\n Try readjusting some data through GUI")
        result[0].append(curr_x)
        result[1].append(curr_y)
        prev_x = curr_x
        prev_y = curr_y

    return result
    

def func(t, args):
    (prev_x, prev_y, coeff) = args
    (A,B,C,D,E) = coeff
    return A*(prev_x+np.cos(t))**2+B*(prev_x+np.cos(t))*(prev_y+np.sin(t))+C*(prev_y+np.sin(t))**2+D*(prev_x+np.cos(t))+E*(prev_y+np.sin(t))-1


def plot_hyperbola_linear(start, end, coeff):
    # equidistant in x
    x = np.linspace(start, end, num=int(end-start)+1)
    quadratic = coeff[2]*np.ones(len(x))
    linear = coeff[1]*x+coeff[4]
    constant = coeff[0]*x**2+coeff[3]*x-1
    result = ([],[])
    for i in range(len(x)):
        roots = np.roots([quadratic[i], linear[i], constant[i]])
        for r in roots:
            if r >= 0:
                result[0].append(x[i])
                result[1].append(r)
    return result
