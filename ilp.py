import glob
import numpy as np
from pulp import *
from osgeo import gdal
from scipy.ndimage import label as nd_label
from fonctions_spatial import read_geotiff, write_geotiff
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import math
import os
import re

def build_constraints(model, protected_pix, Rasters, Mask, pourc_eco, X, Y, nb_protect, species_data=None, vars_dict=None):
    model += lpSum([protected_pix[(i, j)] for i in range(X) for j in range(Y)]) == nb_protect
    model += lpSum([protected_pix[(i, j)] * Mask[i, j] for i in range(X) for j in range(Y)]) == 0
    model += lpSum([protected_pix[(i, j)] * Rasters['PA'][i, j] for i in range(X) for j in range(Y)]) == 0

    if vars_dict.get("biodiversity", 0) > 0:
        print("*Ecosystems and species complementarity*: ", pourc_eco)

        for val in np.unique(Rasters['ecoregions'][(Rasters['ecoregions'] > 0) & (Mask == 0) & (Rasters['PA'] != 1)]):
        
            nb_no_agri_urban = np.sum((Rasters['ecoregions'] == val) & (Mask == 0))
            nb_protected_in_ecoreg = np.sum((Rasters['ecoregions'] == val) & (Rasters['PA'] == 1) & (Mask == 0))
            ecoreg_min = math.ceil((0.18 - nb_protected_in_ecoreg / nb_no_agri_urban) * nb_no_agri_urban)
            mask_temp = np.where(Rasters['ecoregions'] == val, 1, 0)
            model += lpSum([
                protected_pix[(i, j)] * mask_temp[i, j] for i in range(X) for j in range(Y)
            ]) >= ecoreg_min

        for _, _, aire_totale, S in species_data:
            indices = np.argwhere((S == 1) & (Mask == 0))
            protected_terms = [Rasters['PA'][i, j] + protected_pix[(i, j)] for i, j in indices]
            protected_sum = lpSum(protected_terms)
            model += protected_sum >= pourc_eco * aire_totale

def ilp(country, scenario):
    print("*Parameters initialization*")
    base_path = os.path.join("data", "countries", country)
    scenario_path = os.path.join(base_path, "scenarios", scenario)
    results_path = os.path.join(scenario_path, "results")

    tree = ET.parse(os.path.join(results_path, "parameters.xml"))
    root = tree.getroot()
    X = int(root.find("X").text)
    Y = int(root.find("Y").text)
    nb_protect = int(root.find("nb_protect").text)
    pourc_eco = float(root.find("pourc_eco").text)

    vars_dict = {f.get("name"): float(f.get("weight")) for f in root.find("Factors").findall("Factor")}
    print(vars_dict)

    # Variables à charger
    ref_pt = {}
    for var in vars_dict:
        for mode in ['opti', 'worst']:
            tag = f"{mode}_{var}"
            try:
                ref_pt[tag] = float(root.find(tag).text)
            except (AttributeError, TypeError, ValueError):
                ref_pt[tag] = 0.0
        ref_pt[f"diff_{var}"] = abs(ref_pt[f'opti_{var}'] - ref_pt[f'worst_{var}'])

    needed_vars = set(vars_dict.keys()) | {'PA', 'binary', 'ecoregions'}
    Rasters = {}
    raster_dir = os.path.join(base_path, "rasters_5km-norm")
    for rast_path in glob.glob(os.path.join(raster_dir, "5km_*.tif")):
        var = os.path.basename(rast_path).replace("5km_", "").replace(".tif", "")
        if var in needed_vars:
            Rasters[var] = np.array(gdal.Open(rast_path).ReadAsArray())

    mask_path = os.path.join(results_path, "temp", "Masked_data_5km.tif")
    Mask, _ = read_geotiff(mask_path)

    species_data = []

    # Lecture des données espèces quand biodiv activée
    if vars_dict.get("biodiversity", 0) > 0:
        species_dir = os.path.join(raster_dir, "rasters_species")
        species_files = glob.glob(os.path.join(species_dir, "5km_*.tif"))

        for path in species_files:
            filename = os.path.basename(path)
            match = re.search(r'5km_.*?_(\d+)_\d+\.tif', filename)
            if not match:
                continue
            ws = int(match.group(1))
            S = np.array(gdal.Open(path).ReadAsArray())
            aire_totale = np.sum(S[Mask == 0])
            if aire_totale > 0:
                species_data.append((filename, ws, aire_totale, S))

    print("*Model optimization building*")
    model = LpProblem("Conservation Planning", LpMaximize)
    protected_pix = LpVariable.dicts("protected_pix", [(i, j) for i in range(X) for j in range(Y)], cat=LpBinary)

    obj_terms = []
    for var in vars_dict:
        obj = vars_dict[var] * (
            lpSum([protected_pix[(i, j)] * Rasters[var][i, j] for i in range(X) for j in range(Y)])
            - ref_pt[f'worst_{var}']
        ) / ref_pt[f'diff_{var}'] if ref_pt[f'diff_{var}'] > 0 else 0
        obj_terms.append(obj)

    model += lpSum(obj_terms)

    build_constraints(
        model, protected_pix, Rasters, Mask, pourc_eco, X, Y, nb_protect,
        species_data=species_data, vars_dict=vars_dict
    )

    print("*Solving*")
    model.solve(PULP_CBC_CMD())

    print("*Graphic view and export*")
    result = Rasters['binary'].copy() - 1
    result[result < 0] = np.nan
    result[Rasters['PA'] == 1] = 1
    protected_np = np.array([
        protected_pix[(i, j)].varValue for i in range(X) for j in range(Y)
    ]).reshape((X, Y))
    result[protected_np == 1] = 2

    plt.figure(figsize=(15, 8))
    cmap = ListedColormap(["#30678d", "#35b778", "#FFD700"])
    im = plt.imshow(result, cmap=cmap)
    patches = [mpatches.Patch(color=im.cmap(im.norm(i)), label=label) for i, label in enumerate([
        "Unprotected", "Current PAs", "Potential PAs (Step 1)"
    ])]
    plt.axis('off')
    plt.legend(handles=patches, bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0., fontsize=10)
    plt.title("Graphical_output_ilp", multialignment='left')
    plt.savefig(os.path.join(results_path, "Graphical_output_ilp.png"), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

    protected_pix_np = np.array([protected_pix[(i, j)].varValue for i in range(X) for j in range(Y)]).reshape((X, Y))

    # result_t : 0 = hors zone, 1 = existant, 2 = nouveau
    result_t = (Rasters['binary'] - 1) + protected_pix_np + Rasters['PA']
    result_bin = protected_pix_np
    Mask_output = Mask + Rasters['PA'] + protected_pix_np - 1

    structure = np.ones((3, 3))
    labeled, ncomponents = nd_label(result_t, structure)

    ref_raster_path = os.path.join(raster_dir, "5km_binary.tif")
    model_tif_arr, model_tif_ds = read_geotiff(ref_raster_path)
    write_geotiff(os.path.join(results_path, "temp", "Protect_data_5km_ilp.tif"), result_t, model_tif_ds)
    write_geotiff(os.path.join(results_path, "temp", "Protect_bin_5km_ilp.tif"), result_bin, model_tif_ds)
    write_geotiff(os.path.join(results_path, "temp", "Masked_data_5km_ilp.tif"), Mask_output, model_tif_ds)
