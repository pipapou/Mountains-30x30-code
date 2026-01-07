import glob
import numpy as np
from pulp import *
from osgeo import gdal
import xml.etree.ElementTree as ET
import math
import os
import re
from fonctions_spatial import read_geotiff

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

def compute_ref_point(var, mode, vars_dict, Rasters, Mask, nb_protect, pourc_eco, X, Y, species_data):
    print(f"{var} | {mode}")
    model = LpProblem("RefPointCalc", LpMaximize if mode == 'opti' else LpMinimize)
    protected_pix = LpVariable.dicts("protected_pix", [(i, j) for i in range(X) for j in range(Y)], cat=LpBinary)

    model += lpSum([
        protected_pix[(i, j)] * Rasters[var][i, j] * vars_dict[var]
        for i in range(X) for j in range(Y)
    ])

    build_constraints(
        model, protected_pix, Rasters, Mask, pourc_eco, X, Y, nb_protect,
        species_data=species_data, vars_dict=vars_dict
    )
    print("*Solving*")
    model.solve(PULP_CBC_CMD())
    selected = np.array([
        protected_pix[(i, j)].varValue for i in range(X) for j in range(Y)
    ]).reshape((X, Y))
    return np.sum(selected * Rasters[var])

def ref_val(country, scenario):
    print("*Parameters initialization*")
    base_path = os.path.join("data", "countries", country)
    xml_path = f"data/countries/{country}/scenarios/{scenario}/results/parameters.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()

    X = int(root.find("X").text)
    Y = int(root.find("Y").text)
    nb_protect = int(root.find("nb_protect").text)
    pourc_eco = float(root.find("pourc_eco").text)

    vars_dict = {f.get("name"): float(f.get("weight")) for f in root.find("Factors").findall("Factor")}

    needed_vars = set(vars_dict.keys()) | {'ecoregions', 'PA'}
    Rasters = {}
    raster_dir = os.path.join(base_path, "rasters_5km-norm")
    for rast_path in glob.glob(os.path.join(raster_dir, "5km_*.tif")):
        var = os.path.basename(rast_path).replace("5km_", "").replace(".tif", "")
        if var in needed_vars:
            Rasters[var] = np.array(gdal.Open(rast_path).ReadAsArray())

    Mask, _ = read_geotiff(f"data/countries/{country}/scenarios/{scenario}/results/temp/Masked_data_5km.tif")

    species_data = []

    if vars_dict.get("biodiversity", 0) > 0:
        species_files = glob.glob(f"{raster_dir}/rasters_species/5km_*.tif")
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

    for var in vars_dict:
        for mode in ['opti', 'worst']:
            value_score = compute_ref_point(
                var=var,
                mode=mode,
                vars_dict=vars_dict,
                Rasters=Rasters,
                Mask=Mask,
                nb_protect=nb_protect,
                pourc_eco=pourc_eco,
                X=X,
                Y=Y,
                species_data=species_data
            )
            elt = ET.Element(f"{mode}_{var}")
            elt.text = str(value_score)
            root.append(elt)

    tree.write(xml_path)
