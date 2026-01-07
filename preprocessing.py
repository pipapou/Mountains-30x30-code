import glob
import numpy as np
from pulp import *
from osgeo import gdal
import xml.etree.ElementTree as ET
from fonctions_spatial import read_geotiff, write_geotiff
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import re
import os

def preprocess(country, scenario):
    print("*Parameters initialization*")

    # Import all rasters
    Rasters = {}
    for rast in glob.glob(f"./data/countries/{country}/rasters_5km-norm/5km*"):
        temp = gdal.Open(rast)
        name = re.search(r'5km_(.+?)\.tif', rast).group(1)
        Rasters[name] = np.array(temp.ReadAsArray())

    X, Y = Rasters['binary'].shape

    # Mask irrelevant pixels
    to_mask = [
        (Rasters['binary'] == 0),
        (Rasters['MODIS_IGBP_landuse'] == 12),
        (Rasters['MODIS_IGBP_landuse'] == 13),
    ]
    Mask = np.any(to_mask, axis=0).astype(int)

    # Area to protect
    rat_protect = 0.3
    per_protect = rat_protect - Rasters['PA'][Rasters['binary'] == 1].sum() / Rasters['binary'].sum()
    nb_protect = round(per_protect * Rasters['binary'].sum())
    pourc_eco = 0.17

    # FACTORS section
    vars_dict = {
        "biodiversity": 1, # IRREMP + richness
        #"realised_carbon_value": 1/6,
        #"realised_water_value": 1/6,
        #"realised_NBTourism_value": 1/6,
        #"relpressure": -1/2
    }

    print("pixels img:", X * Y,
          "\npixels country:", Rasters['binary'].sum(),
          "\npixels masked:", Mask.sum() + np.sum(Rasters['PA']),
          "\npixels selected:", X * Y - Mask.sum() - np.sum(Rasters['PA']),
          "\npixels protected:", Rasters['PA'][Rasters['binary'] == 1].sum(),
          "\npixels to protect:", nb_protect)

    ####################### XML INIT #######################

    root = ET.Element("Parameters")

    ET.SubElement(root, "X").text = str(X)
    ET.SubElement(root, "Y").text = str(Y)
    ET.SubElement(root, "nb_protect").text = str(nb_protect)
    ET.SubElement(root, "pourc_eco").text = str(pourc_eco)

    factors_element = ET.SubElement(root, "Factors")
    for var, weight in vars_dict.items():
        factor = ET.SubElement(factors_element, "Factor")
        factor.set("name", var)
        factor.set("weight", str(weight))

    # Write XML
    xml_path = f"data/countries/{country}/scenarios/{scenario}/results/parameters.xml"
    tree = ET.ElementTree(root)
    tree.write(xml_path)

    # Save mask raster
    model_tif_arr, model_tif_ds = read_geotiff(f"./data/countries/{country}/rasters_5km-norm/5km_binary.tif")
    write_geotiff(f"data/countries/{country}/scenarios/{scenario}/results/temp/Masked_data_5km.tif", Mask, model_tif_ds)

    ####################### MAP OUTPUT #######################

    result = Rasters['binary'].copy() - 1
    result[result < 0] = np.nan
    result[Rasters['PA'] == 1] = 1

    plt.figure(figsize=(15, 8))
    cmap = ListedColormap(["#30678d", "#35b778"])
    im = plt.imshow(result, cmap=cmap)
    labels = ["Unprotected", "Current PAs"]
    patches = [mpatches.Patch(color=cmap(i), label=label) for i, label in enumerate(labels)]
    plt.axis('off')
    plt.legend(handles=patches, bbox_to_anchor=(0.9, 1), loc=2, fontsize=10)
    plt.title("Graphical_input_pa", multialignment='left')
    plt.savefig(f"data/countries/{country}/scenarios/{scenario}/results/Graphical_input_pa.png", dpi=300, bbox_inches='tight')
    plt.close()
