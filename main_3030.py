from ilp import ilp
from preprocessing import preprocess
from reference_values import ref_val

countries = ["Afghanistan", "BosniaHerzegovina", "Burundi", "Eritrea", "Georgia", "Haiti", "Italy",
             "Kyrgyzstan", "LaoPDR", "Lebanon", "Lesotho", "Montenegro", "Nepal", "Peru", "RepublicofKorea", "Rwanda",
             "Tajikistan", "Vietnam"]
scenarios = ["biodiv","ES", "biodiv_ES", "biodiv_simple_pressure", "biodiv_pressure", "ES_pressure", "biodiv_ES_pressure"]

scenario = "biodiv"

for country in countries:
    print("##############   " + country + "   ***   " + scenario + "   ##############")
    preprocess(country, scenario)

    ref_val(country, scenario)

    ilp(country, scenario)
