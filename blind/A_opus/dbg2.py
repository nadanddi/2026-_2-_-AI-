from common import *
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
A,S=load()
x=A[(A.gh=="F47")&A.day.isin([178,189])][["day","hr","out_temp","out_rad","in_temp","in_hum","in_co2","act_vent","act_heating","act_thermal","sub_temp","sub_ec"]]
print(x.to_string())
