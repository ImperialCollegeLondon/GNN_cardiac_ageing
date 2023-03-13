reticulate::use_python("~/miniforge3/bin/python")
require(reticulate)

predict_from_python <- function(to_pred, columns, file_path) {
py_run_string("
def assign(obj, name):
    globals()[name] = obj
")

import_main()$assign(to_pred, "to_pred")
import_main()$assign(columns, "columns")
import_main()$assign(file_path, "file_path")

prediction <- py_run_string("
import pickle
import numpy as np
import pandas as pd
non_image_columns = [
'GeneticPC_1','GeneticPC_2','GeneticPC_3','GeneticPC_4','GeneticPC_5',
'GeneticPC_6','GeneticPC_7','GeneticPC_8','GeneticPC_9','GeneticPC_10',
'Sex','BSA',
'AAo_maxarea','AAo_minarea','Aao_dist','DAo_maxarea',
'DAo_minarea','Dao_dist', 'WT_Global_mm',
'RVEDV',
'RVESV','RVSV','RVEF',
,'LAV_max','LAV_min','LASV','LAEF','RAV_max',
'RAV_min','RASV','RAEF',
]
image_columns = [
'rad_2','rad_3','rad_4','rad_5','rad_6','rad_7','rad_8','rad_9',
'rad_10','rad_11','rad_12','rad_13','rad_14','rad_15','rad_16',
'rad_17','rad_18','rad_19','rad_20','rad_21','rad_22','rad_23',
'rad_24','rad_25','rad_26','rad_27','rad_28','rad_29','rad_30',
'rad_31','rad_32','rad_33','rad_34','rad_35','rad_36','rad_37',
'rad_38','rad_39','rad_40','rad_41','rad_42','rad_43','rad_44',
'rad_45','rad_46','rad_47','rad_48','rad_49','rad_50'
,'long_2','long_3','long_4','long_5','long_6','long_7','long_8',
'long_9','long_10','long_11','long_12','long_13','long_14',
'long_15','long_16','long_17','long_18','long_19','long_20',
'long_21','long_22','long_23','long_24','long_25','long_26',
'long_27','long_28','long_29','long_30','long_31','long_32',
'long_33','long_34','long_35','long_36','long_37','long_38',
'long_39','long_40','long_41','long_42','long_43','long_44',
'long_45','long_46','long_47','long_48','long_49','long_50',
,'LVEDV','LVESV','LVSV','LVEF','LVCO','LVM']
original_columns = image_columns+non_image_columns

columns = np.array(columns, copy=True)
assert len(original_columns) == len(columns)
to_pred = np.array(to_pred, copy=True)
to_pred = pd.DataFrame(to_pred, columns=original_columns)
to_pred = to_pred[columns].to_numpy()

with open(file_path, 'rb') as f:
    est = pickle.load(f)
prediction = est.predict(to_pred)
")$prediction

return(prediction)
}


columns = c(
'GeneticPC_1','GeneticPC_2','GeneticPC_3','GeneticPC_4','GeneticPC_5',
'GeneticPC_6','GeneticPC_7','GeneticPC_8','GeneticPC_9','GeneticPC_10',
'Sex','BSA',
'AAo_maxarea','AAo_minarea','Aao_dist','DAo_maxarea',
'DAo_minarea','Dao_dist', 'rad_2','rad_3','rad_4','rad_5','rad_6','rad_7','rad_8','rad_9',
'rad_10','rad_11','rad_12','rad_13','rad_14','rad_15','rad_16',
'rad_17','rad_18','rad_19','rad_20','rad_21','rad_22','rad_23',
'rad_24','rad_25','rad_26','rad_27','rad_28','rad_29','rad_30',
'rad_31','rad_32','rad_33','rad_34','rad_35','rad_36','rad_37',
'rad_38','rad_39','rad_40','rad_41','rad_42','rad_43','rad_44',
'rad_45','rad_46','rad_47','rad_48','rad_49','rad_50'
,'long_2','long_3','long_4','long_5','long_6','long_7','long_8',
'long_9','long_10','long_11','long_12','long_13','long_14',
'long_15','long_16','long_17','long_18','long_19','long_20',
'long_21','long_22','long_23','long_24','long_25','long_26',
'long_27','long_28','long_29','long_30','long_31','long_32',
'long_33','long_34','long_35','long_36','long_37','long_38',
'long_39','long_40','long_41','long_42','long_43','long_44',
'long_45','long_46','long_47','long_48','long_49','long_50',
'WT_Global_mm','LVEDV','LVESV','LVSV','LVEF','LVCO','LVM','RVEDV',
'RVESV','RVSV','RVEF','LAV_max','LAV_min','LASV','LAEF','RAV_max',
'RAV_min','RASV','RAEF')
to_pred <- matrix(runif(10*135), 10, 135)
prediction <- predict_from_python(to_pred, columns, '/home/minacio@isd.csc.mrc.ac.uk/minacio/cardiac_ageing/results/optuna_pkls/initial_test_catb/73.pkl')
print(prediction)
