### fast-mRMR ###
# Basado el pseudo-código tomado del trabajo de:
# [Ramírez-Gallego, Sergio. et al.](https://doi.org/10.1002/int.21833) 
# [Jorge Hermo et al.](https://doi.org/10.1016/j.ins.2024.120609)
#
#################


import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def fast_mrmr(
    X,
    y,
    num_features_wanted,
    n_neighbors=3,
    random_state=42,
    target_type="Continuous",
):
    #X: matriz de variables
    #y: Actividades
    #num_features_wanted: num. final de caracteristicas
    
    # 1: INPUT: candidates (X.columns), numFeaturesWanted
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    candidates = list(X.columns)
    # 3: selectedFeatures = ();
    selected_features = []

    if target_type == "Continuous":
        mutual_information = mutual_info_regression
    elif target_type == "Categorical":
        mutual_information = mutual_info_classif
    else:
        raise ValueError("target_type must be 'Continuous' or 'Categorical'")

    # --- [Líneas 4 a 7 del Pseudo-código] ---
    # 4: for each feature f in candidates do
    # 5: relevancesVector[f] = mutualInfo(f, class);
    # Usamos mutual_info_regression para capturar relaciones continuas
    relevances = mutual_information( 
        X,
        y,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    relevances_dict = dict(zip(candidates, relevances))

    # 6: accumulateRedundancy[f] = 0;
    accumulated_redundancy = {f: 0.0 for f in candidates}
    # 7: end for

    # --- [Líneas 8 a 11 del Pseudo-código] ---
    # 8: selected = getMaxRelevance(relevancesVector);
    first_selected = max(relevances_dict, key=relevances_dict.get)

    # 9: lastFeatureSelected = selected;
    last_feature_selected = first_selected

    # 10: selectedFeatures.add(selected);
    selected_features.append(first_selected)

    # 11: candidates.remove(selected);
    candidates.remove(first_selected)

    # --- [Líneas 12 a 26 del Pseudo-código: Bucle Principal] ---
    # 12: while selectedFeatures.size() < numFeaturesWanted do
    while len(selected_features) < num_features_wanted and len(candidates) > 0:

        # 13: max_mrmr = 0;
        # (Usamos -infinito por si los resultados son negativos)
        max_mrmr = -float("inf")
        best_candidate = None

        # Pre-calculamos el MI de la última seleccionada contra todas las candidatas
        # para optimizar el rendimiento (es la esencia del "Fast")
        last_feat_data = X[last_feature_selected].values.reshape(-1, 1)
        mi_with_last = mutual_information(
            X[candidates],
            last_feat_data.ravel(),
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        mi_dict_last = dict(zip(candidates, mi_with_last))

        # 14: for each feature fc in candidates do
        for fc in candidates:
            # 15: relevance = relevancesVector[fc];
            relevance = relevances_dict[fc]

            # 16: accumulatedRedundancy[fc] += mutualInfo(fc, lastFeatureSelected);
            accumulated_redundancy[fc] += mi_dict_last[fc]

            # 17: redundancy = accumulatedRedundancy[fc]/selectedFeatures.size();
            redundancy = accumulated_redundancy[fc] / len(selected_features)

            # 18: mrmr = relevance - redundancy;
            mrmr_score = relevance - redundancy

            # 19: if mrmr > max_mrmr then
            if mrmr_score > max_mrmr:
                # 21: max_mrmr = mrmr;
                max_mrmr = mrmr_score
                # 20: lastFeatureSelected = fc; (Lo guardamos temporalmente en best_candidate)
                best_candidate = fc
            # 22: end if
        # 23: end for

        # 24: selectedFeatures.add(lastFeatureSelected);
        selected_features.append(best_candidate)

        # 25: candidates.remove(lastFeatureSelected);
        candidates.remove(best_candidate)

        # Actualizamos para la siguiente iteración del loop
        last_feature_selected = best_candidate

    # 26: end while
    # 2: OUTPUT: selectedFeatures
    return selected_features