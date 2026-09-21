# src/extension.py
import os
import sys

# Asegura que src/ esté en el path para que PSO.py sea importable
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import knime.extension as knext
import pandas as pd

from PSO import PSOFeatureSelection


@knext.node(
    name="PSO Feature Selector",
    node_type=knext.NodeType.LEARNER,
    icon_path="../../icons/icon.png",
    category="/community/PSO",
)
@knext.input_table(
    name="Input Table",
    description="Table containing the X features and the target Y column",
)
@knext.output_table(
    name="Selected Features",
    description="Table containing the selected feature columns",
)
class PSONode:
    """Selecciona características con Particle Swarm Optimization.

    Tarea: **Regresión** (usa RandomForestRegressor internamente).
    """
    y_column = knext.ColumnParameter(
        label="Target column (Y)",
        description="Select the column to use as the target variable",
    )

    swarm_size = knext.IntParameter(
        label="Tamaño del enjambre",
        description="Número de partículas del PSO",
        default_value=15,
        min_value=5,
    )

    iterations = knext.IntParameter(
        label="Iteraciones",
        description="Número de iteraciones del PSO",
        default_value=15,
        min_value=1,
    )

    w = knext.DoubleParameter(
        label="Inercia (w)",
        description="Coeficiente de inercia",
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
    )

    c1 = knext.DoubleParameter(
        label="Cognitivo (c1)",
        description="Peso del mejor personal",
        default_value=1.5,
        min_value=0.0,
        max_value=4.0,
    )

    c2 = knext.DoubleParameter(
        label="Social (c2)",
        description="Peso del mejor global",
        default_value=1.5,
        min_value=0.0,
        max_value=4.0,
    )

    cv_folds = knext.IntParameter(
        label="Folds de validación cruzada",
        description="Número de particiones para cross_val_score",
        default_value=3,
        min_value=2,
    )

    max_features = knext.IntParameter(
        label="Máximo de descriptores",
        description=("Número máximo de descriptores a seleccionar. "
                     "Usa 0 para modo umbral (sin límite)."),
        default_value=5,
        min_value=0,
    )

    threshold = knext.DoubleParameter(
        label="Umbral (solo modo sin límite)",
        description="Umbral para binarizar la posición si max_features=0",
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
    )

    random_seed = knext.IntParameter(
        label="Semilla aleatoria",
        description="Semilla para reproducibilidad",
        default_value=42,
        min_value=0,
    )

    def configure(self, configure_context, input_schema_1, input_schema_2):
        return None

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()

        y = df[self.y_column]
        X = df.drop(columns=[self.y_column])

        max_features = self.max_features if self.max_features > 0 else None

        pso = PSOFeatureSelection(
            swarm_size=self.swarm_size,
            iterations=self.iterations,
            w=self.w,
            c1=self.c1,
            c2=self.c2,
            cv_folds=self.cv_folds,
            scoring='r2',
            max_features=max_features,
            random_seed=self.random_seed,
            threshold=self.threshold,
        )

        best_mask, best_score, best_features, history = pso.optimize(X, y)

        if len(best_features) == 0:
            raise ValueError(
                "PSO no seleccionó ninguna característica. "
                "Prueba con más iteraciones o revisa la tabla X."
            )

        X_selected = df[best_features]
        return knext.Table.from_pandas(X_selected)