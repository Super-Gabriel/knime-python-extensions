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
    icon_path="../icons/icon.png",
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
        label="Swarm size",
        description="Number of PSO particles",
        default_value=15,
        min_value=5,
    )

    iterations = knext.IntParameter(
        label="Iterations",
        description="Number of PSO iterations",
        default_value=15,
        min_value=1,
    )

    w = knext.DoubleParameter(
        label="Inertia (w)",
        description="Inertia coefficient",
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
    )

    c1 = knext.DoubleParameter(
        label="Cognitive coefficient (c1)",
        description="Weight of the personal best",
        default_value=1.5,
        min_value=0.0,
        max_value=4.0,
    )

    c2 = knext.DoubleParameter(
        label="Social coefficient (c2)",
        description="Weight of the global best",
        default_value=1.5,
        min_value=0.0,
        max_value=4.0,
    )

    cv_folds = knext.IntParameter(
        label="Cross-validation folds",
        description="Number of partitions for cross_val_score",
        default_value=3,
        min_value=2,
    )

    max_features = knext.IntParameter(
        label="Maximum number of features",
        description=("Maximum number of features to select. "
                     "Use 0 for threshold mode (no limit)."),
        default_value=5,
        min_value=0,
    )

    threshold = knext.DoubleParameter(
        label="Threshold (unlimited mode only)",
        description="Threshold for binarizing the position when max_features=0",
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
    )

    random_seed = knext.IntParameter(
        label="Random seed",
        description="Seed for reproducibility",
        default_value=42,
        min_value=0,
    )

    def configure(self, configure_context, input_schema):
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