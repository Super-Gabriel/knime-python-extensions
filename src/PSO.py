# src/PSO.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class PSOFeatureSelection:
    def __init__(self,
                 swarm_size=30,
                 iterations=50,
                 w=0.7,
                 c1=1.5,
                 c2=1.5,
                 cv_folds=5,
                 scoring='r2',
                 max_features=None,
                 min_features=1,
                 random_seed=42,
                 threshold=0.5):
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.max_features = max_features
        self.min_features = min_features
        self.random_seed = random_seed
        self.threshold = threshold
        np.random.seed(random_seed)

        self.X = None
        self.y = None
        self.feature_names = None
        self.n_features = None

        self.particle_pos = None
        self.particle_vel = None
        self.particle_best_pos = None
        self.particle_best_score = None
        self.global_best_pos = None
        self.global_best_score = -np.inf
        self.evaluated_candidates = []
        self._cv_cache = {}               # caché de evaluaciones
        self.history = []

    def _decode_position(self, pos):
        # si el dataset ya tiene <= max_features, no se puede reducir
        n = len(pos)
        if n == 0:
            return np.zeros(0, dtype=int)
        if self.max_features is not None and n <= self.max_features:
            return np.ones(n, dtype=int)

        mask = (pos >= self.threshold).astype(int)
        if (self.max_features is not None
                and self.max_features > 0
                and np.sum(mask) > self.max_features):
            top_k_idx = np.argsort(pos)[-self.max_features:]
            mask = np.zeros(n, dtype=int)
            mask[top_k_idx] = 1
        if np.sum(mask) == 0:
            mask[np.argmax(pos)] = 1
        return mask

    def _count_selected(self, mask):
        return int(np.sum(mask))

    def _record_candidate(self, mask, fold_scores):  # AJUSTE: ahora recibe mask
        fold_scores = np.asarray(fold_scores)
        score = float(np.mean(fold_scores))
        standard_error = (
            float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))
            if len(fold_scores) > 1 else 0.0
        )
        self.evaluated_candidates.append((score, standard_error, mask.copy()))

    def _get_preferred_candidate(self):
        # maneja min_features y lista vacía
        if not self.evaluated_candidates:
            return np.zeros(self.n_features, dtype=int), -np.inf

        best_score, best_standard_error, _ = max(
            self.evaluated_candidates, key=lambda candidate: candidate[0]
        )
        minimum_score = best_score - best_standard_error

        eligible = [
            (score, mask)
            for score, _, mask in self.evaluated_candidates
            if score >= minimum_score
            and self._count_selected(mask) >= self.min_features  # NUEVO
        ]

        if not eligible:  # fallback si min_features es demasiado exigente
            best = max(self.evaluated_candidates,
                       key=lambda candidate: candidate[0])
            return best[2].copy(), best[0]

        score, mask = min(
            eligible,
            key=lambda candidate: (
                self._count_selected(candidate[1]),
                -candidate[0],
            ),
        )
        return mask.copy(), score

    def _evaluate_cv_scores(self, pos):
        mask = self._decode_position(pos)
        n_selected = self._count_selected(mask)
        if n_selected == 0:
            return np.array([-1e9])

        # caché por máscara
        key = tuple(mask.tolist())
        if key in self._cv_cache:
            return self._cv_cache[key]

        X_subset = self.X[:, mask.astype(bool)]
        model = RandomForestRegressor(
            n_estimators=30, max_depth=10,
            random_state=self.random_seed, n_jobs=-1
        )
        scores = cross_val_score(
            model, X_subset, self.y,
            cv=self.cv_folds, scoring=self.scoring
        )
        self._cv_cache[key] = scores
        self._record_candidate(mask, scores)  # se registra solo si es nuevo
        return scores

    def _evaluate_fitness(self, pos):
        return float(np.mean(self._evaluate_cv_scores(pos)))

    def _initialize_swarm(self):
        n = self.n_features
        self.particle_pos = np.random.rand(self.swarm_size, n)
        self.particle_vel = np.random.uniform(-0.1, 0.1, (self.swarm_size, n))
        # la grabación de candidatos ahora ocurre dentro de _evaluate_cv_scores
        self.particle_best_score = np.array([
            np.mean(self._evaluate_cv_scores(pos)) for pos in self.particle_pos
        ])
        self.particle_best_pos = self.particle_pos.copy()
        best_idx = np.argmax(self.particle_best_score)
        self.global_best_pos = self.particle_pos[best_idx].copy()
        self.global_best_score = self.particle_best_score[best_idx]

    def optimize(self, X, y, feature_names=None):
        np.random.seed(self.random_seed)  # reproducibilidad por ejecución

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = ([f'Feature_{i}' for i in range(X.shape[1])]
                                  if feature_names is None else feature_names)

        self.X = np.array(X, dtype=float)
        self.y = np.array(y)
        self.n_features = self.X.shape[1]
        self.evaluated_candidates = []
        self._cv_cache = {}               # resetear caché
        self.history = []
        self.global_best_score = -np.inf

        print(f"[PSO] Descriptores: {self.n_features}, "
              f"Partículas: {self.swarm_size}, Iteraciones: {self.iterations}, "
              f"min_features={self.min_features}, max_features={self.max_features}")

        self._initialize_swarm()

        for it in range(self.iterations):
            for i in range(self.swarm_size):
                current_fold_scores = self._evaluate_cv_scores(
                    self.particle_pos[i]
                )
                current_score = float(np.mean(current_fold_scores))
                if current_score > self.particle_best_score[i]:
                    self.particle_best_score[i] = current_score
                    self.particle_best_pos[i] = self.particle_pos[i].copy()
                if current_score > self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_pos = self.particle_pos[i].copy()

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.n_features)
                r2 = np.random.rand(self.n_features)
                cognitive = self.c1 * r1 * (
                    self.particle_best_pos[i] - self.particle_pos[i]
                )
                social = self.c2 * r2 * (
                    self.global_best_pos - self.particle_pos[i]
                )
                self.particle_vel[i] = (
                    self.w * self.particle_vel[i] + cognitive + social
                )
                self.particle_pos[i] = self.particle_pos[i] + self.particle_vel[i]
                self.particle_pos[i] = np.clip(self.particle_pos[i], 0, 1)

            best_mask, preferred_score = self._get_preferred_candidate()
            self.history.append({
                'iteration': it + 1,
                'best_score': preferred_score,
                'n_selected': self._count_selected(best_mask),
            })

            if (it + 1) % 10 == 0 or it == 0:
                print(f"[PSO] Iter {it+1}/{self.iterations} "
                      f"- Mejor R²: {preferred_score:.4f} "
                      f"- Descriptores: {self._count_selected(best_mask)}")

        best_mask, best_score = self._get_preferred_candidate()
        best_features = [self.feature_names[i]
                         for i in range(self.n_features) if best_mask[i] == 1]

        print(f"[PSO] ¡Listo! R²={best_score:.4f}, "
              f"seleccionados={len(best_features)}")
        return best_mask, best_score, best_features, self.history