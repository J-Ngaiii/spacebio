"""PCA baseline tests: model, shared protocol and training.
Run with python -m unittest tests.test_pca -v."""
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import anndata as ad
import numpy as np

from src.config.config import MODELS, TRAINERS
from src.data import preprocess
from src.evaluation.evaluate import kfold_splits, load_protein_targets, pooled_pearson, zscore_stats
from src.models.pca import PCARidge
from src.training.train.pca_train import train_pca
from src.training.train_conf.pca_config import pca_train_config


def synthetic_adata(n_samples=12, n_genes=40, n_proteins=6, seed=0):
    """Raw counts where each protein follows one gene, measured in two plexes of half the samples each."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rng.uniform(20, 500, n_genes), size=(n_samples, n_genes)).astype(float)
    signal = np.log1p(counts[:, :n_proteins])
    protein = 2 ** (signal - signal.mean(axis=0) + rng.normal(0, 0.05, signal.shape))
    protein[: n_samples // 2, n_proteins // 2:] = np.nan
    protein[n_samples // 2:, : n_proteins // 2] = np.nan
    adata = ad.AnnData(X=counts)
    adata.var_names = [f"ENSG{j:011d}" for j in range(n_genes)]
    adata.obsm["protein"] = protein
    return adata


class PcaRidgeTests(unittest.TestCase):
    def test_components_capped_below_sample_count(self):
        X = np.random.default_rng(0).normal(size=(6, 30))
        model = PCARidge(n_components=50).fit(X, np.random.default_rng(1).normal(size=(6, 2)))
        self.assertEqual(model.n_components_, 5)
        self.assertEqual(model.predict(X).shape, (6, 2))

    def test_nan_targets_are_skipped_not_propagated(self):
        rng = np.random.default_rng(0)
        X, Y = rng.normal(size=(10, 20)), rng.normal(size=(10, 3))
        Y[:5, 0] = np.nan
        Y[:9, 2] = np.nan  # one measured sample: too few to fit
        model = PCARidge(n_components=3).fit(X, Y)
        self.assertTrue(np.isfinite(model.predict(X)).all())
        np.testing.assert_array_equal(model.coef_[2], 0)


class ProtocolTests(unittest.TestCase):
    def test_registered(self):
        self.assertIs(getattr(preprocess, MODELS["pca"]["cleaner"]), preprocess.prep_pca)
        self.assertEqual(TRAINERS["pca"]["train_function"], "train_pca")

    def test_folds_are_deterministic_and_disjoint(self):
        first, second = kfold_splits(12, 3, 42), kfold_splits(12, 3, 42)
        for (train_a, val_a), (train_b, val_b) in zip(first, second):
            np.testing.assert_array_equal(val_a, val_b)
            self.assertFalse(set(train_a) & set(val_a))

    def test_zscore_uses_training_rows_and_survives_unmeasured_proteins(self):
        Y = np.array([[1.0, np.nan], [3.0, np.nan], [100.0, 5.0]])
        mean, std = zscore_stats(Y[:2])
        np.testing.assert_allclose(mean, [2.0, 0.0])
        np.testing.assert_allclose(std, [1.0, 1.0])

    def test_log2_rejects_nonpositive_protein(self):
        adata = synthetic_adata()
        adata.obsm["protein"][0, 0] = 0.0
        with self.assertRaises(ValueError):
            load_protein_targets(adata, log2=True)

    def test_pooled_pearson_ignores_nan(self):
        targets = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 6.0]])
        self.assertAlmostEqual(pooled_pearson(targets * 2, targets), 1.0)


class TrainPcaTests(unittest.TestCase):
    def test_learns_signal_with_finite_fold_scores(self):
        result = train_pca(synthetic_adata(n_samples=24), pca_train_config(n_components=10, wandb_mode="disabled"))
        metrics = result["metrics"]
        self.assertEqual(len(metrics["cv_scores"]), 3)
        self.assertTrue(np.isfinite(metrics["cv_scores"]).all())
        self.assertGreater(metrics["oof_mean_pearson"], 0.3)

    def test_records_folds_and_predictions(self):
        adata = synthetic_adata()
        result = train_pca(adata, pca_train_config(wandb_mode="disabled"))
        validated = sorted(s for fold in result["folds"] for s in fold["validation_sample_ids"])
        self.assertEqual(validated, sorted(adata.obs_names))
        self.assertEqual(np.asarray(result["oof_predictions"]).shape, adata.obsm["protein"].shape)


if __name__ == "__main__":
    unittest.main()
