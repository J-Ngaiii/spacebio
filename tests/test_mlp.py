"""MLP pipeline tests: prep, model, training and prep-to-train paths.
Run with python -m unittest tests.test_mlp -v. All file I/O is redirected into temp directories."""
import ast
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import torch

from src.config.config import MODELS
from src.data import preprocess
from src.data.artifact import DataArtifact
from src.data.modelprep import ModelPrep
from src.models.simple_mlp import MLP
from src.training.train.mlp_train import train_mlp
from src.training.train_conf.logreg_config import logreg_train_config
from src.training.train_conf.mlp_config import mlp_train_config

rw = importlib.import_module("src.data.read_writer")
train_mlp_script = importlib.import_module("scripts.train_mlp")
train_logreg_script = importlib.import_module("scripts.train_logreg")

PY310_FILES = ["src/data/preprocess.py", "src/models/simple_mlp.py", "scripts/train_mlp.py", "scripts/train_logreg.py"]


def synthetic_tables(n_per_class=6, n_genes=20, seed=0):
    """Raw-count tables where the first 5 genes separate Flight from Ground."""
    rng = np.random.default_rng(seed)
    samples = [f"s{i}" for i in range(2 * n_per_class)]
    arms = ["Flight"] * n_per_class + ["Ground.Control"] * n_per_class
    genes = [f"g{j}" for j in range(n_genes)]
    counts = rng.poisson(50, size=(n_genes, len(samples))).astype(float)
    counts[:5, :n_per_class] *= 8
    return {
        "OSD-105_obs.csv": pd.DataFrame({"sample_id": samples, "arm": arms}),
        "OSD-105_var.csv": pd.DataFrame({"ensembl_id": genes}),
        "OSD-105_expression_counts.csv": pd.DataFrame(counts, columns=samples).assign(gene=genes)[["gene", *samples]],
    }


FAST_MLP = dict(epochs=15, hidden_dim=8, batch_size=4, cv_folds=3, wandb_mode="disabled")


class PrepSimpleMlpTests(unittest.TestCase):
    def test_registered_cleaner_resolves(self):
        self.assertIs(getattr(preprocess, MODELS["simple-mlp"]["cleaner"]), preprocess.prep_simple_mlp)

    def test_matches_prep_logreg_and_keeps_raw_counts(self):
        tables = synthetic_tables()
        a, b = preprocess.prep_simple_mlp(tables), preprocess.prep_logreg(tables)
        np.testing.assert_array_equal(a.X, b.X)
        pd.testing.assert_frame_equal(a.obs, b.obs)
        pd.testing.assert_frame_equal(a.var, b.var)
        raw = tables["OSD-105_expression_counts.csv"].set_index("gene").T.to_numpy()
        np.testing.assert_array_equal(a.X, raw)  # no log/normalization applied

    def test_forwards_kwargs(self):
        tables = synthetic_tables()
        samples = tables["OSD-105_obs.csv"]["sample_id"]
        tables["OSD-105_protein_matrix.csv"] = pd.DataFrame({"sym": ["p"], "plex": ["x"], **{s: [1.0] for s in samples}})
        self.assertIn("protein", preprocess.prep_simple_mlp(tables).obsm)
        self.assertNotIn("protein", preprocess.prep_simple_mlp(tables, include_protein=False).obsm)


class MlpModelTests(unittest.TestCase):
    def test_default_signature_still_works(self):
        model = MLP(10, 4, 1)
        self.assertEqual(model(torch.randn(3, 10)).shape, (3, 1))
        self.assertEqual(model.net[2].p, 0.0)

    def test_dropout_active_in_train_only(self):
        torch.manual_seed(0)
        model, x = MLP(10, 64, 1, dropout=0.5), torch.randn(8, 10)
        model.train()
        self.assertFalse(torch.equal(model(x), model(x)))
        model.eval()
        self.assertTrue(torch.equal(model(x), model(x)))

    def test_state_dict_roundtrip_from_architecture(self):
        result = train_mlp(preprocess.prep_simple_mlp(synthetic_tables()), mlp_train_config(dropout=0.3, **FAST_MLP))
        rebuilt = MLP(**result["architecture"])
        rebuilt.load_state_dict(result["state_dict"])  # strict: keys and shapes must match
        self.assertEqual(rebuilt.net[2].p, 0.3)


class TrainMlpTests(unittest.TestCase):
    def test_trains_with_and_without_dropout(self):
        adata = preprocess.prep_simple_mlp(synthetic_tables())
        for dropout in (0.0, 0.5):
            with self.subTest(dropout=dropout):
                # 15 epochs (~30 steps at lr=1e-3) leaves the model near-constant; it needs ~200 to learn
                metrics = train_mlp(adata, mlp_train_config(dropout=dropout, **{**FAST_MLP, "epochs": 200}))["metrics"]
                self.assertEqual(len(metrics["cv_scores"]), 3)
                self.assertTrue(all(0.0 <= s <= 1.0 for s in metrics["cv_scores"]))
                self.assertGreaterEqual(metrics["oof_auroc"], 0.8)  # signal is strong; model should find it

    def test_deterministic_for_fixed_seed(self):
        adata = preprocess.prep_simple_mlp(synthetic_tables())
        config = mlp_train_config(dropout=0.2, **FAST_MLP)
        self.assertEqual(train_mlp(adata, config)["metrics"], train_mlp(adata, config)["metrics"])

    def test_rejects_invalid_dropout(self):
        adata = preprocess.prep_simple_mlp(synthetic_tables())
        with self.assertRaises(ValueError):
            train_mlp(adata, mlp_train_config(dropout=1.0, **FAST_MLP))


class PrepToTrainPathTests(unittest.TestCase):
    """ModelPrep writes and the train scripts read the same file, regardless of name casing."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for target in (rw, train_mlp_script, train_logreg_script):
            attr = "DATA_ROOT" if target is rw else "PROJECT_ROOT"
            value = self.root / "data" if target is rw else self.root
            p = patch.object(target, attr, value)
            p.start()
            self.addCleanup(p.stop)
        self.interim = self.root / "interim_in"
        self.interim.mkdir()
        tables = synthetic_tables()
        for name, table in tables.items():
            table.to_csv(self.interim / name, index=False)
        self.files = tuple(tables)

    def prep(self, model_type, name):
        artifact = DataArtifact(name=name, stage="interim", path=self.interim, metadata={})
        return ModelPrep.from_interim(artifact, model_type=model_type, input_filename=self.files, output_filename="model_data.pkl")

    def test_mlp_script_reads_what_modelprep_writes(self):
        out = self.prep("simple-mlp", "OSD-105")
        self.assertEqual(out.path, self.root / "data/processed/simple-mlp/osd-105")
        for variant in ("OSD-105", "osd-105", " OSD-105 "):
            with self.subTest(dataset=variant):
                train_mlp_script.run_pipeline(variant, config=mlp_train_config(**FAST_MLP))

    def test_logreg_script_reads_what_modelprep_writes(self):
        out = self.prep("logreg", "OSD-105")
        self.assertEqual(out.path, self.root / "data/processed/logreg/osd-105")
        for variant in ("OSD-105", "osd-105", " OSD-105 "):
            with self.subTest(dataset=variant):
                train_logreg_script.run_pipeline(variant, config=logreg_train_config(cv_folds=3))

    def test_exact_lowercase_path_is_used(self):
        """Guards Linux/CI, where a case mismatch would be a missing file."""
        self.prep("simple-mlp", "OSD-105")
        opened = []
        real_open = Path.open
        def spy(self, *args, **kwargs):
            opened.append(self)
            return real_open(self, *args, **kwargs)
        with patch.object(Path, "open", spy):
            train_mlp_script.run_pipeline("OSD-105", config=mlp_train_config(**FAST_MLP))
        self.assertIn(self.root / "data/processed/simple-mlp/osd-105/model_data.pkl", opened)

    def test_missing_processed_data_raises(self):
        with self.assertRaises(FileNotFoundError):
            train_mlp_script.run_pipeline("OSD-105", config=mlp_train_config(**FAST_MLP))


class CompatibilityTests(unittest.TestCase):
    def test_files_parse_as_python_310(self):
        # Dockerfile uses 3.10, CI uses 3.11, setup.py requires <3.12
        for path in PY310_FILES:
            with self.subTest(path=path):
                ast.parse((REPO / path).read_text(encoding="utf-8"), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
