"""Run with python -m unittest discover -s tests -v."""
import contextlib
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.train_logreg import run_pipeline
from src.data import preprocess
from src.data.artifact import DataArtifact
from src.data.modelprep import ModelPrep
from src.training.train.logreg_train import train_logreg
from src.training.train_conf.logreg_config import logreg_train_config

rw = importlib.import_module("src.data.read_writer")
train_logreg_script = importlib.import_module("scripts.train_logreg")

class LogregTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # ModelPrep writes under DATA_ROOT; the script reads under PROJECT_ROOT/data. Point both at the temp dir.
        for target, attr, value in ((rw, "DATA_ROOT", self.root / "data"), (train_logreg_script, "PROJECT_ROOT", self.root)):
            patcher = patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.samples = [f"sample{i}" for i in range(6)]
        self.tables = {
            "OSD_105_obs.csv": pd.DataFrame({"sample_id": self.samples, "arm": ["Flight"] * 3 + ["Ground.Control"] * 3}),
            "OSD_105_var.csv": pd.DataFrame({"ensembl_id": ["g2", "g1"]}),
            "OSD_105_expressioncount.csv": pd.DataFrame({"gene": ["g1", "g2"], **{sample: [i + 1, i + 2] for i, sample in enumerate(self.samples)}}),
        }

    # def test_pipeline_uses_real_io_and_preserves_metadata(self):
    #     for filename, table in self.tables.items():
    #         table.to_csv(self.root / filename, index=False)
    #     output = run_pipeline("OSD-105", data_dir=self.root, config=logreg_train_config(cv_folds=3))
    #     self.assertEqual(output.stage, "predictions")
    #     self.assertEqual(output.metadata["source"], "nasa")
    #     self.assertEqual(output.metadata["model_type"], "logreg")
    #     self.assertEqual(output.path, self.root / "outputs/predictions/osd-105/logreg")
    #     result = pd.read_pickle(output.path / "probe.pkl")
    #     self.assertEqual(result["feature_names"], ["g2", "g1"])
    #     self.assertEqual(len(result["metrics"]["cv_scores"]), 3)
    #     self.assertEqual(result["model"].n_features_in_, 2)

    def test_script_reads_what_modelprep_writes_and_reports_cv(self):
        interim = self.root / "interim_in"
        interim.mkdir()
        for filename, table in self.tables.items():
            table.to_csv(interim / filename, index=False)
        artifact = DataArtifact(name="OSD-105", stage="interim", path=interim, metadata={})
        processed = ModelPrep.from_interim(artifact, model_type="logreg", input_filename=tuple(self.tables), output_filename="model_data.pkl")
        self.assertEqual(processed.stage, "processed")
        self.assertEqual(processed.path, self.root / "data/processed/logreg/osd-105")

        for variant in ("OSD-105", "osd-105"):
            with self.subTest(dataset=variant):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertIsNone(run_pipeline(variant, config=logreg_train_config(cv_folds=3)))
                self.assertIn("CV AUROC", out.getvalue())
                self.assertIn("6 samples x 2 genes", out.getvalue())

    def test_alignment_and_structural_protein_missingness(self):
        tables = dict(self.tables)
        tables["OSD_105_protein_matrix.csv"] = pd.DataFrame({"gene_symbol": ["gene"], "tmt_plex": ["plex_1"], **{sample: [np.nan if i == 0 else i] for i, sample in enumerate(reversed(self.samples))}})
        adata = preprocess.prep_logreg(tables)
        np.testing.assert_array_equal(adata.X[0], [2, 1])
        self.assertEqual(adata.obsm["protein"][0, 0], 5)
        self.assertTrue(np.isnan(adata.obsm["protein"][-1, 0]))

    def test_invalid_labels_and_small_classes_are_rejected(self):
        adata = preprocess.prep_logreg(self.tables)
        for label in (None, "unexpected"):
            with self.subTest(label=label):
                invalid = adata.copy()
                invalid.obs.loc[self.samples[0], "arm"] = label
                with self.assertRaises(ValueError):
                    train_logreg(invalid)
        with self.assertRaises(ValueError):
            train_logreg(adata, logreg_train_config(cv_folds=4))

    def test_multi_read_validates_all_names_before_loading(self):
        with self.assertRaises(ValueError):
            rw.read_writer("model", "osd-105", action="read", filename=("valid.csv", "../escape.csv"), model_name="logreg")


if __name__ == "__main__":
    unittest.main(verbosity=2)
