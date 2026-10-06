import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config import MODELS, STANDARDIZERS
from src.data import preprocess
from src.data.artifact import DataArtifact
from src.data.dataprep import DataPrep
from src.data.modelprep import ModelPrep
from src.data.sources import nasa

rw = importlib.import_module("src.data.read_writer") # import the read_writer fun to test it

# Run from the repository root: python scripts/test.py
class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.storage = patch.object(rw, "DATA_ROOT", self.root / "data")
        self.storage.start()
        self.addCleanup(self.storage.stop)
        self.network = patch.object(nasa.requests, "get", side_effect=AssertionError("Unexpected network request"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def artifact(self, stage):
        path = self.root / stage
        path.mkdir(exist_ok=True)
        return DataArtifact(name="osdr-114", stage=stage, path=path, metadata={"source": "nasa"})

    def test_registry_functions_exist(self):
        for name in [*STANDARDIZERS.values(), *MODELS.values()]: # iterate through all standardizers and models that should be in preprocess.py
            with self.subTest(function=name): 
                # individually assert that the standardizers and models instantiated in the config.py file exist in callable form in preprocess.py
                # function=name, makes it so that the test report can show the function on which the test failed
                self.assertTrue(callable(getattr(preprocess, name))) 

    def test_raw_write_returns_artifact_and_saves_json(self):
        result = rw.read_writer("curl", " OSDR-114 ", data={"example": 1}, filename="metadata.json")
        self.assertIsInstance(result, DataArtifact)
        self.assertEqual(result.name, "osdr-114")
        self.assertEqual(result.stage, "raw")
        self.assertEqual(result.path, self.root / "data/raw/osdr-114")
        self.assertIn('"example": 1', (result.path / "metadata.json").read_text())

    def test_raw_write_rejects_unknown_dataset(self):
        with self.assertRaises(ValueError):
            rw.read_writer("curl", "unknown", data={}, filename="metadata.json")

    def test_raw_write_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            rw.read_writer("curl", "osdr-114", data={}, filename="../escape.json")
        self.assertFalse((self.root / "data").exists())

    def test_nasa_ingestion_uses_registered_accession_and_saves_metadata(self):
        response = Mock()
        response.json.return_value = {"accession": "OSD-114"}
        with patch.object(nasa.requests, "get", return_value=response) as get:
            result = nasa.pull_dataset(" OSDR-114 ")
        self.assertEqual(get.call_args.args[0], nasa._endpoint("dataset", accession="OSD-114"))
        response.raise_for_status.assert_called_once()
        self.assertTrue((result.path / "metadata.json").is_file())
        self.assertEqual(result.metadata["accession"], "OSD-114")

    def test_nasa_failed_request_does_not_save_data(self):
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("HTTP failure")
        with patch.object(nasa.requests, "get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "HTTP failure"):
                nasa.pull_dataset("osdr-114")
        self.assertFalse((self.root / "data").exists())
        response.json.assert_not_called()

    def test_nasa_rejects_wrong_source_before_request(self):
        with self.assertRaises(ValueError):
            nasa.pull_dataset("archs4-a")

    def test_factories_reject_wrong_input_types(self):
        for factory, keyword in [(DataPrep, "osdr"), (ModelPrep, "pca")]: 
            # tuple unpacking: factory maps to DataPrep then ModelPrep, keyword to "osdr" then "pca"
            with self.subTest(factory=factory.__name__):
                with self.assertRaises(TypeError):
                    factory.from_pretrained("osdr-114", keyword) # first arg should be a DataArtifact

    def test_factories_reject_wrong_stages(self):
        for factory, stage, keyword in [(DataPrep, "interim", "osdr"), (ModelPrep, "raw", "pca")]:
            with self.subTest(factory=factory.__name__):
                with self.assertRaises(ValueError):
                    factory.from_pretrained(self.artifact(stage), keyword) # inputting an artifact with the wrong stage should error
                    # DataPrep only takes in artifacts in the "raw" stage
                    # ModelPrep only takes in artifacts in the "interim" stage

    def test_factories_reject_unknown_transformations(self):
        for factory, stage in [(DataPrep, "raw"), (ModelPrep, "interim")]:
            with self.subTest(factory=factory.__name__):
                with self.assertRaises(ValueError):
                    # no dataset and thus no preprocessing modules for "unkown" -> should error
                    factory.from_pretrained(self.artifact(stage), "unknown") 

if __name__ == "__main__":
    unittest.main(verbosity=2)