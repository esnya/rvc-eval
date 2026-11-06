import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import git
import numpy as np
import torch
from jinja2 import Environment
from lxml import etree

from rvc_eval.vc_infer_pipeline import VC


class TinyFeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 256)
        self.final_proj = torch.nn.Identity()

    def extract_features(self, source, padding_mask, output_layer):
        frames = source.unfold(-1, 320, 320).mean(-1).unsqueeze(-1)
        return (self.projection(frames),)


class TinyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(256, 1)

    def infer(self, phone, lengths, pitch, pitchf, sid):
        audio = self.projection(phone).transpose(1, 2).repeat_interleave(160, dim=-1)
        return (audio,)


class TestNativeSmoke(unittest.TestCase):
    def test_pitch_and_tensor_pipeline(self):
        phase = 2 * np.pi * 220 * np.arange(16000) / 16000
        audio = sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 9))
        audio = (audio / np.max(np.abs(audio))).astype(np.float32)
        vc = VC(16000, torch.device("cpu"), False, 0)
        for method in ("pm", "harvest"):
            with self.subTest(method=method):
                coarse, pitch = vc.get_f0(audio, 100, 0, method)
                self.assertTrue(np.isfinite(pitch).all())
                voiced = pitch[pitch > 0]
                self.assertGreater(voiced.size, 0)
                self.assertAlmostEqual(float(np.median(voiced)), 220, delta=20)
                self.assertTrue(((coarse >= 1) & (coarse <= 255)).all())
                output = vc.pipeline(
                    TinyFeatureModel().eval(), TinyGenerator().eval(),
                    0, audio, 0, method,
                )
                self.assertEqual(output.shape, (16000,))
                self.assertTrue(torch.isfinite(output).all())

    def test_lxml_does_not_resolve_external_entities_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "entity.txt"
            secret.write_text("local-entity-must-not-expand", encoding="utf-8")
            xml = (
                '<!DOCTYPE root [<!ENTITY local SYSTEM "'
                + secret.as_uri() + '">]><root>&local;</root>'
            ).encode("utf-8")
            for parse in (
                lambda: [etree.fromstring(xml, parser=etree.ETCompatXMLParser())],
                lambda: [element for _, element in etree.iterparse(io.BytesIO(xml))],
            ):
                try:
                    elements = parse()
                except etree.XMLSyntaxError as error:
                    self.assertIn("Entity 'local' not defined", str(error))
                else:
                    for element in elements:
                        self.assertNotIn(
                            "local-entity-must-not-expand", etree.tostring(element).decode()
                        )

    def test_gitpython_reads_local_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = git.Repo.init(directory)
            self.assertEqual(Path(repo.working_tree_dir).resolve(), Path(directory).resolve())
            self.assertFalse(repo.bare)
            repo.close()

    def test_gitpython_rejects_git_directory_before_cloning(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = git.Repo.init(directory)
            with patch.object(git.Submodule, "_clone_repo") as clone:
                with self.assertRaises(ValueError):
                    repo.create_submodule(
                        "invalid", ".git/hooks", "https://example.invalid/submodule.git"
                    )
                clone.assert_not_called()
            repo.close()

    def test_jinja_escaping(self):
        template = Environment(autoescape=True).from_string("{{ value }}")
        self.assertEqual(template.render(value="<tag>"), "&lt;tag&gt;")


if __name__ == "__main__":
    unittest.main()
