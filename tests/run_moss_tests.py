"""Run with the ComfyUI Python environment: python tests/run_moss_tests.py."""

import importlib
import gc
import io
import json
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
package = types.ModuleType("realtime_lora_moss_tests")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package

import torch
import comfy.lora
import comfy.model_base
import comfy.model_patcher
import comfy.sd
import comfy.supported_models
from safetensors import safe_open
from safetensors.torch import load_file, save_file

moss = importlib.import_module(f"{package.__name__}.moss_lora")
selective = importlib.import_module(f"{package.__name__}.selective_lora_loader")
v1 = importlib.import_module(f"{package.__name__}.lora_analyzer")
v2 = importlib.import_module(f"{package.__name__}.lora_analyzer_v2")


class MOSSProfiles(unittest.TestCase):
    def test_formats_and_connected_profile(self):
        for count in (42, 43):
            parsed = selective._parse_block_weights_string(
                ",".join(["0.5"] * count), "MOSS_IMAGE")
            self.assertEqual(len(parsed), 43)
            self.assertEqual(parsed["layer_41"], (True, 0.5))
            self.assertEqual(parsed["other_weights"][1], 0.5 if count == 43 else 1.0)
        named = moss.parse_moss_profile("%default=0, layer14-27=1.3, layer_41=-0.5, other=0.2")
        self.assertEqual(named["layer_13"], (False, 0.0))
        self.assertEqual(named["layer_14"], (True, 1.3))
        self.assertEqual(named["layer_41"], (True, -0.5))
        self.assertEqual(named["other_weights"], (True, 0.2))

    def test_invalid_profiles(self):
        for text in ("1,2", "%layer42=1", "%layer9-2=1", "%foo=1", "%layer=nan",
                     "%default=inf", ",".join(["1"] * 42 + ["bad"]), "%"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                moss.parse_moss_profile(text)

    def test_key_mapping_and_detection(self):
        for key in ("diffusion_model.layers.41.proj.q.lora_up.weight",
                    "layers.41.proj.qkv.lora_B", "lora_unet_layers_41_proj_ffn_down.alpha"):
            self.assertEqual(moss.extract_moss_layer(key), "layer_41")
            self.assertTrue(moss.is_moss_lora([key]))
            self.assertEqual(v1._detect_architecture([key]), "MOSS_IMAGE")
            self.assertEqual(v2._detect_architecture_v2([key])[0], "MOSS_IMAGE")
        for key in ("diffusion_model.text_embedder.blocks.0.qkv.weight",
                    "text_encoders.qwen35_9b.transformer.model.layers.4.self_attn.q_proj.weight"):
            self.assertEqual(moss.extract_moss_layer(key), "other")
        self.assertEqual(selective._detect_architecture([
            "diffusion_model.blocks.49.attn.out_proj.lora_up.weight"]), "MINIMAX_H3")
        self.assertEqual(selective._detect_architecture([
            "diffusion_model.blocks.27.attn.wq.lora_up.weight"]), "KREA2")
        self.assertEqual(selective._detect_architecture([
            "diffusion_model.layers.29.attention.to_q.lora_up.weight"]), "ZIMAGE")

    def test_product_decompositions_scale_once(self):
        prefix = "lora_unet_layers_0_proj_q"
        generator = torch.Generator().manual_seed(20)
        for kind, suffixes in (
            ("lokr", ("lokr_w1_a", "lokr_w1_b", "lokr_w2")),
            ("hada", ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")),
        ):
            source = {f"{prefix}.{suffix}": torch.randn(2, 2, generator=generator) for suffix in suffixes}
            result = moss.filter_moss_lora(source, {"layer_0"}, {"layer_0": -0.5}, False)

            def delta(state):
                get = lambda name: state[f"{prefix}.{kind}_{name}"]
                first = get("w1_a") @ get("w1_b")
                return torch.kron(first, get("w2")) if kind == "lokr" else first * (get("w2_a") @ get("w2_b"))

            torch.testing.assert_close(delta(result), delta(source) * -0.5)


class MOSSNativeLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cpu = torch.device("cpu")
        cls.models = []
        for fused in (False, True):
            config = comfy.supported_models.MOSSImage({
                "image_model": "moss_image", "dim": 128, "ffn_dim": 128,
                "num_heads": 1, "num_layers": 42, "fused_qkv": fused,
            })
            model = comfy.model_base.MOSSImage(config, device=cpu)
            cls.models.append(comfy.model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu))

    @classmethod
    def tearDownClass(cls):
        cls.models.clear()

    def make_lora(self, style):
        generator = torch.Generator().manual_seed(10)
        result = {}
        for module in ("layers.0.proj.q", "layers.1.proj.q", "layers.41.proj.o", "video_embedder"):
            prefix = f"diffusion_model.{module}"
            if style == "training":
                prefix = "lora_unet_" + module.replace(".", "_")
            elif style == "bare":
                prefix = module
            up = ".lora_B" if style == "bare" else ".lora_up.weight"
            down = ".lora_A" if style == "bare" else ".lora_down.weight"
            result[prefix + up] = torch.randn(128, 2, generator=generator)
            result[prefix + down] = torch.randn(2, 128, generator=generator)
            result[prefix + ".alpha"] = torch.tensor(2.0)
        return result

    def deltas(self, model):
        state = model.model.state_dict()
        return {key: comfy.lora.calculate_weight(patches, torch.zeros_like(state[key]), key)
                for key, patches in model.patches.items()}

    def test_connected_analyzers_send_scores_to_rendered_widgets(self):
        node_executable = shutil.which("node")
        if node_executable is None:
            self.skipTest("Node.js is required for the Python-to-frontend integration check")
        payloads = {}
        with tempfile.TemporaryDirectory() as directory:
            source_path = str(Path(directory) / "source.safetensors")
            save_file(self.make_lora("native"), source_path)
            for node_class in (v1.LoRALoaderWithAnalysis, v2.LoRALoaderWithAnalysisV2):
                with patch.object(v1.folder_paths, "get_full_path", return_value=source_path), redirect_stdout(io.StringIO()):
                    output = node_class().load_lora_with_analysis(self.models[0], None, "source", 1.0, 0.0)
                self.assertIsInstance(output, dict)
                self.assertEqual(len(output["result"]), 5)
                self.assertEqual(output["ui"]["analysis_json"][0], output["result"][3])
                data = json.loads(output["ui"]["analysis_json"][0])
                self.assertEqual(data["architecture"], "MOSS_IMAGE")
                self.assertGreater(data["blocks"]["layer_1"]["score"], 0)
                payloads[node_class.__name__] = output["ui"]
                del output
                gc.collect()
            process = subprocess.run([node_executable, str(ROOT / "tests/moss_frontend.cjs"), "--analysis-payloads"],
                                     input=json.dumps(payloads), text=True, capture_output=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_pruning_signed_scaling_and_save_equivalence(self):
        for style in ("native", "training", "bare"):
            for model in self.models:
                with self.subTest(style=style, fused=model.model.diffusion_model.layers[0].fused_qkv), tempfile.TemporaryDirectory() as directory:
                    source = self.make_lora(style)
                    source_path = str(Path(directory) / "source.safetensors")
                    metadata = {"modelspec.architecture": "moss_image", "adapter_config": '{"r":2}', "test": "preserved"}
                    save_file(source, source_path, metadata=metadata)
                    profile = "%default=0, layer1=-0.5, layer41=1.3, other=0.2"
                    node = moss.MOSSImageSelectiveLoRALoader()
                    result = node.load_lora(model, None, "source", "0.9", "All Off",
                                            lora_path_opt=source_path, block_weights_string=profile,
                                            save_refined_lora=True, save_path=directory, save_filename="refined")
                    selective_out = result["result"][0]
                    self.assertEqual(len(result["result"]), 4)
                    self.assertEqual(len(result["result"][3].split(",")), 43)
                    saved_path = next(Path(directory).glob("refined_*.safetensors"))
                    saved = {key: tensor.clone() for key, tensor in load_file(str(saved_path)).items()}
                    self.assertFalse(any(moss.extract_moss_layer(key) == "layer_0" for key in saved))
                    with safe_open(str(saved_path), framework="pt") as file:
                        self.assertEqual(file.metadata()["adapter_config"], metadata["adapter_config"])
                        self.assertEqual(file.metadata()["test"], "preserved")
                    reloaded, _ = comfy.sd.load_lora_for_models(model, None, saved, 0.9, 0.0)
                    expected_model, _ = comfy.sd.load_lora_for_models(model, None, source, 0.9, 0.0)
                    expected = self.deltas(expected_model)
                    actual = self.deltas(selective_out)
                    self.assertEqual(actual.keys(), self.deltas(reloaded).keys())
                    for key, delta in actual.items():
                        factor = -0.5 if ".layers.1." in key else 1.3 if ".layers.41." in key else 0.2
                        torch.testing.assert_close(delta, expected[key] * factor)
                        self.assertTrue(torch.equal(delta, self.deltas(reloaded)[key]))
                    combined = v2.NODE_CLASS_MAPPINGS["MOSSImageAnalyzerSelectiveLoaderV2"]()
                    with patch.object(v2, "_save_save_paths_config"):
                        out = combined.load_analyze_and_filter(model, [], [], "source", 0.9, "All Off",
                                                              lora_path_opt=source_path, block_weights_string=profile,
                                                              save_refined_lora=True, save_path=directory, save_filename="combined")
                    combined_saved = load_file(str(next(Path(directory).glob("combined_*.safetensors"))))
                    self.assertEqual(saved.keys(), combined_saved.keys())
                    for key in saved:
                        self.assertTrue(torch.equal(saved[key], combined_saved[key]))
                    del combined_saved
                    self.assertEqual(out["result"][5], result["result"][3])
                    for key, delta in self.deltas(out["result"][0]).items():
                        self.assertTrue(torch.equal(delta, actual[key]))
                    analysis = json.loads(out["result"][4])
                    self.assertIn("layer_1", analysis["blocks"])
                    self.assertGreater(analysis["blocks"]["layer_1"]["score"], 0)
                    # Model patches retain memory-mapped source tensors on Windows.
                    del result, selective_out, out
                    gc.collect()

    def test_all_off_and_missing_path_output(self):
        node = moss.MOSSImageSelectiveLoRALoader()
        result = node.load_lora(self.models[0], None, "missing", 1.0, "All Off", lora_path_opt="missing")
        self.assertEqual(len(result["result"]), 4)
        self.assertEqual(result["result"][3], ", ".join(["0.00"] * 43))
        with tempfile.TemporaryDirectory() as directory:
            source_path = str(Path(directory) / "source.safetensors")
            save_file(self.make_lora("native"), source_path)
            result = node.load_lora(self.models[0], None, "source", 1.0, "All Off",
                                    lora_path_opt=source_path, save_refined_lora=True, save_path=directory)
            self.assertIs(result["result"][0], self.models[0])
            self.assertIn("All layers disabled", result["result"][2])
            self.assertEqual(len(list(Path(directory).glob("*.safetensors"))), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
