"""CPU integration checks for both Darcy tasks, not accuracy benchmarks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import uuid

import numpy as np
import torch
from torch import nn

from operatorbench.data.datasets import load_darcy_dataset
from operatorbench.data.preprocessing import fit_task_normalization, select_input_field
from operatorbench.models.deeponet import DeepONet2d
from scripts import generate_darcy_splits as generate
from scripts import train_darcy as train
from scripts import evaluate_darcy as evaluate
from scripts import export_darcy_report as exporter

ROOT=Path(__file__).resolve().parents[1]


def invoke(module, *args):
    with mock.patch.object(sys, "argv", [module.__file__, *map(str,args)]):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            module.main()
    return output.getvalue()


class DarcyWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory()
        cls.work=Path(cls.folder.name)
        cls.run_dirs=[]
        cls.cases={}
        for task,filename in (("a_to_u","darcy_data_a_to_u.json"),("f_to_u","darcy_data_f_to_u.json")):
            settings=json.loads((ROOT/"configs"/filename).read_text())
            settings["N"]=17
            settings["dataset"]=str(cls.work/f"{task}.npz")
            for split,count in (("train",4),("val",2),("test",3)):
                settings["splits"][split]["n_samples"]=count
            config=cls.work/f"{task}_data.json"
            config.write_text(json.dumps(settings))
            invoke(generate,"--config",config)
            data,metadata=load_darcy_dataset(settings["dataset"])
            cls.cases[task]=(settings,config,data,metadata)

    @classmethod
    def tearDownClass(cls):
        for path in cls.run_dirs:
            resolved=path.resolve()
            # Remove only test runs created here, inside this checkout's outputs.
            if resolved.parent != (ROOT/"outputs").resolve() or not resolved.name.startswith("workflow_check_"):
                raise RuntimeError("Unexpected cleanup path.")
            if resolved.exists():
                shutil.rmtree(resolved)
        cls.folder.cleanup()

    def test_fixed_and_varying_fields_match_task(self):
        for task,(_,_,data,_) in self.cases.items():
            if task=="a_to_u":
                self.assertTrue(np.isin(data["A_train"],[3,12]).all())
                self.assertTrue(np.all(data["F_train"]==1))
            else:
                self.assertTrue(np.all(data["A_train"]==1))
                self.assertFalse(np.array_equal(data["F_train"][0],data["F_train"][1]))
                np.testing.assert_array_equal(select_input_field(data,task,"train"),data["F_train"])

    def test_cache_is_loaded_without_solving_again(self):
        for task,(settings,config,_,_) in self.cases.items():
            before=Path(settings["dataset"]).read_bytes()
            with mock.patch.object(generate,"build_dataset",side_effect=AssertionError("Unexpected solve")):
                log=invoke(generate,"--config",config)
            self.assertIn("without regenerating",log)
            self.assertEqual(Path(settings["dataset"]).read_bytes(),before)

    def test_mismatched_cache_configuration_is_rejected(self):
        settings,_,_,_=self.cases["f_to_u"]
        changed=copy.deepcopy(settings)
        changed["source"]["amplitude"]=0.5
        file=self.work/"changed_config.json"
        file.write_text(json.dumps(changed))
        before=Path(settings["dataset"]).read_bytes()
        with self.assertRaisesRegex(ValueError,"metadata does not match"):
            invoke(generate,"--config",file)
        self.assertEqual(Path(settings["dataset"]).read_bytes(),before)

    def test_only_training_fields_fit_normalization(self):
        _,_,data,_=self.cases["f_to_u"]
        before=fit_task_normalization(data,"f_to_u")
        changed=dict(data)
        changed["F_val"]=data["F_val"]*np.float32(100)
        changed["F_test"]=data["F_test"]*np.float32(-50)
        changed["U_test"]=data["U_test"]*np.float32(700)
        self.assertEqual(before,fit_task_normalization(changed,"f_to_u"))
        self.assertEqual(before["input_mean"],np.float32(data["F_train"].mean()))

    def test_branch_activation_changes_only_branch_hidden_layers(self):
        _,_,data,_=self.cases["f_to_u"]
        coords,_=train.make_deeponet_coords(data["x"],data["y"])
        params=dict(n_sensors=289,latent_dim=8,branch_hidden=[16,16],trunk_hidden=[16],coords01=torch.from_numpy(coords))
        torch.manual_seed(2026)
        original=DeepONet2d(**params)
        torch.manual_seed(2026)
        gelu=DeepONet2d(**params,branch_activation="gelu")
        for name,value in original.state_dict().items():
            torch.testing.assert_close(value,gelu.state_dict()[name],rtol=0,atol=0)
        self.assertTrue(any(isinstance(layer,nn.GELU) for layer in gelu.branch.net))
        self.assertFalse(any(isinstance(layer,nn.GELU) for layer in gelu.trunk.net))

    def test_both_tasks_train_resume_evaluate_all_samples(self):
        for task,(settings,_,data,metadata) in self.cases.items():
            baseline=json.loads((ROOT/"configs"/"darcy_baseline.json").read_text())
            baseline.update(dataset=settings["dataset"],N=17,task=task,
                            source=settings["source"],coefficient=settings["coefficient"],reference=settings["reference"])
            baseline["split_sizes"]={s:v["n_samples"] for s,v in settings["splits"].items()}
            baseline["split_seeds"]={s:v["seed"] for s,v in settings["splits"].items()}
            baseline["models"]={
                "deeponet":{"latent_dim":8,"branch_hidden":[16,16],"trunk_hidden":[16],
                            "use_hard_dirichlet":False,"branch_activation":"gelu" if task=="f_to_u" else "tanh"},
                "fno":{"modes1":4,"modes2":4,"width":8},
            }
            baseline["training"].update(num_steps=2,batch_size=2,validate_every=1,eval_batch_size=2)
            config=self.work/f"{task}_train.json"
            config.write_text(json.dumps(baseline))
            paths={}
            for model in ("deeponet","fno"):
                run_name=f"workflow_check_{uuid.uuid4().hex}"
                run_dir=ROOT/"outputs"/run_name
                self.run_dirs.append(run_dir)
                args=["--config",config,"--model",model,"--device","cpu","--run-name",run_name]
                if model=="deeponet":
                    invoke(train,*args,"--stop-after","1")
                    invoke(train,*args,"--resume")
                else:
                    invoke(train,*args)
                summary=json.loads((run_dir/"summary.json").read_text())
                self.assertEqual(summary["completed_steps"],2)
                self.assertTrue(summary["completed"])
                if model=="deeponet":self.assertEqual(summary["resume_count"],1)
                paths[model]=run_dir/"best.pt"
            result=self.work/f"{task}_evaluation"
            log=invoke(evaluate,"--device","cpu","--batch-size","2",
                       "--deeponet-checkpoint",paths["deeponet"],"--fno-checkpoint",paths["fno"],
                       "--dataset",settings["dataset"],"--output-dir",result,"--sample-indices","2")
            self.assertIn("Darcy test evaluation completed",log)
            summary=json.loads((result/"summary.json").read_text())
            self.assertEqual(summary["task"],task)
            self.assertEqual(summary["figure_sample_indices"],[2])
            self.assertEqual([r["n_test"] for r in summary["metrics"]],[3,3])
            self.assertEqual(len(list(result.glob("sample_*.png"))),1)
            export_root=self.work/f"{task}_public_report"
            with mock.patch.object(exporter,"__file__",str(export_root/"scripts/export_darcy_report.py")):
                invoke(exporter,"--result-dir",result,"--report-name","workflow_report")
            public=export_root/"docs/results/workflow_report/metadata.json"
            self.assertEqual(json.loads(public.read_text())["task"],task)
            figures=export_root/"docs/figures/workflow_report"
            self.assertEqual([p.name for p in figures.glob("*.png")],["sample_002.png"])
            with np.load(result/"predictions.npz") as cached:
                for model,key in (("DeepONet","U_deeponet"),("FNO","U_fno")):
                    diff=cached[key].astype(np.float64)-data["U_test"].astype(np.float64)
                    manual=(np.linalg.norm(diff.reshape(3,-1),axis=1)/np.linalg.norm(data["U_test"].astype(np.float64).reshape(3,-1),axis=1)).mean()
                    row=next(r for r in summary["metrics"] if r["model"]==model)
                    self.assertAlmostEqual(row["test_relative_l2_mean"],manual,places=7)
            saved=evaluate.load_best(paths["deeponet"],"DeepONet")
            saved["run_info"]["code_sha256"]["operatorbench/data/preprocessing.py"]="wrong"
            with self.assertRaisesRegex(ValueError,"Inference source differs"):
                evaluate.check_compatibility(saved,data,metadata,train.file_sha256(settings["dataset"]),ROOT,"deeponet")

            self.check_rejects_bad_fixed_fields(task, data, metadata, config, paths)
            wrong_config = self.work / f"{task}_wrong_grid.json"
            wrong_config.write_text(json.dumps({**baseline, "N": 19}))
            with self.assertRaisesRegex(ValueError, "does not match the configuration"):
                invoke(train, "--config", wrong_config, "--model", "deeponet",
                       "--device", "cpu", "--run-name", f"workflow_check_{uuid.uuid4().hex}")

            changed_targets = self.work / f"{task}_changed_targets.npz"
            arrays = {**data, "U_test": data["U_test"].copy()}
            arrays["U_test"][0, 1, 1] += 1
            np.savez_compressed(changed_targets, **arrays, metadata_json=json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Dataset SHA-256 differs"):
                invoke(evaluate, "--device", "cpu", "--dataset", changed_targets,
                       "--deeponet-checkpoint", paths["deeponet"], "--fno-checkpoint", paths["fno"],
                       "--output-dir", self.work / f"{task}_rejected")

    def check_rejects_bad_fixed_fields(self, task, data, metadata, config, paths):
        field = "F" if task == "a_to_u" else "A"
        for split in ("train", "val", "test"):
            key = f"{field}_{split}"
            for uniform in (False, True):
                with self.subTest(task=task, split=split, uniform=uniform):
                    arrays = dict(data)
                    arrays[key] = data[key].copy()
                    if uniform:
                        arrays[key] += 1
                    else:
                        arrays[key][0, 1, 1] += 1
                    dataset = self.work / f"{task}_{split}_{uniform}.npz"
                    np.savez_compressed(dataset, **arrays, metadata_json=json.dumps(metadata))
                    name = f"workflow_check_{uuid.uuid4().hex}"
                    self.run_dirs.append(ROOT / "outputs" / name)
                    with self.assertRaisesRegex(ValueError, rf"{key}: fixed field does not match"):
                        if split == "test":
                            invoke(evaluate, "--device", "cpu", "--dataset", dataset,
                                   "--deeponet-checkpoint", paths["deeponet"], "--fno-checkpoint", paths["fno"],
                                   "--output-dir", self.work / name)
                        else:
                            invoke(train, "--device", "cpu", "--dataset", dataset,
                                   "--config", config, "--model", "deeponet", "--run-name", name)


if __name__=="__main__":
    unittest.main()
