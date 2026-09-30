import unittest
import torch
import json
from pathlib import Path
from dataclasses import replace
from research_v3.contracts import AlphabetSpec, Gate
from research_v3.data.generators import record_from_word
from research_v3.cli.probe_gpu_fullwidth_joint import _model
from research_v3.train.neural_refresh_solver_v13 import NeuralRefreshSolver


class NeuralRefreshContractTests(unittest.TestCase):
    def test_invalid_explicit_budgets_and_field_config_rejected(self):
        search = dict(beam_width=2, shortlist=2, max_depth=3,
                      inference_batch=2, max_expanded=30)
        field = dict(mc_samples=2, mc_seed=7, max_pair_states=30, head='qhat')
        for key in search:
            for value in (0, -1, True, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    NeuralRefreshSolver(torch.nn.Linear(1, 1),
                        search_config={**search, key: value}, field_config=field,
                        source_sha256={})
        for override in ({'mc_samples': True}, {'max_pair_states': 0},
                         {'mc_seed': 1.5}, {'head': 'hamming'}):
            with self.assertRaises(ValueError):
                NeuralRefreshSolver(torch.nn.Linear(1, 1), search_config=search,
                    field_config={**field, **override}, source_sha256={})


@unittest.skipUnless(torch.cuda.is_available(), 'Molab CUDA gate')
class NeuralRefreshCudaTests(unittest.TestCase):
    def test_full_word_and_training_modes_and_rng(self):
        config = json.loads((Path(__file__).resolve().parents[2] /
                             'full_training_config.json').read_text())
        model = _model(config).train()
        record = record_from_word(AlphabetSpec.all_to_all(8), (Gate(0, 1),),
                                  'train:refresh-cuda', 7, source='ordinary_walk_v1')
        record = replace(record, split='train')
        solver = NeuralRefreshSolver(model,
            search_config=dict(beam_width=2, shortlist=56, max_depth=1,
                               inference_batch=2, max_expanded=100),
            field_config=dict(mc_samples=config['diffusion']['mc_samples'],
                              mc_seed=config['feature_seed'],
                              max_pair_states=config['diffusion']['max_pair_states'],
                              head='qhat'), source_sha256={})
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        modes = [m.training for m in model.modules()]
        word = solver(record, 1)
        self.assertIsNotNone(word)
        self.assertEqual(word.physical_cost, 1)
        self.assertEqual(word.start, record.state)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))


if __name__ == '__main__': unittest.main()
