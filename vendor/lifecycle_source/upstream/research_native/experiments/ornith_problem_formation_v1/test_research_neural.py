"""Actual tiny HF neural integration, not full True or research capability."""
import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from .research_neural import (ResearchDecoderBlock, ResearchNeuralRuntime,
                              install_research_neural)


def tiny_research_model():
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=256,
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=1, linear_num_value_heads=2,
        layer_types=['linear_attention', 'full_attention'],
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': .5, 'mrope_section': [2, 1, 1]},
        pad_token_id=0, bos_token_id=1, eos_token_id=None)
    config._attn_implementation = 'eager'
    return Qwen3_5ForCausalLM(config).eval()


class ResearchNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(811)
        self.model = tiny_research_model()
        self.ids = torch.tensor([[1, 3, 7, 11]])

    def install(self):
        self.model.requires_grad_(False)
        self.runtime = install_research_neural(self.model, width=12, slots=3, layer=0)
        self.runtime.set_evidence([[2, 3, 9], [4, 6, 8]], ['first', 'second'])
        return self.runtime

    def logits(self, model=None):
        return (model or self.model)(self.ids, use_cache=False).logits

    def test_real_backbone_zero_bridge_exact_parity_and_registration(self):
        expected = self.logits().detach()
        old_ids = {id(p): p.requires_grad for p in self.model.parameters()}
        runtime = install_research_neural(self.model, width=12, slots=3, layer=0)
        runtime.set_evidence([[2, 3, 9], [4, 6, 8]], ['a', 'b'])
        with runtime.activate():
            self.assertTrue(torch.equal(expected, self.logits().detach()))
        names = dict(self.model.named_parameters())
        for name, p in runtime.components.named_parameters():
            self.assertIs(names['research_neural.' + name], p)
            self.assertTrue(p.requires_grad)
        for p in self.model.parameters():
            if id(p) in old_ids:
                self.assertEqual(p.requires_grad, old_ids[id(p)])
        self.assertIs(self.model.model.layers[0].research_components, self.model.research_neural)
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules()))

    def test_task_loss_updates_bridge_then_real_upstream_gradients(self):
        runtime = self.install()
        optimizer = torch.optim.AdamW(runtime.components.parameters(), lr=.01, weight_decay=0)
        with runtime.activate(training=True):
            loss = F.cross_entropy(self.logits()[0, :-1], self.ids[0, 1:])
        loss.backward()
        self.assertGreater(runtime.components.bridge.output.weight.grad.norm().item(), 0)
        self.assertEqual(runtime.components.bridge.value.weight.grad.norm().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        runtime.mark_updated()
        with runtime.activate(training=True):
            loss2 = F.cross_entropy(self.logits()[0, :-1], self.ids[0, 1:])
        loss2.backward()
        for parameter in (runtime.components.encoder.projection.weight,
                          runtime.components.core.proposal[0].weight,
                          runtime.components.bridge.value.weight):
            self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is None for name, p in self.model.named_parameters()
                            if not name.startswith('research_neural.')))
        self.assertEqual(runtime.binding.stack, [])

    def test_observation_loss_updates_encoder_core_and_outcome_without_bridge_regularizer(self):
        runtime = self.install()
        scores = runtime.score_candidates([[7, 8], [3, 5, 9]], training=True)
        F.cross_entropy(scores['outcome_logits'][0], torch.tensor([0, 1])).backward()
        for parameter in (runtime.components.encoder.projection.weight,
                          runtime.components.core.k.weight,
                          runtime.components.core.outcome_head[-1].weight):
            self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertIsNone(runtime.components.bridge.output.weight.grad)
        self.assertIsNone(runtime.components.core.action_score.weight.grad)

    def test_evidence_encoder_is_real_token_feature_projection_with_bridge_bypass(self):
        runtime = self.install()
        before = runtime.encode_only([[2, 3, 9], [9, 3, 2]])
        with torch.no_grad():
            runtime.components.bridge.output.weight.fill_(3.)
        with runtime.activate():
            after = runtime.encode_only([[2, 3, 9], [9, 3, 2]])
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(torch.equal(before[0], before[1]))
        with torch.no_grad():
            raw = self.model.model(torch.tensor([[2, 3, 9]]), use_cache=False).last_hidden_state.mean(1)
        expected = runtime.components.encoder(raw)
        self.assertTrue(torch.allclose(before[:1], expected, atol=1e-7, rtol=1e-6))

    def test_evidence_frontier_updates_once_not_per_decode_token_or_candidate(self):
        runtime = self.install()
        version = runtime.input_snapshot()
        with runtime.activate():
            self.logits()
            self.logits()
            runtime.score_candidates([[7], [9]])
        self.assertEqual(runtime.state_recomputations, 1)
        runtime.set_evidence([[2, 3, 9], [4, 6, 8]], ['first', 'second'])
        with runtime.activate():
            self.logits()
        self.assertEqual(runtime.state_recomputations, 1)
        self.assertEqual(runtime.input_snapshot(), version)
        runtime.set_evidence([[2, 3, 9], [4, 6, 8], [8, 9]], ['first', 'second', 'third'])
        with runtime.activate():
            self.logits()
        self.assertEqual(runtime.state_recomputations, 2)

    def test_frozen_snapshot_recomputation_and_nested_scopes_preserve_history(self):
        runtime = self.install()
        original = runtime.input_snapshot()
        runtime.set_evidence([[5, 6]], ['later'], concept_version='c2', source_version='new')
        current = runtime.input_snapshot()
        with runtime.activate() as outer:
            with runtime.use_snapshot(original, training=True) as inner:
                self.assertIsNotNone(inner.grad_fn)
                runtime.score_candidates([[7], [8]], training=True)
                self.assertFalse(torch.equal(outer, inner))
            self.assertIs(runtime.binding.stack[-1]['state'], outer)
        self.assertEqual(runtime.input_snapshot(), current)
        self.assertEqual(runtime.binding.stack, [])

    def test_explicit_empty_evidence_has_initial_state_and_no_fake_observation(self):
        runtime = install_research_neural(self.model, width=12, slots=3)
        with runtime.activate() as state:
            self.assertTrue(torch.equal(state, runtime.components.core.initial[None]))
            self.assertTrue(torch.isfinite(self.logits()).all())
        self.assertEqual(runtime.input_snapshot()['event_ids'], [])
        with self.assertRaisesRegex(ValueError, 'NO_USABLE'):
            runtime.score_candidates([])

    def test_invalid_rows_and_event_rebinding_fail_before_mutation(self):
        runtime = self.install()
        before = runtime.input_snapshot()
        for rows, ids in (([[64]], ['other']), ([[]], ['other']), ([[True]], ['other']),
                          ([[2]], ['first']), ([[2], [3]], ['same', 'same'])):
            with self.subTest(rows=rows, ids=ids), self.assertRaises(ValueError):
                runtime.set_evidence(rows, ids)
            self.assertEqual(runtime.input_snapshot(), before)
        with runtime.activate(), self.assertRaisesRegex(RuntimeError, 'active decode'):
            runtime.set_evidence([[4]], ['new'])

    def test_native_cache_rejected_including_after_z_change(self):
        runtime = self.install()
        with runtime.activate(), self.assertRaisesRegex(ValueError, 'RESEARCH_CACHE_FORBIDDEN'):
            self.model(self.ids, use_cache=True)
        original = tiny_research_model()
        cache = original(self.ids, use_cache=True).past_key_values
        runtime.set_evidence([[4]], ['new'])
        with runtime.activate(), self.assertRaisesRegex(ValueError, 'RESEARCH_CACHE_FORBIDDEN'):
            self.model(self.ids[:, -1:], past_key_values=cache, use_cache=False)

    def test_generation_uses_registered_bridge_full_prefix_without_reingest(self):
        runtime = self.install()
        with torch.no_grad():
            runtime.components.bridge.output.weight.normal_(std=.2)
        with runtime.activate(), torch.no_grad():
            actual = self.model.generate(self.ids, max_new_tokens=3, do_sample=False, use_cache=False)
        self.assertEqual(actual.shape, (1, 7))
        self.assertEqual(runtime.state_recomputations, 1)
        with runtime.activate():
            amended = self.logits()
        with runtime.activate(bypass=True):
            base = self.logits()
        self.assertFalse(torch.equal(amended, base))

    def test_inference_mode_outer_scope_retains_versioned_detached_state(self):
        runtime = self.install()
        with torch.inference_mode(), runtime.activate() as state:
            self.assertFalse(torch.is_inference(state))
            self.assertIsNone(state.grad_fn)
            self.logits()
        with torch.inference_mode(), runtime.activate():
            self.logits()
        self.assertEqual(runtime.state_recomputations, 1)

    def test_official_eos_direct_and_rescue_use_same_core_without_native_caches(self):
        from .freeform import EOSGenerator
        from .generation_rescue import RescueChannel
        from .runtime_features import configure_runtime_features
        from .test_runtime_features import TinyTransportTokenizer
        configure_runtime_features({'model': self.model})
        runtime = self.install()
        with torch.no_grad():
            runtime.components.bridge.output.weight.normal_(std=.1)
        generator = EOSGenerator(self.model, TinyTransportTokenizer(), seed=14, max_new_tokens=2)
        response = generator.generate([{'role': 'user', 'content': 'engineering fixture'}])
        self.assertEqual(len(response['generated_ids']), 2)
        self.assertEqual(response['research_neural']['status'], 'INSTALLED')
        self.assertEqual(response['cache_policy'], 'FULL_PREFIX_PER_TOKEN_NO_NATIVE_CACHE')
        self.assertEqual(response['precision_cache']['status'], 'DISABLED_RESEARCH_FULL_PREFIX_RECOMPUTATION')
        self.assertFalse(response['sampling_execution']['effective_generation_config']['use_cache'])
        self.assertEqual(runtime.binding.stack, [])
        self.assertEqual(runtime.state_recomputations, 1)
        rescue = RescueChannel()
        rescue.capture(generator)
        recovered = rescue.generate([{'role': 'user', 'content': 'rescue engineering fixture'}])
        self.assertEqual(recovered['research_neural']['status'], 'INSTALLED')
        self.assertEqual(recovered['cache_policy'], 'FULL_PREFIX_PER_TOKEN_NO_NATIVE_CACHE')
        self.assertEqual(runtime.state_recomputations, 1)

    def test_eos_preserves_explicit_frozen_preobservation_scope(self):
        from .freeform import EOSGenerator
        from .test_runtime_features import TinyTransportTokenizer
        runtime = self.install()
        old = runtime.input_snapshot()
        runtime.set_evidence([[9]], ['future'], concept_version='future')
        generator = EOSGenerator(self.model, TinyTransportTokenizer(), seed=14, max_new_tokens=1)
        with runtime.use_snapshot(old) as state:
            response = generator.generate([{'role': 'user', 'content': 'fixture'}])
            self.assertIs(runtime.binding.stack[-1]['state'], state)
            self.assertEqual(response['research_neural']['state_version'], runtime.version(old))
        self.assertEqual(runtime.input_snapshot()['event_ids'], ['future'])

    def test_mutated_weights_rebuild_inference_state_and_reject_mid_decode(self):
        runtime = self.install()
        with runtime.activate() as first:
            self.logits()
        with torch.no_grad():
            runtime.components.encoder.projection.weight.add_(.01)
        with runtime.activate() as second:
            self.assertFalse(torch.equal(first, second))
            with torch.no_grad():
                runtime.components.bridge.output.weight.add_(.01)
            with self.assertRaisesRegex(RuntimeError, 'STALE_DURING'):
                self.logits()
        self.assertEqual(runtime.state_recomputations, 2)

    def test_parameter_replacement_and_source_or_concept_version_rebuild_state(self):
        runtime = self.install()
        with runtime.activate():
            self.logits()
        old = runtime.components.encoder.projection.weight
        runtime.components.encoder.projection.weight = torch.nn.Parameter(old.detach().clone())
        with runtime.activate():
            self.logits()
        runtime.set_evidence([[2, 3, 9], [4, 6, 8]], ['first', 'second'], concept_version='new')
        with runtime.activate():
            self.logits()
        self.assertEqual(runtime.state_recomputations, 3)

    def test_exposed_state_cannot_silently_mutate_mid_decode_or_inference_cache(self):
        runtime = self.install()
        with runtime.activate() as state:
            original = state.clone()
            state.add_(.25)
            with self.assertRaisesRegex(RuntimeError, 'STALE_DURING'):
                self.logits()
            with self.assertRaisesRegex(RuntimeError, 'STALE_DURING'):
                runtime.score_candidates([[7]])
        with runtime.activate() as next_state:
            self.assertTrue(torch.equal(original, next_state))

    def test_context_exception_cleanup_and_active_checkpoint_rejection(self):
        runtime = self.install()
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with runtime.activate(training=True):
                with self.assertRaisesRegex(RuntimeError, 'ACTIVE_CONTEXT_NOT_CHECKPOINT'):
                    pickle.dumps(runtime)
                raise RuntimeError('intentional')
        self.assertEqual(runtime.binding.stack, [])
        loaded = pickle.loads(pickle.dumps(runtime))
        self.assertEqual(loaded.binding.stack, [])
        self.assertIsNone(loaded._inference_key)

    def test_deepcopy_candidate_has_independent_parameters_and_no_active_context(self):
        runtime = self.install()
        with runtime.activate(training=True):
            candidate = copy.deepcopy(self.model)
        wrapper = candidate.model.layers[0]
        self.assertIsInstance(wrapper, ResearchDecoderBlock)
        self.assertEqual(wrapper.binding.stack, [])
        self.assertIs(wrapper.research_components, candidate.research_neural)
        self.assertIsNot(candidate.research_neural.bridge.output.weight,
                         runtime.components.bridge.output.weight)
        candidate_runtime = ResearchNeuralRuntime(candidate, wrapper.binding, 0)
        candidate_runtime.set_evidence([[2, 3, 9], [4, 6, 8]], ['first', 'second'])
        with candidate_runtime.activate(training=True):
            self.logits(candidate).sum().backward()
        self.assertIsNone(runtime.components.bridge.output.weight.grad)
        self.assertGreater(candidate.research_neural.bridge.output.weight.grad.norm().item(), 0)

    def test_cold_pickle_next_optimizer_update_matches_uninterrupted(self):
        runtime = self.install()
        optimizer = torch.optim.AdamW(runtime.components.parameters(), lr=.005)
        def update():
            optimizer.zero_grad(set_to_none=True)
            with runtime.activate(training=True):
                F.cross_entropy(self.logits()[0, :-1], self.ids[0, 1:]).backward()
            optimizer.step()
            runtime.mark_updated()
        update()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'payload.pkl'
            path.write_bytes(pickle.dumps({'runtime': runtime, 'optimizer': optimizer,
                                          'rng': torch.get_rng_state(), 'ids': self.ids}))
            update()
            expected = {k: v.detach().cpu() for k, v in runtime.components.state_dict().items()}
            code = '''import pickle,sys,torch
from pathlib import Path
from torch.nn import functional as F
torch.set_num_threads(1)
payload=pickle.loads(Path(sys.argv[1]).read_bytes())
r=payload['runtime']; o=payload['optimizer']; ids=payload['ids']
torch.set_rng_state(payload['rng'])
o.zero_grad(set_to_none=True)
with r.activate(training=True):
    F.cross_entropy(r.model(ids,use_cache=False).logits[0,:-1],ids[0,1:]).backward()
o.step(); r.mark_updated()
assert r.model.model.layers[0].research_components is r.components
assert r.model.model.layers[0].binding is r.binding
torch.save(r.components.state_dict(),sys.argv[2])
print(r.weight_revision)
'''
            output = Path(tmp) / 'next.pt'
            result = subprocess.run([sys.executable, '-c', code, str(path), str(output)],
                                    capture_output=True, text=True, encoding='utf-8', timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            actual = torch.load(output, weights_only=True)
            self.assertEqual(int(result.stdout.strip().splitlines()[-1]), runtime.weight_revision)
            for key, tensor in expected.items():
                self.assertTrue(torch.equal(tensor, actual[key]), key)

    def test_actual_conditional_qwen_text_route_and_installation_receipt(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5VisionConfig
        text = self.model.config.to_dict()
        vision = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=48,
            num_heads=4, in_channels=3, patch_size=16, temporal_patch_size=2,
            spatial_merge_size=2, out_hidden_size=32, num_position_embeddings=2304)
        config = Qwen3_5Config(text_config=text, vision_config=vision)
        config._attn_implementation = 'eager'
        model = Qwen3_5ForConditionalGeneration(config).eval()
        runtime = install_research_neural(model, width=12, slots=3, layer=0)
        runtime.set_evidence([[2, 3], [4, 8]], ['a', 'b'])
        with runtime.activate(training=True):
            F.cross_entropy(model(self.ids, use_cache=False).logits[0, :-1], self.ids[0, 1:]).backward()
        self.assertGreater(runtime.components.bridge.output.weight.grad.norm().item(), 0)
        receipt = runtime.installation_receipt()
        self.assertEqual(receipt['status'], 'INSTALLED')
        self.assertFalse(receipt['encoder_backbone_gradients'])
        self.assertEqual(receipt['state_version']['event_index'], 2)
        json.dumps(receipt)
        with self.assertRaisesRegex(ValueError, 'already installed'):
            install_research_neural(model)

    def test_production_checkpoint_isolate_preserves_runtime_alias_and_again_updates(self):
        from .continuation import Ancestor, Checkpoints
        runtime = self.install()
        optimizer = torch.optim.AdamW(runtime.components.parameters(), lr=.01)
        with runtime.activate(training=True):
            self.logits().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        runtime.mark_updated()
        with tempfile.TemporaryDirectory() as tmp:
            store = Checkpoints(Path(tmp) / 'store', Ancestor(self.model, {'fixture': 'research-neural'}))
            receipt = store.save({'model': self.model, 'runtime': runtime, 'optimizer': optimizer})
            restored = store.load(checkpoint_id=receipt['manifest'], ancestor_policy='isolate')
            other, model, opt = restored['runtime'], restored['model'], restored['optimizer']
            self.assertIs(other.model, model)
            self.assertIs(model.model.layers[0].binding, other.binding)
            self.assertIs(model.model.layers[0].research_components, model.research_neural)
            before = runtime.components.bridge.output.weight.detach().clone()
            self.assertNotEqual(before.data_ptr(), other.components.bridge.output.weight.data_ptr())
            with other.activate(training=True):
                self.logits(model).sum().backward()
            opt.step()
            other.mark_updated()
            self.assertTrue(torch.equal(before, runtime.components.bridge.output.weight))
            self.assertFalse(torch.equal(before, other.components.bridge.output.weight))


if __name__ == '__main__':
    unittest.main()
