"""CPU-only accounting and integration tests; no checkpoint loading."""
import threading
import unittest
from types import SimpleNamespace as NS

import torch

from qwen_streaming import ChunkedCodec, StreamingCancelled, codec_context_frames, stream_custom_voice


class AccountingTests(unittest.TestCase):
    def make(self, chunk=3, context=2, callback=None, cancelled=lambda: False):
        self.chunks, self.windows = [], []
        def decode(codes):
            self.windows.append(codes.clone())
            return codes[:, :1].float().repeat_interleave(2, dim=-1)
        return ChunkedCodec(decode, groups=2, hop=2, sample_rate=24,
                            chunk_frames=chunk, context_frames=context,
                            on_audio=callback or self.chunks.append, cancelled=cancelled)

    def test_boundaries_tail_and_owned_storage(self):
        for count in (0, 1, 2, 3, 4, 6, 7, 20):
            for chunk in (1, 3):
                for context in (0, 2, 10):
                    with self.subTest(count=count, chunk=chunk, context=context):
                        codec = self.make(chunk, context)
                        for i in range(count):
                            frame = torch.tensor([[i, i]])
                            codec.push(frame)
                            frame.fill_(-1)
                            self.assertLessEqual(len(codec.buffer), context + chunk)
                        codec.finish()
                        codec.finish()
                        expected = torch.arange(count).float().repeat_interleave(2)
                        actual = torch.cat([c.samples for c in self.chunks]) if self.chunks else torch.empty(0)
                        torch.testing.assert_close(actual, expected)
                        self.assertEqual(codec.samples, count * 2)
                        offset = 0
                        for c in self.chunks:
                            self.assertEqual(c.start_sample, offset)
                            offset += c.samples.numel()
                        self.assertFalse(codec.buffer)

    def test_first_chunk_then_larger_chunks(self):
        codec = self.make(chunk=25, context=2)
        codec.first_chunk_frames = 1
        for i in range(53):
            codec.push(torch.tensor([[i, i]]))
        codec.finish()
        self.assertEqual([c.samples.numel() for c in self.chunks], [2, 50, 50, 4])
        self.assertEqual([c.start_sample for c in self.chunks], [0, 2, 52, 102])
        torch.testing.assert_close(torch.cat([c.samples for c in self.chunks]),
                                   torch.arange(53).float().repeat_interleave(2))

    def test_cancel_discards_tail(self):
        event = threading.Event()
        codec = self.make(cancelled=event.is_set)
        codec.push(torch.ones((1, 2), dtype=torch.long))
        event.set()
        with self.assertRaises(StreamingCancelled):
            codec.finish()
        self.assertFalse(self.chunks)

    def test_callback_and_decode_errors(self):
        def fail(_):
            raise LookupError("consumer")
        codec = self.make(chunk=1, callback=fail)
        with self.assertRaisesRegex(LookupError, "consumer"):
            codec.push(torch.ones((1, 2), dtype=torch.long))
        codec = self.make(chunk=1)
        codec.decode = lambda _: torch.zeros(1, 1, 1)
        with self.assertRaisesRegex(RuntimeError, "Unexpected codec shape"):
            codec.push(torch.ones((1, 2), dtype=torch.long))

    def test_context_bound(self):
        cfg = NS(sliding_window=72, layer_types=["sliding_attention"] * 8,
                 num_hidden_layers=8, upsample_rates=(8, 5, 4, 3), upsampling_ratios=(2, 2))
        self.assertGreater(codec_context_frames(cfg), 8 * 71)
        cfg.layer_types = ["full_attention"]
        with self.assertRaises(ValueError):
            codec_context_frames(cfg)


class Qwen3TTSTokenizerV2Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = NS(num_quantizers=2, sliding_window=1, layer_types=["sliding_attention"],
                         num_hidden_layers=1, upsample_rates=(2,), upsampling_ratios=(2,))
        self.total_upsample = 2
        self.pre_transformer = NS(rotary_emb=NS(rope_type="default"))

    def forward(self, codes):
        return codes[:, :1].float().repeat_interleave(2, dim=-1)


class FakeTalker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = NS(codec_eos_token_id=99)
        self.generation_config = NS(num_beams=1, num_return_sequences=1)
        self.seen = None

    def forward(self, frame):
        return NS(hidden_states=(None, frame))

    def generate(self, **kwargs):
        self.seen = kwargs
        self(None)
        for i in range(4):
            self(torch.tensor([[i, i]]))
        return torch.zeros(1)


def fake_model():
    talker = FakeTalker()
    decoder = Qwen3TTSTokenizerV2Decoder()
    decoder.eval()
    tokenizer = NS(training=False, decoder=decoder,
                   get_model_type=lambda: "qwen3_tts_tokenizer_12hz",
                   get_output_sample_rate=lambda: 24)
    core = NS(talker=talker, speech_tokenizer=NS(model=tokenizer),
              training=False, tts_model_type="custom_voice")
    def custom(**kwargs):
        talker.generate(output_hidden_states=True, return_dict_in_generate=True)
        raise AssertionError("Offline decode must not run")
    return NS(model=core, generate_custom_voice=custom)


class IntegrationTests(unittest.TestCase):
    def test_hook_and_restoration(self):
        model = fake_model()
        talker = model.model.talker
        chunks = []
        result = stream_custom_voice(model, text="x", speaker="y", on_audio=chunks.append, chunk_frames=3)
        self.assertEqual((result.frames, result.samples), (4, 8))
        self.assertEqual([c.samples.numel() for c in chunks], [6, 2])
        self.assertFalse(talker.seen["output_hidden_states"])
        self.assertNotIn("generate", talker.__dict__)
        self.assertFalse(talker._forward_hooks)
        self.assertFalse(talker._forward_pre_hooks)
        self.assertFalse(hasattr(talker, "_streaming_active"))

    def test_error_and_cancel_restore_instance_override(self):
        for cancel in (False, True):
            model = fake_model()
            talker = model.model.talker
            prior = talker.generate
            talker.generate = prior
            def fail(_):
                raise LookupError("consumer")
            with self.assertRaises(StreamingCancelled if cancel else LookupError):
                stream_custom_voice(model, text="x", speaker="y", on_audio=fail,
                                    chunk_frames=1, cancelled=lambda: cancel)
            self.assertIs(talker.generate, prior)
            self.assertFalse(talker._forward_hooks)
            self.assertFalse(hasattr(talker, "_streaming_active"))

    def test_reject_short_context_and_unsupported_options(self):
        for kwargs in ({"context_frames": 0}, {"streamer": object()}, {"text": ["x"]}):
            args = dict(text="x", speaker="y", on_audio=lambda _: None)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                stream_custom_voice(fake_model(), **args)


if __name__ == "__main__":
    unittest.main()
