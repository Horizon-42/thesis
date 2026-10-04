"""The live executor's drive of the single-flight executor (`ts_transformer.experiments.training_flights.fly_single`)
against the batched torch executor the formal replay and the Training export fly (`replay.fly_batch`): a synthetic
closed-loop sentence at Δ = 2, 4 and 8 s flown by both — the same cycles, the states apart by round-off only, the same
verdict — then stopped before a cycle, and stopped by a newer request. Synthetic flights only: nothing here opens an
artefact, a spec or the frontend's data (the spec's reference tracks check the single-flight executor on real flights,
`autopilot.conformance`; a published set is checked by `check_live`)."""

import unittest

import torch

import aeroviz_backend.autopilot_segment  # noqa: F401 — puts `ts_transformer` (under 4dTrajectory/) on the path
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.experiments import training_flights
from ts_transformer.tests.support import closed_loop_flight

CPU = torch.device("cpu")


class SingleAgainstBatchTest(unittest.TestCase):
    def test_a_closed_loop_sentence_flies_the_same_alone_as_in_the_batch_at_every_row_interval(self):
        for interval in (2.0, 4.0, 8.0):
            with self.subTest(interval=interval):
                one = closed_loop_flight(interval)
                batch, verdicts = replay.fly_batch(one.batch, one.params, one.words, device=CPU)
                alone, stopped = training_flights.fly_single(one.batch, one.inputs, 0, one.params, one.words)
                self.assertFalse(stopped)
                done = int(batch.done_cycle[0])
                self.assertEqual(int(alone.done_cycle[0]), done)
                apart = (alone.states[0, : done + 2] - batch.states[0, : done + 2]).abs().max()
                self.assertLess(float(apart), STATE_BOUND_M)
                for name in batch.modes:
                    self.assertTrue(torch.equal(alone.modes[name][0, : done + 1], batch.modes[name][0, : done + 1]), name)
                (judged,) = replay.judge_batch(one.batch, alone, one.words)
                self.assertEqual((judged.outcome, judged.end_row), (verdicts[0].outcome, verdicts[0].end_row))

    def test_a_flight_stopped_before_a_cycle_is_the_whole_flight_up_to_it(self):
        one = closed_loop_flight(4.0)
        whole, _ = training_flights.fly_single(one.batch, one.inputs, 0, one.params, one.words)
        part, stopped = training_flights.fly_single(one.batch, one.inputs, 0, one.params, one.words, stop_cycle=120)
        self.assertTrue(stopped)
        self.assertEqual(int(part.done_cycle[0]), 119)
        self.assertEqual(part.states.shape[1], 121)
        self.assertTrue(torch.equal(part.states[0], whole.states[0, :121]))
        # a stop past the flight's end does not stop it
        late, stopped = training_flights.fly_single(one.batch, one.inputs, 0, one.params, one.words, stop_cycle=10_000)
        self.assertFalse(stopped)
        self.assertTrue(torch.equal(late.states, whole.states))

    def test_a_newer_request_stops_the_flight_before_its_next_cycle(self):
        one = closed_loop_flight(2.0)
        asked = []

        def superseded():
            asked.append(1)
            return len(asked) > 5

        with self.assertRaisesRegex(InterruptedError, "superseded after 5 cycles"):
            training_flights.fly_single(one.batch, one.inputs, 0, one.params, one.words, superseded=superseded)


if __name__ == "__main__":
    unittest.main()
