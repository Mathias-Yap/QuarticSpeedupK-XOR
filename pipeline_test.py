"""Manual pipeline smoke script.

This file is *not* a pytest test module.
It lives at repo root for convenience, but we guard execution so `pytest` doesn't
run it during collection.
"""

from kxor_code.algorithms.algorithm_pipeline import AlgorithmPipeline
from kxor_code.algorithms.compute_kikuchi_step import ComputeKikuchiStep
from kxor_code.algorithms.quartic_quantum_algorithm_step import QuarticQuantumAlgorithm
from kxor_code.algorithms.key_extraction_step import KeyExtractionStep
from kxor_code.problem_set_generation.kxor_dataset_generator import KXORDatasetGenerator
from kxor_code.problem_set_generation.kxor_instance import KXORInstance
from kxor_code.algorithms.base_alg_step import ProblemRecord


def main() -> None:
	step = ComputeKikuchiStep()
	step2 = QuarticQuantumAlgorithm()
	step3 = KeyExtractionStep()
	pipeline = AlgorithmPipeline(steps=[step, step2, step3], verbose=True)

	generator = KXORDatasetGenerator()
	instance_path = generator.generate_single_instance(
		folder_path="data/problem_instances", n=5, k=2, m=5, rho=0.8, seed=42
	)
	instance = KXORInstance.load(instance_path)
	problem = ProblemRecord(problem_id="test_problem", instance=instance)
	problem.add_field("ell", 2)
	problem.add_field("threshold", 0.5)
	pipeline.run(problem)
	print("x_hat:", problem.get_field("x_hat"))


if __name__ == "__main__":
	main()