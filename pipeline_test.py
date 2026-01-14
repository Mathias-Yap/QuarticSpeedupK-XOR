from typing import Sequence
from kxor_code.algorithms.base_alg_step import NoOpStep, ProblemRecord, StepConfig, StepStats, StepContext, BaseAlgorithmStep
from kxor_code.algorithms.classical_eigenvalues_step import ClassicalEigenvaluesStep
from kxor_code.algorithms.compute_kikuchi_step import ComputeKikuchiStep
from kxor_code.algorithms.power_iteration_step import PowerIterationStep
from kxor_code.algorithms.quartic_quantum_algorithm_step import QuarticQuantumAlgorithm
from kxor_code.algorithms.threshold_step import ThresholdStep
from kxor_code.problem_set_generation.kxor_instance import KXORInstance
from kxor_code.algorithms.algorithm_pipeline import AlgorithmPipeline
from kxor_code.problem_set_generation.kxor_dataset_generator import KXORDatasetGenerator

step = ComputeKikuchiStep()
# step2 = QuarticQuantumAlgorithm()
step2 = ThresholdStep()
step3 = ClassicalEigenvaluesStep()
pipeline = AlgorithmPipeline(steps=[step, step2, step3], verbose=True)
generator = KXORDatasetGenerator()
instance_path = generator.generate_single_instance(folder_path="data", n=80, k=2, m=842, rho=1, seed=42)
instance = KXORInstance.load(instance_path)
problem = ProblemRecord(problem_id="test_problem", instance=instance)
problem.add_field("ell", 2)
problem.add_field("kappa", 1)
pipeline.run(problem)
print(problem)
problem.save_compact("data/problem_record_test")
problem.load_compact("data/problem_record_test.npz")
print("reloaded")
print(problem)