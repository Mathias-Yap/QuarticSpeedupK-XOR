from typing import Sequence
from kxor_code.algorithms.base_alg_step import NoOpStep, ProblemRecord, StepConfig, StepStats, StepContext, BaseAlgorithmStep
from kxor_code.algorithms.classical_eigenvalues_step import ClassicalEigenvaluesStep
from kxor_code.algorithms.compute_kikuchi_step import ComputeKikuchiStep
from kxor_code.problem_set_generation.kxor_instance import KXORInstance
from kxor_code.algorithms.algorithm_pipeline import AlgorithmPipeline
from kxor_code.problem_set_generation.kxor_dataset_generator import KXORDatasetGenerator
step = ComputeKikuchiStep()
step2 = ClassicalEigenvaluesStep()
step3 = NoOpStep()
pipeline = AlgorithmPipeline(steps=[step, step2, step3], verbose=True)
generator = KXORDatasetGenerator()
instance = generator.generate_single_instance(folder_path="data/problem_instances", n=400, k=9, m=1000, rho=0.2, seed=42)
instance = KXORInstance.load("data/problem_instances/kxor_instance_id2_n3_k2_m1_rho0.2.npz")
problem = ProblemRecord(problem_id="test_problem", instance=instance)
problem.add_field("ell", 2)
pipeline.run(problem)